"""Background host-cache encoding and authenticated packed-KV recovery."""
import ctypes,json,math,os,time,threading
from concurrent.futures import ThreadPoolExecutor
import torch
from .npu_predict import _register_host_range
from .optimized_decode import decode_into


def settings():
    path=os.environ.get('KV_CODEC_CONFIG','')
    cfg=json.loads(open(path).read())if path else{}
    mode=cfg.get('codec',os.environ.get('KV_CODEC','none'))
    pipeline=bool(cfg.get('pipeline',int(os.environ.get('KV_RECOVERY_PIPELINE','1'))))
    if mode not in ('none','int8','int4','mixed'):raise ValueError(mode)
    return mode,pipeline


def layer_bits(mode,layer,layers):
    return 8 if mode=='int8' or(mode=='mixed'and layer<math.ceil(layers/3))else 4


def encode(source,indices,bits,kind,layer):
    b=len(indices)
    x=source.index_select(0,indices).float()
    if x.ndim!=4 or x.shape[1]!=128:raise ValueError(tuple(x.shape))
    _,t,h,d=x.shape;g=64
    shape=(b,t//g,g,h,d)if kind==0 else(b,t,h,d//g,g)
    x=x.reshape(shape)
    axis=2 if kind==0 else 4
    low=x.amin(axis,keepdim=True)
    high=x.amax(axis,keepdim=True)
    scale=((high-low)/((1<<bits)-1)).clamp_min(1e-12)
    q=((x-low)/scale).round().clamp(0,(1<<bits)-1).reshape(b,-1)
    if bits==4:q=q[:,0::2]+16*q[:,1::2]
    q=q.to(torch.uint8);qb=q.shape[1];sb=scale[0].numel()*4
    width=(qb+2*sb+16+15)//16*16
    packet=torch.zeros((b,width),dtype=torch.uint8)
    packet[:,:qb]=q
    packet[:,qb:qb+sb]=scale.contiguous().view(torch.uint8).reshape(b,-1)
    packet[:,qb+sb:qb+2*sb]=low.contiguous().view(torch.uint8).reshape(b,-1)
    packet[:,qb+2*sb:qb+2*sb+16]=torch.tensor([1,layer,kind,bits],dtype=torch.int32).view(torch.uint8)
    return packet,qb,sb


class EncodedStore:
    def __init__(self,handler):
        self.handler=handler
        self.lock=threading.RLock()
        self.io_lock=threading.RLock()
        self.retire=ctypes.CDLL(None).secure_infer_retire_transfer_prefetches
        self.retire.argtypes=[];self.retire.restype=None
        self.versions=[0]*handler.cpu_tensors[0][0].shape[0]
        self.arrays={};self.ready={};self.jobs={};self.futures={};self.held={}
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='kv-encode')

    def submit(self,job,spec):
        from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
        if isinstance(spec[0],GPULoadStoreSpec):
            assert self.handler.block_size_factor==1
            ids=[int(x)for x in spec[1].block_ids]
            mode,_=settings()
            with self.lock:
                for i in ids:self.versions[i]+=1
                versions=tuple(self.versions[i]for i in ids)
                self.jobs[job]=(ids,versions,mode,time.monotonic_ns())
        return self.raw_submit(job,spec)

    def encode_job(self,job):
        ids,versions,mode,queued=self.jobs[job]
        idx=torch.tensor(ids,dtype=torch.long)
        started=time.monotonic_ns()
        layers=len(self.handler.cpu_tensors)
        for layer,components in enumerate(self.handler.cpu_tensors):
            bits=layer_bits(mode,layer,layers)
            for kind,source in enumerate(components):
                packet,qb,sb=encode(source,idx,bits,kind,layer)
                key=(layer,kind,bits)
                with self.io_lock,self.lock:
                    if tuple(self.versions[i]for i in ids)!=versions:
                        raise RuntimeError('CPU cache slots changed during asynchronous encoding')
                    if key not in self.arrays:
                        cpu=torch.zeros((len(self.versions),packet.shape[1]),dtype=torch.uint8,pin_memory=True)
                        _register_host_range(cpu,layer,kind)
                        self.arrays[key]=(cpu,qb,sb)
                        self.ready[key]=[None]*len(self.versions)
                    cpu,_,_=self.arrays[key]
                    self.retire()
                    cpu.index_copy_(0,idx,packet)
                    self.retire()
                    for i,version in zip(ids,versions):self.ready[key][i]=version
        from .native_layerwise import log
        log('async_encode.jsonl',dict(job=job,codec=mode,blocks=len(ids),
            encode_cpu_ms=(time.monotonic_ns()-started)/1e6,
            store_to_encoded_ready_ms=(time.monotonic_ns()-queued)/1e6,
            encoded_before_cache_publication=True))

    def launch(self,job):
        with self.lock:
            if job not in self.futures:self.futures[job]=self.pool.submit(self.encode_job,job)
            return self.futures[job]

    def finished(self):
        results=[]
        for result in self.raw_finished():
            job=result.job_id if hasattr(result,'job_id')else result[0]
            entry=self.jobs.get(job)
            if entry is None or entry[2]=='none':
                self.jobs.pop(job,None)
                results.append(result)
            else:
                self.held[job]=result
                self.launch(job)
        for job,result in list(self.held.items()):
            if self.futures[job].done():
                self.futures.pop(job).result()
                self.jobs.pop(job)
                self.held.pop(job)
                results.append(result)
        return results

    def wait(self,jobs):
        self.raw_wait(jobs)
        for job in jobs:
            if job in self.jobs and self.jobs[job][2]!='none':self.launch(job).result()

    def get(self,layer,kind,bits,ids):
        key=(layer,kind,bits)
        with self.lock:
            if key not in self.arrays:raise RuntimeError('No encoded cache; reload must never encode')
            if any(self.ready[key][int(i)]!=self.versions[int(i)]for i in ids):
                raise RuntimeError('Encoded cache missing or stale; reload must never encode')
            return self.arrays[key]


def install_store(handler):
    store=EncodedStore(handler)
    store.raw_submit=handler.transfer_async
    store.raw_finished=handler.get_finished
    store.raw_wait=handler.wait
    handler.transfer_async=store.submit
    handler.get_finished=store.finished
    handler.wait=store.wait
    return store


class PackedTransfer:
    def __init__(self,original,store):
        self.original=original;self.store=store
        self.buffers={};self.temporary_buffers={};self.mode='none';self.request_id='';self.layers=40;self.records=[]
        self.wait_auth=ctypes.CDLL(None).secure_infer_recovery_wait_layer_authentication
        self.wait_auth.argtypes=[ctypes.c_uint32];self.wait_auth.restype=ctypes.c_int

    def transfer(self,payload,destinations):
        old=self.original
        if self.mode=='none':return self.raw_transfer(payload,destinations)
        host,mapping=payload;layer=old.layer;b=len(mapping)
        bits=layer_bits(self.mode,layer,self.layers)
        started=time.monotonic_ns()
        local_mapping=torch.stack((mapping[:,0],torch.arange(b)),dim=1).long()
        packets=[]
        for kind,destination in enumerate(destinations):
            cpu,qb,sb=self.store.get(layer,kind,bits,mapping[:,0].tolist())
            key=(layer,kind,bits,b)
            if key not in self.buffers:
                self.buffers[key]=torch.empty((b,cpu.shape[1]),dtype=torch.uint8,device=destination.device)
            packets.append((cpu,self.buffers[key],qb,sb,destination))
        indices=mapping[:,1].long().to(device=destinations[0].device)
        old.retained.extend([local_mapping,indices])
        assert old.begin(2*b)==0
        try:
            for cpu,gpu,*_ in packets:torch.ops._C_ascend.swap_blocks(cpu,gpu,local_mapping)
        finally:assert old.end()==0
        ready=torch.npu.Event();ready.record(torch.npu.current_stream());ready.synchronize()
        assert self.wait_auth(layer)==0,'Packed KV authentication failed'
        decode_start=time.monotonic_ns()
        for kind,(cpu,gpu,qb,sb,destination)in enumerate(packets):
            # Header fields never select allocation/layout: the trusted cache
            # index supplies geometry. Entire packet remains GCM-authenticated.
            key=(kind,b)
            if key not in self.temporary_buffers:
                self.temporary_buffers[key]=torch.empty(b*destination[0].numel(),dtype=torch.float32,device=gpu.device)
            decode_into(gpu,destination,indices,bits,kind,qb,sb,temporary=self.temporary_buffers[key])
        # The caller records the completion event after these kernels. The
        # authentication observer releases attention only when it completes.
        self.records.append(dict(layer=layer,bits=bits,blocks=b,
            original_bytes=sum(b*x[0].numel()*x.element_size()for x in host),
            wire_bytes=sum(b*cpu.shape[1]for cpu,*_ in packets),
            encode_cpu_ms=0.,reload_reencoded=False,optimized_decode=True,
            decode_host_submit_ms=(time.monotonic_ns()-decode_start)/1e6,
            host_elapsed_ms=(time.monotonic_ns()-started)/1e6))
        old.layer+=1

    def flush(self):
        if not self.records:return
        from .native_layerwise import log
        log('codec_metrics.jsonl',dict(request_id=self.request_id,codec=self.mode,layers=self.records,
            original_bytes=sum(r['original_bytes']for r in self.records),
            wire_bytes=sum(r['wire_bytes']for r in self.records),encode_cpu_ms=0.))
        self.records=[]


def attach(ops,store):
    from .pipelined_codec import PipelinedTransfer
    adapter=PipelinedTransfer(ops,store)
    adapter.raw_transfer=ops.single_layer_kv_transfer
    ops.single_layer_kv_transfer=adapter.transfer
    return adapter
