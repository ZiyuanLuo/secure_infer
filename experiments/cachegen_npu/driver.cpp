#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_mm.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

extern "C" int secure_infer_aicpu_gcm_verify_only_aad_do(void*,void*,void*,void*,void*,void*,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,void*,uint64_t);
extern "C" int secure_infer_aicpu_aes_ctr_do(void*,void*,void*,void*,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t);
extern "C" int entropy_restore_do(void*,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t);

struct Entry {
    void *host,*cipher,*key,*hash,*tag,*status,*aad,*plain,*restored,*scratch,*answer;
    uint64_t bytes;
    uint32_t tiles,tile,bits,elements,iv[4];
};

extern "C" int run_entropy_pipeline(Entry* rows,int count,void* query,void* a_stream,
    void* d_stream,void* c_stream,int pipeline,int repeats,int lanes,double* times,float* trace)
{
    using Clock=std::chrono::steady_clock;
    auto outer=Clock::now();
    auto a=reinterpret_cast<aclrtStream>(a_stream);
    auto* decode_streams=reinterpret_cast<aclrtStream*>(d_stream);
    auto d=decode_streams[0];
    auto c=reinterpret_cast<aclrtStream>(c_stream);
    std::vector<aclrtEvent> ae(count),de(count);
    std::vector<aclTensor*> tensors;
    std::vector<aclOpExecutor*> ops(count*repeats);
    std::vector<uint64_t> sizes(count*repeats);
    std::vector<aclrtEvent> marks(trace ? count*6 : 0);
    aclrtEvent origin=nullptr;
    void* workspace=nullptr;
    uint64_t max_workspace=0;
    auto check=[](int code,const char*where){
        if(code) std::fprintf(stderr,"native driver: %s returned %d\n",where,code);
        return code;
    };
    for(int i=0;i<count;++i){
        auto&e=rows[i];
        int rc=aclrtCreateEvent(&ae[i]);if(check(rc,"create A event"))return rc;
        rc=aclrtCreateEvent(&de[i]);if(check(rc,"create D event"))return rc;
        int64_t qshape[]={128,128},qstride[]={128,1};
        int64_t kshape[]={128,e.elements/128},kstride[]={1,128};
        int64_t kstorage[]={e.elements/128,128};
        int64_t oshape[]={128,e.elements/128},ostride[]={e.elements/128,1};
        auto*q=aclCreateTensor(qshape,2,ACL_BF16,qstride,0,ACL_FORMAT_ND,qshape,2,query);
        auto*k=aclCreateTensor(kshape,2,ACL_BF16,kstride,0,ACL_FORMAT_ND,kstorage,2,e.restored);
        auto*o=aclCreateTensor(oshape,2,ACL_BF16,ostride,0,ACL_FORMAT_ND,oshape,2,e.answer);
        tensors.insert(tensors.end(),{q,k,o});
        for(int r=0;r<repeats;++r){
            int j=i*repeats+r;
            rc=aclnnMmGetWorkspaceSize(q,k,o,0,&sizes[j],&ops[j]);
            if(check(rc,"MmGetWorkspaceSize"))return rc;
            max_workspace=std::max(max_workspace,sizes[j]);
        }
    }
    if(max_workspace){int rc=aclrtMalloc(&workspace,max_workspace,ACL_MEM_MALLOC_NORMAL_ONLY);if(check(rc,"workspace"))return rc;}
    if(trace){
        aclrtCreateEventWithFlag(&origin,ACL_EVENT_TIME_LINE);
        for(auto& event:marks)aclrtCreateEventWithFlag(&event,ACL_EVENT_TIME_LINE);
        aclrtRecordEvent(origin,c);
        aclrtStreamWaitEvent(a,origin);
        for(int i=0;i<lanes;++i)aclrtStreamWaitEvent(decode_streams[i],origin);
    }
    auto start=Clock::now();
    auto auth=[&](int i,aclrtStream stream){
        auto&e=rows[i];
        if(trace)aclrtRecordEvent(marks[i*6],stream);
        int rc=aclrtMemcpyAsync(e.cipher,e.bytes,e.host,e.bytes,ACL_MEMCPY_HOST_TO_DEVICE,stream);
        if(rc)return check(rc,"H2D");
        rc=check(secure_infer_aicpu_gcm_verify_only_aad_do(stream,e.cipher,e.key,e.hash,e.tag,e.status,e.bytes,
            e.iv[0],e.iv[1],e.iv[2],e.iv[3],e.aad,64),"authenticate");
        if(trace)aclrtRecordEvent(marks[i*6+1],stream);
        return rc;
    };
    auto restore=[&](int i,aclrtStream stream){
        auto&e=rows[i];
        if(trace)aclrtRecordEvent(marks[i*6+2],stream);
        int rc=secure_infer_aicpu_aes_ctr_do(stream,e.cipher,e.key,e.plain,e.bytes,e.iv[0],e.iv[1],e.iv[2],e.iv[3]);
        if(rc)return check(rc,"decrypt");
        if(e.bits)rc=entropy_restore_do(stream,e.plain,e.restored,e.scratch,e.status,e.tiles,e.tile,e.bits);
        if(trace)aclrtRecordEvent(marks[i*6+3],stream);
        return check(rc,"decode");
    };
    auto compute=[&](int i,aclrtStream stream){
        if(trace)aclrtRecordEvent(marks[i*6+4],stream);
        for(int r=0;r<repeats;++r){
            int j=i*repeats+r;
            int rc=aclnnMm(workspace,sizes[j],ops[j],stream);
            if(rc)return check(rc,"Mm");
        }
        if(trace)aclrtRecordEvent(marks[i*6+5],stream);
        return 0;
    };
    if(pipeline){
        for(int step=0;step<count+2;++step){
            int rc=0;
            if(step<count){rc=auth(step,a);if(rc)return rc;rc=aclrtRecordEvent(ae[step],a);if(rc)return rc;}
            if(step>=1&&step-1<count){int i=step-1; auto d=decode_streams[i%lanes];rc=aclrtStreamWaitEvent(d,ae[i]);if(rc)return rc;rc=restore(i,d);if(rc)return rc;rc=aclrtRecordEvent(de[i],d);if(rc)return rc;}
            if(step>=2){int i=step-2;rc=aclrtStreamWaitEvent(c,de[i]);if(rc)return rc;rc=compute(i,c);if(rc)return rc;}
        }
    }else{
        for(int i=0;i<count;++i){int rc=auth(i,c);if(rc)return rc;rc=restore(i,c);if(rc)return rc;rc=compute(i,c);if(rc)return rc;}
    }
    int rc=aclrtSynchronizeStream(c);if(check(rc,"synchronize"))return rc;
    auto done=Clock::now();
    if(trace){
        for(int i=0;i<count*6;++i){
            int status=aclrtEventElapsedTime(&trace[i],origin,marks[i]);
            if(check(status,"trace elapsed time"))return status;
            aclrtDestroyEvent(marks[i]);
        }
        aclrtDestroyEvent(origin);
    }
    for(auto tensor:tensors)aclDestroyTensor(tensor);
    for(auto event:ae)aclrtDestroyEvent(event);
    for(auto event:de)aclrtDestroyEvent(event);
    if(workspace)aclrtFree(workspace);
    times[0]=std::chrono::duration<double,std::milli>(done-start).count();
    times[1]=std::chrono::duration<double,std::milli>(Clock::now()-outer).count();
    return 0;
}
