"""Cross-process Cake progress for a single request.

The compute pointer is shared by the scheduler.  Each tensor-parallel worker
has its own I/O pointer because every rank owns a different KV shard and must
reload the same token range.  Completion is also tracked per rank: workers
may finish their chunks out of order, but authentication remains local to the
worker that performed the transfer.
"""
import fcntl
import json
import os
from pathlib import Path

class Progress:
    def __init__(self,request_id,rank=None):
        self.path=Path(os.environ['RUNTIME_RESULT_DIR'])/'cake_progress'/request_id
        self.path.parent.mkdir(exist_ok=True)
        self.rank=None if rank is None else str(int(rank))

    def transact(self,fn):
        with self.path.open('r+') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            state=json.load(f)
            if state.get('error'):raise RuntimeError(state['error'])
            answer=fn(state)
            f.seek(0);json.dump(state,f);f.truncate();f.flush()
            return answer

    def create(self,base,assigned,end,world_size=1):
        assert base<=assigned<=end and base%128==assigned%128==end%128==0
        world_size=int(world_size)
        assert world_size>0
        state=dict(
            base=base, compute=assigned, io=end, ready=end,
            range_end=end,
            io_by_rank={str(rank):end for rank in range(world_size)},
            ready_by_rank={str(rank):end for rank in range(world_size)},
            ready_ranges={str(rank):[] for rank in range(world_size)},
            error=None,
        )
        try:
            with self.path.open('x') as f:
                json.dump(state,f)
        except FileExistsError:
            # TP workers can receive the same kickoff metadata.  One worker
            # creates the state; the others validate and reuse it.
            with self.path.open() as f:
                current=json.load(f)
            assert (current['base'],current['compute'],current['range_end']) == \
                (base,assigned,end)
            assert set(current.get('io_by_rank',{})) == set(state['io_by_rank'])

    def read(self):return self.transact(lambda s:dict(s))

    def compute(self,size):
        def claim(s):
            io_by_rank=s.get('io_by_rank')
            io=min(io_by_rank.values()) if io_by_rank else s['io']
            s['io']=io
            if s['compute']==io:return None
            assert s['compute']<io
            s['compute']=min(s['compute']+size,io)
            return s['compute']
        return self.transact(claim)

    def load(self,size,rank=None):
        rank=self.rank if rank is None else str(int(rank))
        if rank is None:rank='0'
        def claim(s):
            io_by_rank=s.setdefault('io_by_rank',{'0':s['io']})
            assert rank in io_by_rank
            io=io_by_rank[rank]
            if io==s['compute']:return None
            assert io>s['compute']
            end=io;io_by_rank[rank]=max(s['compute'],end-size)
            s['io']=min(io_by_rank.values())
            return io_by_rank[rank],end
        return self.transact(claim)

    def ready(self,start,end,rank=None):
        rank=self.rank if rank is None else str(int(rank))
        if rank is None:rank='0'
        def done(s):
            ready_by_rank=s.setdefault('ready_by_rank',{'0':s['range_end']})
            ready_ranges=s.setdefault('ready_ranges',{'0':[]})
            assert rank in ready_by_rank
            assert s['base']<=start<end<=s['range_end']
            ranges=ready_ranges.setdefault(rank,[])
            interval=[start,end]
            if interval not in ranges:ranges.append(interval)
            # Chunks can complete out of order across TP workers.  Retain
            # only the contiguous frontier for the rank; no rank may mark a
            # different rank's transfer as ready.
            frontier=ready_by_rank[rank]
            changed=True
            while changed:
                changed=False
                for candidate in list(ranges):
                    if candidate[1]==frontier:
                        frontier=candidate[0]
                        ranges.remove(candidate)
                        changed=True
            ready_by_rank[rank]=frontier
            # The frontier moves downward from range_end.  The largest
            # per-rank frontier is therefore the least-ready rank and is the
            # earliest point that is safe for all ranks to consume.
            s['ready']=max(ready_by_rank.values())
        self.transact(done)

    def error(self,message):
        self.transact(lambda s:s.update(error=message))
