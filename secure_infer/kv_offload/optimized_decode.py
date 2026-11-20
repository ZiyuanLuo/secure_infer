"""NPU recovery with parameter views and reusable conversion buffers."""
import torch


def decode_into(wire,destination,indices,bits,kind,qbytes,sbytes,group=64,temporary=None):
    b=wire.shape[0]
    _,tokens,heads,dim=destination.shape
    total=b*tokens*heads*dim
    if temporary is None:temporary=torch.empty(total,dtype=torch.float32,device=wire.device)
    full=temporary.reshape(b,-1)
    if bits==8:
        full.copy_(wire[:,:qbytes])
    else:
        packed=wire[:,:qbytes].float()
        high=(packed/16).floor()
        torch.stack((packed-16*high,high),dim=-1,out=full.view(b,-1,2))
    shape=(b,tokens//group,group,heads,dim)if kind==0 else(b,tokens,heads,dim//group,group)
    parameter_shape=(b,tokens//group,1,heads,dim)if kind==0 else(b,tokens,heads,dim//group,1)
    parameters=wire.view(torch.float32)
    scale=parameters[:,qbytes//4:(qbytes+sbytes)//4].view(parameter_shape)
    offset=parameters[:,(qbytes+sbytes)//4:(qbytes+2*sbytes)//4].view(parameter_shape)
    values=full.view(shape)
    values.mul_(scale)
    values.add_(offset)
    restored=values.reshape(b,tokens,heads,dim).to(destination.dtype)
    destination.index_copy_(0,indices,restored)
    return temporary
