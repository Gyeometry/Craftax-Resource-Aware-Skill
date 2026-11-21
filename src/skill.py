"""공개 관측 CNN-GRU와 시연 기반 Option 종료. 기본 PPO와 파라미터 공유 없음."""
import jax,jax.numpy as jnp,flax.linen as nn
from masks import action_mask,extract

def facts(ob):
    iv,ct,fn,nk,fb,fk,mob,sleep,b,ik=extract(ob)
    materials=(iv[...,:4]>=1).all(-1);goal=materials&ct&fn&~sleep
    candidate=materials&(iv[...,8]==0)&(iv[...,11]==0)&~sleep&~goal
    bad=~materials|sleep
    return goal,candidate,bad,ct,fn,materials

class Skill(nn.Module):
    @nn.compact
    def __call__(self,h,ob,elapsed):
        goal,cand,bad,ct,fn,mat=facts(ob);mask=action_mask(ob,2)
        spatial=ob[...,:1323].reshape((-1,7,9,21))
        x=nn.relu(nn.Conv(32,(3,3),padding='SAME',name='conv1')(spatial))
        x=nn.relu(nn.Conv(32,(3,3),padding='SAME',name='conv2')(x)).reshape((ob.shape[0],-1))
        g=jnp.stack([ct,fn,mat,goal,bad],-1).astype(jnp.float32)
        x=jnp.concatenate([x,ob[:,1323:],mask.astype(jnp.float32),g,elapsed[:,None]/16],-1)
        x=nn.relu(nn.Dense(128,name='input')(x));h,x=nn.GRUCell(128,name='gru')(h,x)
        x=nn.relu(nn.Dense(128,name='readout')(x))
        logits=nn.Dense(17,name='actor')(x);term=nn.Dense(1,name='termination')(x)[:,0]
        logits=jnp.where(mask,logits,-1e9)
        return h,logits,term

NET=Skill()
def sequence(params,ob):
    def one(h,x):
        o,t=x;hn,l,b=NET.apply(params,h,o,jnp.full(o.shape[0],t));return hn,(l,b)
    _,(l,b)=jax.lax.scan(one,jnp.zeros((ob.shape[0],128)),(ob.swapaxes(0,1),jnp.arange(17)))
    return l.swapaxes(0,1),b.swapaxes(0,1)
