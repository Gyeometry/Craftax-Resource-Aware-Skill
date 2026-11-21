"""공개 조건의 Skill 실행기. Skill 경로는 Planner를 호출하지 않는다."""
import json,contextlib
from functools import partial
import numpy as np,jax,jax.numpy as jnp
from flax import serialization
import runtime as e
from skill import NET,facts
B=e.B

def load_skill(seed):return e.load_skill(seed)

def initial(base,started=None,on=None):
    return (base,jnp.zeros((B,128)),jnp.zeros(B,bool) if on is None else on,jnp.zeros(B,bool) if started is None else started,jnp.zeros(B,jnp.int32),jnp.zeros(B,jnp.int32))

@partial(jax.jit,static_argnames=('mode',))
def mixed_chunk(par,sp,carry,world,proot,eroot,sroot,mode):
    def step(cc,_):
        c,sh,on,started,count,reason=cc;ob=c[0];alive=c[3]
        hn,p,mask,pa,value=e.policy(par,c[2],ob,world,proot,c[4]);goal,candidate,bad,*_=facts(ob)
        first=alive&~started&candidate;on=(on|first)&alive;started=started|first
        sh=jnp.where(first[:,None],0,sh);count=jnp.where(first,0,count)
        ns,logits,bl=jax.lax.cond(on.any(),lambda _:NET.apply(sp,sh,ob,count),lambda _:(sh,jnp.where(mask,jnp.zeros_like(mask,dtype=jnp.float32),-1e9),jnp.zeros(B)),operand=None)
        prob=jax.nn.softmax(logits);beta=jax.nn.sigmoid(bl)
        proposal=jax.vmap(jax.random.categorical)(e.keys(sroot,world,c[4]),logits)
        bt=jax.vmap(jax.random.bernoulli)(e.keys(sroot+1000,world,c[4]),beta)
        stop=on&(bad|(count>=16))
        rr=jnp.where(on&bad,3,jnp.where(on&(count>=16),2,reason))
        if mode=='B1':stop=stop|(on&(count>=1));rr=jnp.where(on&(count>=1),5,rr)
        if mode=='B2':stop=stop|(on&goal);rr=jnp.where(on&goal,1,rr)
        if mode=='B3':stop=stop|(on&(count>0)&bt);rr=jnp.where(on&(count>0)&bt,6,rr)
        craft=on&~stop&((proposal==13)|(proposal==16));stop=stop|craft;rr=jnp.where(craft,7,rr)
        execute=on&~stop;action=jnp.where(execute,proposal,pa)
        nc,tr=e.transition(c,hn,action,world,eroot,execute)
        postgoal,*_=facts(nc[0]);count=count+execute.astype(jnp.int32)
        after=execute&(tr['done']|(count>=16));rr=jnp.where(execute&tr['done'],4,jnp.where(execute&(count>=16),2,rr))
        if mode=='B1':after=after|execute;rr=jnp.where(execute&~tr['done'],5,rr)
        if mode=='B2':after=after|(execute&postgoal);rr=jnp.where(execute&postgoal&~tr['done'],1,rr)
        end=(on&stop)|after
        tr.update(public_obs=ob,sampled_action=pa,skill_trigger=first,skill_executed=execute,skill_proposal=proposal,
            skill_probability=jnp.take_along_axis(prob,proposal[:,None],-1)[:,0],skill_beta=beta,
            skill_pre_h=jnp.where(on[:,None],sh,0),handoff=end,ppo_h_at_handoff=jnp.where(end[:,None],hn,0),
            skill_goal_after=execute&postgoal&~tr['done'],skill_count=count,skill_reason=rr)
        return (nc,jnp.where(on[:,None],ns,sh),on&~stop&~after&nc[3],started,count,rr),tr
    return jax.lax.scan(step,carry,None,16)

@contextlib.contextmanager
def no_planner():
    import planner,controller,ablations
    saved=[(m,m.public_plan) for m in [planner,controller,ablations]];called=[]
    def forbidden(*args,**kw):called.append(1);raise AssertionError('PLANNER_CALLED_IN_SKILL_EVALUATION')
    for m,_ in saved:m.public_plan=forbidden
    try:yield called
    finally:
        for m,f in saved:m.public_plan=f

def base_trace(par,c,world,pr,er):
    nc,(tr,extra)=e.capture_chunk(par,c,world,pr,er)
    return nc,tr

def teacher16(par,c,world,pr,er,controllers,ticks):
    rows=[]
    for t in range(16):
        hn,p,m,sa,v=jax.block_until_ready(e.policy(par,c[2],c[0],world,pr,c[4]));act=np.asarray(sa).copy();forced=np.zeros(B,bool);alive=np.asarray(c[3]);ob=np.asarray(c[0])
        for j in np.flatnonzero(alive):
            aa,ff,*_=controllers[j].decide(ob[j:j+1],np.asarray(m)[j:j+1],np.asarray(p)[j:j+1],act[j:j+1],np.ones(1,bool),int(ticks[j]+t));act[j]=aa[0];forced[j]=ff[0]
        nc,tr=jax.block_until_ready(e.transition(c,hn,jnp.asarray(act),world,er,jnp.asarray(forced)));tr=e.host(tr);tr['sampled_action']=np.asarray(sa);rows.append(tr)
        for j in np.flatnonzero(alive):controllers[j].finish(np.asarray(nc[0])[j:j+1],tr['done'][j:j+1],np.ones(1,bool),int(ticks[j]+t))
        c=nc
    return c,{k:np.stack([q[k] for q in rows]) for k in rows[0]}
