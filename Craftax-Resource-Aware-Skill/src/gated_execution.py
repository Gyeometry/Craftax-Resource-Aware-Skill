"""시작 판정만 변경한 동결 B2 실행기. build_execution.py로 생성했다."""
from functools import partial
import numpy as np,jax,jax.numpy as jnp
import runtime as e
from skill import NET,facts
from head import head_probability,simple_rule
from planner import public_plan
from auto_gate import auto_gate
B=e.B

def plan_labels(ob,mask,pending):
    ob=np.asarray(ob);mask=np.asarray(mask);pending=np.asarray(pending);out=np.zeros(B,bool)
    for j in np.flatnonzero(pending):
        p=public_plan(ob[j].copy(),16);a=p['actions'][0] if p['actions'] else -1
        out[j]=bool(0<=a<17 and mask[j,a])
    return out

def initial(c):
    import execution as original
    return (*original.initial(c),jnp.full(B,-1,jnp.int32))

@partial(jax.jit,static_argnames=('mode',))
def gated_chunk(par,sp,hp,carry,world,proot,eroot,sroot,mode):
    def step(cc,_):
        c,sh,on,started,count,reason,first_candidate=cc;ob=c[0];alive=c[3]
        hn,p,mask,pa,value=e.policy(par,c[2],ob,world,proot,c[4]);goal,candidate,bad,*_=facts(ob)
        pending=alive&~started&candidate
        first_candidate=jnp.where((first_candidate<0)&pending,c[4],first_candidate)
        query=jnp.zeros(B,bool);feasible=jnp.zeros(B,bool);score=jnp.zeros(B)
        if mode in ['G1','COLLECT0','COLLECTG0']:
            query=pending
            feasible=jax.lax.cond(pending.any(),lambda _:jax.pure_callback(plan_labels,jax.ShapeDtypeStruct((B,),jnp.bool_),ob,mask,pending),lambda _:jnp.zeros(B,bool),None)
            score=feasible.astype(jnp.float32)
        if mode=='AUTO':score=auto_gate(ob).astype(jnp.float32)
        if mode in ['G2','RULE']:
            score=head_probability(hp,ob,mask) if mode=='G2' else simple_rule(ob).astype(jnp.float32)
        allow=jnp.ones(B,bool)
        if mode=='G1':allow=feasible
        if mode=='G2':allow=score>=hp['threshold']
        if mode in ['RULE','AUTO']:allow=score>.5
        if mode=='G3':allow=(first_candidate>=0)&(c[4]-first_candidate>=16)
        if mode=='COLLECT0':allow=jnp.zeros(B,bool)
        first=pending&allow;on=(on|first)&alive;started=started|first
        sh=jnp.where(first[:,None],0,sh);count=jnp.where(first,0,count)
        ns,logits,bl=jax.lax.cond(on.any(),lambda _:NET.apply(sp,sh,ob,count),lambda _:(sh,jnp.where(mask,jnp.zeros_like(mask,dtype=jnp.float32),-1e9),jnp.zeros(B)),operand=None)
        prob=jax.nn.softmax(logits);beta=jax.nn.sigmoid(bl)
        proposal=jax.vmap(jax.random.categorical)(e.keys(sroot,world,c[4]),logits)
        bt=jax.vmap(jax.random.bernoulli)(e.keys(sroot+1000,world,c[4]),beta)
        stop=on&(bad|(count>=16))
        rr=jnp.where(on&bad,3,jnp.where(on&(count>=16),2,reason))
        
        if True:stop=stop|(on&goal);rr=jnp.where(on&goal,1,rr)
        
        craft=on&~stop&((proposal==13)|(proposal==16));stop=stop|craft;rr=jnp.where(craft,7,rr)
        execute=on&~stop;action=jnp.where(execute,proposal,pa)
        nc,tr=e.transition(c,hn,action,world,eroot,execute)
        postgoal,*_=facts(nc[0]);count=count+execute.astype(jnp.int32)
        after=execute&(tr['done']|(count>=16));rr=jnp.where(execute&tr['done'],4,jnp.where(execute&(count>=16),2,rr))
        
        if True:after=after|(execute&postgoal);rr=jnp.where(execute&postgoal&~tr['done'],1,rr)
        end=(on&stop)|after
        tr.update(init_candidate=pending,first_candidate_t=first_candidate,planner_query=query,planner_feasible=feasible,gate_score=score,public_obs=ob,sampled_action=pa,skill_trigger=first,skill_executed=execute,skill_proposal=proposal,
            skill_probability=jnp.take_along_axis(prob,proposal[:,None],-1)[:,0],skill_beta=beta,
            skill_pre_h=jnp.where(on[:,None],sh,0),handoff=end,ppo_h_at_handoff=jnp.where(end[:,None],hn,0),
            skill_goal_after=execute&postgoal&~tr['done'],skill_count=count,skill_reason=rr)
        return (nc,jnp.where(on[:,None],ns,sh),on&~stop&~after&nc[3],started,count,rr,first_candidate),tr
    return jax.lax.scan(step,carry,None,16)

