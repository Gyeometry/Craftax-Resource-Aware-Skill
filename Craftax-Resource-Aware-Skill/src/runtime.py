"""배치 128을 보존하는 동결 정책 추론 어댑터. 학습 기능 없음."""
from pathlib import Path
import sys,json,types
import numpy as np
import jax,jax.numpy as jnp
from flax import serialization
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'vendor/Craftax'))
from craftax.craftax_env import make_craftax_env_from_name
import craftax.craftax_classic.game_logic as gl
from network import ActorCriticRNN
from masks import action_mask
from ablations import AccessController
B=128
g=types.SimpleNamespace(action_mask=action_mask)
NET=ActorCriticRNN(17,config={'LAYER_SIZE':512})
ENV=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',False)
EP=ENV.default_params
ns=gl.__dict__.copy()
exec(compile((P/'src/instrumented_logic.py').read_text('utf-8'), 'instrumented_logic.py','exec'),ns)
AUDIT=ns['audit_craftax_step']
def load(seed):
    return jax.tree.map(jnp.asarray,serialization.msgpack_restore((P/f'models/PPO_{seed}.msgpack').read_bytes()))
def load_skill(seed):
    return jax.tree.map(jnp.asarray,serialization.msgpack_restore((P/f'models/Skill_{seed}.msgpack').read_bytes()))


@jax.jit
def keys(root,world,t):
    return jax.vmap(lambda w,s:jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(root),w),s))(world,t)

@jax.jit
def reset(world):return jax.vmap(ENV.reset,in_axes=(0,None))(jax.vmap(jax.random.PRNGKey)(world),EP)

@jax.jit
def policy(par,h,ob,world,root,t):
    mask=g.action_mask(ob,2);hn,pi,v=NET.apply(par,h,(ob[None],jnp.zeros((1,B),bool),mask[None]))
    a=jax.vmap(jax.random.categorical)(keys(root,world,t),pi.logits[0])
    return hn,pi.probs[0],mask,a,v[0]

def choose(mask,a,b):
    return jax.tree.map(lambda x,y:jnp.where(mask.reshape((B,)+(1,)*(x.ndim-1)),x,y),a,b)

def initial(ob,st,valid=None,t=None,h=None):
    return (ob,st,jnp.zeros((B,512)) if h is None else h,
            jnp.ones(B,bool) if valid is None else valid,jnp.zeros(B,jnp.int32) if t is None else t,
            st,jnp.zeros((B,512)))

def one(key,state,a):
    new,r,ledger=AUDIT(key,state,a,EP,ENV.static_env_params)
    return ENV.get_obs(new),new,r,ENV.is_terminal(new,EP),ledger

@jax.jit
def transition(c,hn,a,world,eroot,forced):
    ob,st,h,alive,t,terminal,terminal_h=c
    no,new,r,done,ledger=jax.vmap(one)(keys(eroot,world,t),st,a)
    fin=alive&done;terminal=choose(fin,new,terminal);terminal_h=jnp.where(fin[:,None],hn,terminal_h)
    view=ob[:,:1323].reshape(B,7,9,21);grid=view[...,:17].argmax(-1).astype(jnp.uint8)
    mob=jnp.sum((view[...,17:21]>.5).astype(jnp.uint8)*jnp.array([1,2,4,8],jnp.uint8),-1)
    near=grid[:,2:5,3:6];sel=jnp.array([[1,1,1],[1,0,1],[1,1,1]],bool)
    ct=((near==11)&sel).any((-1,-2));fn=((near==12)&sel).any((-1,-2));iv=jnp.rint(ob[:,1323:1335]*10).astype(jnp.int32)
    post=jnp.stack([getattr(new.inventory,k) for k in new.inventory.__dataclass_fields__],-1)
    tr=dict(active=alive,action=a,reward=jnp.where(alive,r,0),done=done,forced=alive&forced,
            new_achievements=(new.achievements&~st.achievements)&alive[:,None],
            coaccess=alive&(iv[:,:4]>=1).all(-1)&ct&fn&(ob[:,-1]<.5),
            grid=grid,mob=mob,obs_tail=ob[:,1323:],pre_inventory=iv,post_inventory=post,
            health=st.player_health,post_health=new.player_health,ledger=ledger,absolute_t=t,
            position=st.player_position,post_position=new.player_position,direction=st.player_direction,
            vitals=jnp.stack([st.player_health,st.player_food,st.player_drink,st.player_energy],-1),
            post_vitals=jnp.stack([new.player_health,new.player_food,new.player_drink,new.player_energy],-1))
    return (no,new,hn,alive&~done,t+1,terminal,terminal_h),tr

@jax.jit
def capture_chunk(par,c,world,proot,eroot):
    def step(c,_):
        hn,p,mask,a,v=policy(par,c[2],c[0],world,proot,c[4])
        nc,tr=transition(c,hn,a,world,eroot,jnp.zeros(B,bool))
        extra=dict(state=c[1],obs=c[0],h=c[2],prob=p,mask=mask,value=v)
        tr['sampled_action']=a
        return nc,(tr,extra)
    return jax.lax.scan(step,c,None,16)

@jax.jit
def chunk(par,c,world,proot,eroot):
    def step(c,_):
        hn,p,mask,a,v=policy(par,c[2],c[0],world,proot,c[4]);nc,tr=transition(c,hn,a,world,eroot,jnp.zeros(B,bool))
        tr['sampled_action']=a
        return nc,tr
    return jax.lax.scan(step,c,None,16)

def decode(grid,mob,tail):
    sp=np.concatenate([np.eye(17,dtype=np.float32)[grid],np.stack([(mob&bit)>0 for bit in [1,2,4,8]],-1)],-1)
    return np.concatenate([sp.reshape(*grid.shape[:-2],1323),tail],-1).astype(np.float32)

def host(tree):return jax.tree.map(lambda x:np.asarray(x).copy(),tree)
