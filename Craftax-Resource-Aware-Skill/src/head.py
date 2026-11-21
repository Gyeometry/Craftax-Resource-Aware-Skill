"""공개 관측만 사용하는 시작 분류기."""
import numpy as np,jax,jax.numpy as jnp
from skill import facts

def features(ob,mask):return jnp.concatenate([ob,mask.astype(jnp.float32)],-1)
def head_logit(hp,x):
    h=jax.nn.relu(x@hp['w1']+hp['b1']);h=jax.nn.relu(h@hp['w2']+hp['b2']);return (h@hp['w3']+hp['b3'])[:,0]
def head_probability(hp,ob,mask):return jax.nn.sigmoid(head_logit(hp,features(ob,mask)))
def simple_rule(ob):
    goal,cand,bad,ct,fn,mat=facts(ob);iv=jnp.rint(ob[:,1323:1335]*10)
    return cand&((ct&(iv[:,1]>=2))|(fn&(iv[:,0]>=3))|((iv[:,0]>=3)&(iv[:,1]>=2)))
def initialize(seed=154001):
    rng=np.random.default_rng(seed);n=[1362,128,128,1];p={}
    for i in range(1,4):
        p[f'w{i}']=jnp.asarray(rng.normal(0,np.sqrt(2/n[i-1]),(n[i-1],n[i])).astype(np.float32));p[f'b{i}']=jnp.zeros(n[i])
    p['threshold']=jnp.float32(.5);return p
