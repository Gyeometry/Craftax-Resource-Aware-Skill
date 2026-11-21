"""Only the original public symbolic observation enters policy masks."""
import jax,jax.numpy as jnp

def extract(obs):
 v=obs[...,:1323].reshape(obs.shape[:-1]+(7,9,21));b=v[...,:17].argmax(-1);known=v[...,:17].sum(-1)>.5
 inv=jnp.rint(obs[...,1323:1335]*10);valid_inv=jnp.isfinite(inv).all(-1)
 near=b[...,2:5,3:6];nk=known[...,2:5,3:6];select=jnp.array([[1,1,1],[1,0,1],[1,1,1]],bool)
 table=((near==11)&select).any((-1,-2));furnace=((near==12)&select).any((-1,-2));nearknown=(nk|~select).all((-1,-2))
 dr=jnp.array([3,3,2,4])[obs[...,1339:1343].argmax(-1)];dc=jnp.array([3,5,4,4])[obs[...,1339:1343].argmax(-1)];ix=dr*9+dc
 front=jnp.take_along_axis(v.reshape(obs.shape[:-1]+(63,21)),ix[...,None,None],axis=-2)[...,0,:]
 fb=front[...,:17].argmax(-1);fk=front[...,:17].sum(-1)>.5;mob=(front[...,17:20]>.5).any(-1)
 return inv,table,furnace,nearknown,fb,fk,mob,obs[...,1344]>.5,b,valid_inv

def all_masks(obs):
 iv,ct,fn,nk,fb,fk,mob,sleep,b,ik=extract(obs);wood,stone,coal,iron=iv[...,0],iv[...,1],iv[...,2],iv[...,3]
 wood_ok=wood>=1;st_ok=wood_ok&(stone>=1);iron_ok=st_ok&(iron>=1)&(coal>=1)
 # Unknown local cells cannot establish an absent facility; allow conservatively.
 ct_possible=ct|~nk;fn_possible=fn|~nk
 craft=[ct_possible&wood_ok,ct_possible&st_ok,ct_possible&fn_possible&iron_ok]*2
 m1=jnp.ones(obs.shape[:-1]+(17,),bool)
 for a,q in zip([11,12,13,14,15,16],craft):m1=m1.at[...,a].set(q|~ik)
 solid=jnp.isin(fb,jnp.array([3,4,5,8,9,10,11,12,15,16]));validfront=(fb!=1)&~mob
 place=[(stone>=1)&((fb==3)|~solid), (wood>=2)&~solid,(stone>=1)&~solid,(iv[...,5]>=1)&(fb==2)]
 for a,q in zip([7,8,9,10],place):m1=m1.at[...,a].set((q&validfront)|~fk|~ik)
 # Appendix C.2 Listing 2, independent public-observation implementation.
 m2=jnp.ones_like(m1)
 for a,q in zip([7,8,9,10],[stone>=1,wood>=2,stone>=1,iv[...,5]>=1]):m2=m2.at[...,a].set(q|~ik)
 for a,q,i in zip([11,12,13,14,15,16],craft,[6,7,8,9,10,11]):m2=m2.at[...,a].set((q&(iv[...,i]==0))|~ik)
 neighbors=[b[...,3,3],b[...,3,5],b[...,2,4],b[...,4,4]]
 for a,q in zip([1,2,3,4],neighbors):m2=m2.at[...,a].set(q!=14)
 can_do=mob|(fb==5)|(fb==3)|((fb==4)&(iv[...,6]>0))|((fb==8)&(iv[...,6]>0))|((fb==9)&(iv[...,7]>0))|((fb==10)&(iv[...,8]>0))|(fb==16)|(fb==2)
 m2=m2.at[...,5].set(can_do|~fk|~ik)
 only=jnp.zeros_like(m1).at[...,0].set(True)
 return jnp.where(sleep[...,None],only,m1),jnp.where(sleep[...,None],only,m2)

def action_mask(obs,mode):
 a,b=all_masks(obs);return jnp.where(mode==0,jnp.ones_like(a),jnp.where(mode==1,a,b))
