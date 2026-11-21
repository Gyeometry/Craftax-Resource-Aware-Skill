"""Restricted constructive plans. Public entry point accepts only original observation."""
from collections import deque
import numpy as np
SOLID={3,4,5,8,9,10,11,12,15,16};DELTA=[(0,0),(0,-1),(0,1),(-1,0),(1,0)];OPPOSITE={1:2,2:1,3:4,4:3};CRAFT=17
def adjacency(grid,kind):
 out=np.zeros(grid.shape,bool);h,w=grid.shape
 for r,c in np.argwhere(grid==kind):out[max(0,r-1):min(h,r+2),max(0,c-1):min(w,c+2)]=True
 # A station cannot be a standing cell; center exclusion is handled explicitly.
 out[grid==kind]=False
 return out
def public_view(obs):
 obs=np.asarray(obs);v=obs[:1323].reshape(7,9,21);known=v[:,:,:17].sum(-1)>.5;grid=v[:,:,:17].argmax(-1);mobs=(v[:,:,17:20]>.5).any(-1);iv=np.rint(obs[1323:1335]*10).astype(int);direction=int(obs[1339:1343].argmax())+1
 return grid,known,mobs,(3,4),direction,iv,bool(obs[1344]>.5)
def full_view(state):
 iv=np.array([state['inventory'][k] for k in ['wood','stone','coal','iron','diamond','sapling','wood_pickaxe','stone_pickaxe','iron_pickaxe','wood_sword','stone_sword','iron_sword']],int)
 grid=np.asarray(state['map']);return grid,np.ones(grid.shape,bool),np.asarray(state['mob_map'],bool),tuple(np.asarray(state['player_position']).tolist()),int(state['player_direction']),iv,bool(state['is_sleeping'])
def plan(view,budget=16,maxdepth=32):
 grid,known,mob,start,direction,iv,sleep=view;h,w=grid.shape
 if sleep:return dict(actions=[],reason='SLEEPING',length=None,kind=None)
 if min(iv[:4])<1:return dict(actions=[],reason='MATERIALS_MISSING',length=None,kind=None)
 if iv[8]>0 and iv[11]>0:return dict(actions=[],reason='TOOLS_ALREADY_OWNED',length=0,kind=None)
 nearT=adjacency(grid,11);nearF=adjacency(grid,12)
 def inside(r,c):return 0<=r<h and 0<=c<w and known[r,c] and int(grid[r,c]) not in (0,1)
 def walk(r,c):return inside(r,c) and int(grid[r,c]) not in SOLID and int(grid[r,c])!=14 and not mob[r,c]
 def place(r,c):return inside(r,c) and int(grid[r,c]) not in SOLID and not mob[r,c]
 tableReserve=iv[0]>=3;furnaceReserve=iv[1]>=2;bothReserve=tableReserve and furnaceReserve
 root=(*start,direction);queue=deque([root]);dist={root:0};parent={};best=None;bestkind=None
 def path(q):
  arr=[]
  while q!=root:q,a=parent[q];arr.append(a)
  return arr[::-1]
 def consider(q,tail,kind):
  nonlocal best,bestkind
  n=dist[q]+len(tail)
  if best is not None and n>len(best):return
  candidate=path(q)+tail
  if best is None or (n,tuple(candidate))<(len(best),tuple(best)):best=candidate;bestkind=kind
 while queue:
  q=queue.popleft();r,c,d=q;dep=dist[q]
  if dep>maxdepth or (best is not None and dep+1>len(best)):continue
  if nearT[r,c] and nearF[r,c]:consider(q,[CRAFT],'existing_stations')
  dr,dc=DELTA[d];tr,tc=r+dr,c+dc
  if place(tr,tc):
   if tableReserve and nearF[r,c]:consider(q,[8,CRAFT],'add_table')
   if furnaceReserve and nearT[r,c]:consider(q,[9,CRAFT],'add_furnace')
   if bothReserve:
    for ed in range(1,5):
     er,ec=DELTA[ed]
     if dr*er+dc*ec!=0:continue
     if place(r+er,c+ec) and walk(r-er,c-ec):
      consider(q,[8,OPPOSITE[ed],ed,9,CRAFT],'add_both')
      consider(q,[9,OPPOSITE[ed],ed,8,CRAFT],'add_both')
  if dep==maxdepth:continue
  for a in range(1,5):
   ar,ac=DELTA[a];rr,cc=r+ar,c+ac
   if not inside(rr,cc) or grid[rr,cc]==14:continue
   rr,cc=(rr,cc) if walk(rr,cc) else (r,c)
   nxt=(rr,cc,a)
   if nxt not in dist:dist[nxt]=dep+1;parent[nxt]=(q,a);queue.append(nxt)
 if best is None:return dict(actions=[],reason='NO_RESTRICTED_PLAN',length=None,kind=None)
 if len(best)>budget:return dict(actions=[],reason='PLAN_EXCEEDS_REMAINING_BUDGET',length=len(best),kind=bestkind)
 return dict(actions=best,reason='PLAN_FOUND',length=len(best),kind=bestkind)
def public_plan(obs,budget=16):return plan(public_view(obs),budget)
def oracle_plan(state,budget=16):return plan(full_view(state),budget)
def smoke():
 g=np.full((7,9),2,int);k=np.ones_like(g,bool);m=np.zeros_like(g,bool);iv=np.array([3,2,1,1,0,0,0,0,0,0,0,0]);g[2,4]=11
 a=plan((g,k,m,(3,4),2,iv,False));assert a['length']==2 and a['actions']==[9,17]
 g[3,5]=12;b=plan((g,k,m,(3,4),2,iv,False));assert b['actions']==[17]
 z=plan((g,k,m,(3,4),2,iv,False),0);assert z['reason']=='PLAN_EXCEEDS_REMAINING_BUDGET'
 g[:]=2;c=plan((g,k,m,(3,4),2,iv,False));assert c['length']==5
 ob=np.zeros(1345,np.float32);v=ob[:1323].reshape(7,9,21);v[:,:,2]=1;ob[1323:1327]=[.3,.2,.1,.1];ob[1340]=1
 assert public_plan(ob)==public_plan(ob.copy());assert public_plan(ob)['length']==5
 return dict(passed=True,fixtures=5,environment_steps=0,public_function_inputs=['obs','budget'],public_hidden_state_inputs=False)
if __name__=='__main__':print(smoke())
