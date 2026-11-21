"""Separate ablation implementations; the original C2 files are never modified."""
from collections import deque
import numpy as np
from planner import public_plan,public_view,adjacency,SOLID,DELTA,CRAFT
from controller import Controller
def eligible(obs):
 obs=np.asarray(obs);single=obs.ndim==1
 if single:obs=obs[None]
 iv=np.rint(obs[:,1323:1335]*10).astype(int);grid=obs[:,:1323].reshape(-1,7,9,21)[...,:17].argmax(-1);near=grid[:,2:5,3:6];exclude=np.array([[1,1,1],[1,0,1],[1,1,1]],bool);ct=((near==11)&exclude).any((-1,-2));fn=((near==12)&exclude).any((-1,-2));result=(iv[:,:4]>=1).all(-1)&ct&fn&(obs[:,1344]<.5)
 return bool(result[0]) if single else result
def candidate(obs):
 obs=np.asarray(obs);iv=np.rint(obs[1323:1335]*10).astype(int);return bool(min(iv[:4])>=1 and iv[8]==0 and iv[11]==0 and obs[1344]<.5 and not eligible(obs))
def existing_plan(obs,budget=16,maxdepth=32):
 grid,known,mob,start,d,iv,sleep=public_view(obs);h,w=grid.shape
 if sleep:return dict(actions=[],reason='SLEEPING',length=None,kind=None)
 if min(iv[:4])<1:return dict(actions=[],reason='MATERIALS_MISSING',length=None,kind=None)
 if iv[8]>0 and iv[11]>0:return dict(actions=[],reason='TOOLS_ALREADY_OWNED',length=0,kind=None)
 if not (grid==11).any() or not (grid==12).any():return dict(actions=[],reason='NO_EXISTING_STATION_PLAN',length=None,kind=None)
 nt=adjacency(grid,11);nf=adjacency(grid,12);root=(*start,d);queue=deque([root]);dist={root:0};parent={};best=None
 def path(q):
  out=[]
  while q!=root:q,a=parent[q];out.append(a)
  return out[::-1]
 while queue:
  q=queue.popleft();r,c,d=q;dep=dist[q]
  if dep>maxdepth or (best is not None and dep+1>len(best)):continue
  if nt[r,c] and nf[r,c]:
   p=path(q)+[CRAFT]
   if best is None or (len(p),tuple(p))<(len(best),tuple(best)):best=p
  if dep==maxdepth:continue
  for a in range(1,5):
   dr,dc=DELTA[a];rr,cc=r+dr,c+dc
   if not (0<=rr<h and 0<=cc<w and known[rr,cc] and int(grid[rr,cc]) not in (0,1,14)):continue
   walk=int(grid[rr,cc]) not in SOLID and not mob[rr,cc];rr,cc=(rr,cc) if walk else (r,c);nxt=(rr,cc,a)
   if nxt not in dist:dist[nxt]=dep+1;parent[nxt]=(q,a);queue.append(nxt)
 if best is None:return dict(actions=[],reason='NO_EXISTING_STATION_PLAN',length=None,kind=None)
 if len(best)>budget:return dict(actions=[],reason='PLAN_EXCEEDS_REMAINING_BUDGET',length=len(best),kind='existing_stations')
 return dict(actions=best,reason='PLAN_FOUND',length=len(best),kind='existing_stations')

# Isolated subclass: the original Controller.decide is compiled unchanged except
# planner dispatch and the terminal CRAFT treatment. No module monkey-patching.
import inspect,ast,textwrap
tree=ast.parse(textwrap.dedent(inspect.getsource(Controller.decide)));fn=tree.body[0]
for node in ast.walk(fn):
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='public_plan':node.func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='get_plan',ctx=ast.Load())
for idx,node in enumerate(fn.body):
 if isinstance(node,ast.For):
  for k,child in enumerate(node.body):
   if isinstance(child,ast.If) and isinstance(child.test,ast.Compare) and isinstance(child.test.comparators[0],ast.Name) and child.test.comparators[0].id=='CRAFT':
    child.body=ast.parse("self.success[j]=True\nself.on[j]=False\nself.end_t[j]=t\nself.status[j]='ACCESS_COMPLETE'\nself.logs.append(dict(row=int(j),t=int(t),phase='access_return',plan_reason='ACCESS_COMPLETE',plan_length=pl['length'],plan_kind=pl['kind'],planned_actions=pl['actions'],action=int(actions[j]),overridden=False,triggered=False,remaining_budget=budget,mask_allowed=bool(mask[j,actions[j]]),seconds=elapsed))\ncontinue").body
namespace=dict(np=np,CRAFT=CRAFT);exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),namespace)
class AccessController(Controller):
 decide=namespace['decide']
 def __init__(self,n=256,existing=False):super().__init__(n);self.get_plan=existing_plan if existing else public_plan
 def finish(self,post_obs,done,pre_alive,t):
  goal=eligible(post_obs)
  for j in np.flatnonzero(self.on&pre_alive):
   if done[j]:self.died_during[j]=True;self.status[j]='DIED';self.on[j]=False;self.end_t[j]=t+1
   elif goal[j]:self.success[j]=True;self.status[j]='ACCESS_COMPLETE';self.on[j]=False;self.end_t[j]=t+1
   elif self.steps[j]>=16:self.status[j]='LIMIT16';self.on[j]=False;self.end_t[j]=t+1
def smoke():
 ob=np.zeros((1,1345),np.float32);v=ob[0,:1323].reshape(7,9,21);v[:,:,2]=1;v[2,4,2]=0;v[2,4,11]=1;ob[0,1323:1327]=[.3,.2,.1,.1];ob[0,1340]=1;mask=np.ones((1,17),bool);p=np.ones((1,17))/17;c=AccessController(1);a,o,*_=c.decide(ob,mask,p,np.array([5]),np.ones(1,bool),0);assert a[0]==9 and o[0]
 after=ob.copy();va=after[0,:1323].reshape(7,9,21);va[3,5,2]=0;va[3,5,12]=1;c.finish(after,np.zeros(1,bool),np.ones(1,bool),0);assert c.success[0] and not c.on[0];a,o,*_=c.decide(after,mask,p,np.array([5]),np.ones(1,bool),1);assert not o[0] and a[0]==5
 a2=existing_plan(ob[0]);assert not a2['actions'];ob2=ob.copy();vb=ob2[0,:1323].reshape(7,9,21);vb[2,6,2]=0;vb[2,6,12]=1;pl=existing_plan(ob2[0]);assert pl['actions'] and not any(x in [8,9,13,16] for x in pl['actions']);d=AccessController(1,True);a,o,*_=d.decide(ob2,mask,p,np.array([5]),np.ones(1,bool),0);assert o[0] and a[0] in [1,2,3,4]
 return dict(passed=True,environment_steps=0,A1_returns_before_craft=True,A2_no_install_no_craft=True,original_C2_unmodified=True)
if __name__=='__main__':print(smoke())
