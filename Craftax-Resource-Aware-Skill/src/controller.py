"""Online decision interface: public observations and own episode history only."""
import numpy as np
from planner import public_plan,CRAFT
class Controller:
 def __init__(self,n=256):
  self.n=n;self.started=np.zeros(n,bool);self.on=np.zeros(n,bool);self.steps=np.zeros(n,int);self.candidate=np.zeros(n,bool);self.queries=np.zeros(n,int);self.deferrals=np.zeros(n,int);self.feasible=np.zeros(n,bool);self.first_candidate=np.full(n,-1,int);self.start_t=np.full(n,-1,int);self.end_t=np.full(n,-1,int);self.status=np.full(n,'NOT_TRIGGERED',object);self.success=np.zeros(n,bool);self.died_during=np.zeros(n,bool);self.logs=[];self.plan_seconds=0.
 def decide(self,obs,mask,prob,sampled,alive,t):
  import time
  obs=np.asarray(obs);iv=np.rint(obs[:,1323:1335]*10).astype(int);v=obs[:,:1323].reshape(-1,7,9,21);grid=v[:,:,:,:17].argmax(-1);near=grid[:,2:5,3:6];exclude=np.array([[1,1,1],[1,0,1],[1,1,1]],bool);ct=((near==11)&exclude).any((-1,-2));fn=((near==12)&exclude).any((-1,-2));base=(iv[:,:4]>=1).all(-1)&(iv[:,8]==0)&(iv[:,11]==0)&(obs[:,1344]<.5)&~(ct&fn)&alive
  self.candidate|=base;new=base&(self.first_candidate<0);self.first_candidate[new]=t;actions=np.array(sampled).copy();override=np.zeros(self.n,bool);query=np.zeros(self.n,bool);fire=np.zeros(self.n,bool)
  for j in np.flatnonzero(alive&(self.on|(~self.started&base))):
   on=bool(self.on[j]);budget=16-int(self.steps[j]) if on else 16;begin=time.perf_counter();pl=public_plan(obs[j],budget);elapsed=time.perf_counter()-begin;self.plan_seconds+=elapsed;query[j]=True
   if not on:self.queries[j]+=1
   a=pl['actions'][0] if pl['actions'] else -1
   if a==CRAFT:
    allowed=[x for x in [13,16] if mask[j,x]];a=max(allowed,key=lambda x:(prob[j,x],-x)) if allowed else -1
   reason=pl['reason']
   if a>=0 and mask[j,a]:
    self.feasible[j]=True
    if not on:self.started[j]=True;self.on[j]=True;self.start_t[j]=t;fire[j]=True
    actions[j]=a;override[j]=True;self.steps[j]+=1;self.status[j]='RUNNING'
   else:
    reason=reason if not pl['actions'] else 'MASK_REJECTION_ABORT'
    if on:self.on[j]=False;self.status[j]=reason;self.end_t[j]=t
    else:self.deferrals[j]+=1;self.status[j]='DEFERRED_ONLY'
   self.logs.append(dict(row=int(j),t=int(t),phase='intervention' if on else 'pretrigger',plan_reason=reason,plan_length=pl['length'],plan_kind=pl['kind'],planned_actions=pl['actions'],action=int(actions[j]),overridden=bool(override[j]),triggered=bool(fire[j]),remaining_budget=budget,mask_allowed=bool(mask[j,actions[j]]),seconds=elapsed))
  return actions,override,base,query,fire
 def finish(self,post_obs,done,pre_alive,t):
  iv=np.rint(np.asarray(post_obs)[:,1323:1335]*10).astype(int)
  for j in np.flatnonzero(self.on&pre_alive):
   acquired=bool(iv[j,8]>0 or iv[j,11]>0)
   if done[j]:self.died_during[j]=True
   if acquired:self.success[j]=True;self.status[j]='SUCCESS';self.on[j]=False;self.end_t[j]=t+1
   elif done[j]:self.status[j]='DIED';self.on[j]=False;self.end_t[j]=t+1
   elif self.steps[j]>=16:self.status[j]='LIMIT16';self.on[j]=False;self.end_t[j]=t+1
 def metadata(self):
  return [dict(row=j,candidate=bool(self.candidate[j]),pretrigger_queries=int(self.queries[j]),deferred_queries=int(self.deferrals[j]),first_candidate_t=int(self.first_candidate[j]),plan_feasible=bool(self.feasible[j]),triggered=bool(self.started[j]),trigger_t=int(self.start_t[j]),end_t=int(self.end_t[j]),intervention_steps=int(self.steps[j]),status=str(self.status[j]),intervention_success=bool(self.success[j]),died_during=bool(self.died_during[j])) for j in range(self.n)]
def smoke():
 # Public synthetic fixture: no simulator/reset, not a performance experiment.
 ob=np.zeros((1,1345),np.float32);v=ob[0,:1323].reshape(7,9,21);v[:,:,2]=1;v[2,4,2]=0;v[2,4,11]=1;ob[0,1323:1327]=[.3,.2,.1,.1];ob[0,1340]=1;mask=np.ones((1,17),bool);pr=np.ones((1,17))/17;c=Controller(1);a,over,base,q,fire=c.decide(ob,mask,pr,np.array([0]),np.array([True]),0);assert fire[0] and over[0] and a[0]==9
 after=ob.copy();after[0,1331]=.1;c.finish(after,np.array([False]),np.array([True]),0);assert c.success[0]
 a,over,*_=c.decide(ob,mask,pr,np.array([0]),np.array([True]),1);assert not over[0] and c.steps[0]==1
 plain=ob.copy();plain[0,:1323]=0;vv=plain[0,:1323].reshape(7,9,21);vv[:,:,2]=1;plain[0,1323:1327]=.1;d=Controller(1);a,over,base,q,fire=d.decide(plain,mask,pr,np.array([0]),np.array([True]),0);assert q[0] and not over[0] and not d.started[0]
 return dict(passed=True,environment_steps=0,one_activation_verified=True,public_arguments_only=True,deferral_does_not_spend_activation=True)
if __name__=='__main__':print(smoke())
