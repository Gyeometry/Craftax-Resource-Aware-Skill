"""동결 모델 정상 게임 평가. 경로는 이 파일 위치를 기준으로 해석한다."""
from pathlib import Path
import argparse, csv, hashlib, json, os, sys, time
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION','highest')
sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',choices=['phase55','phase56'],default='phase56')
    p.add_argument('--conditions',nargs='+',choices=['B0','G0','G1','G2','G3','RULE','AUTO','A1','C2'])
    p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--worlds',type=int,help='생략하면 원래 평가 월드 수')
    p.add_argument('--world-start',type=int)
    p.add_argument('--rng-pairs',type=int,choices=[1,2],default=2)
    p.add_argument('--max-steps',type=int,default=10000)
    p.add_argument('--out',required=True)
    p.add_argument('--trace',action='store_true',help='유효 에피소드의 축약 전이 기록 저장')
    a=p.parse_args()
    c=json.loads((ROOT/f'configs/{a.protocol}.json').read_text())
    seeds=a.seeds or c['base_seeds'];conditions=a.conditions or c['conditions']
    if not set(seeds)<=set(c['base_seeds']):p.error('배포된 모델의 seed만 사용 가능')
    worlds=c['worlds'] if a.worlds is None else a.worlds
    if worlds<1 or not 1<=a.max_steps<=10000 or a.max_steps%16:p.error('worlds>=1, max-steps는 16의 배수이며 10000 이하')
    start=c['world_start'] if a.world_start is None else a.world_start
    out=Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):p.error('결과 덮어쓰기 금지: 빈 출력 경로를 지정하세요')
    out.mkdir(parents=True,exist_ok=True)
    import numpy as np,jax,jax.numpy as jnp
    from flax import serialization
    import runtime as e,execution as x,gated_execution as gx
    from head import initialize
    from controller import Controller
    from ablations import AccessController
    begin=time.perf_counter();rows=[];costs=[]
    for seed in seeds:
        sk=c['skill_seeds'][c['base_seeds'].index(seed)]
        par=e.load(seed);sp=e.load_skill(sk)
        model_hash=hashlib.sha256((ROOT/f'models/PPO_{seed}.msgpack').read_bytes()).hexdigest()
        for condition in conditions:
            hp=jax.tree.map(jnp.asarray,serialization.msgpack_restore((ROOT/'models/head_model.msgpack').read_bytes())) if condition=='G2' else initialize()
            for pair in range(a.rng_pairs):
                pr,er,sr=[c[k][pair] for k in ['policy_roots','environment_roots','skill_roots']]
                for offset in range(0,worlds,e.B):
                    n=min(e.B,worlds-offset);world=jnp.arange(start+offset,start+offset+e.B,dtype=jnp.int32)
                    ob,st=e.reset(world);base=e.initial(ob,st,valid=jnp.arange(e.B)<n);carry=gx.initial(base)
                    ctrl=Controller(e.B) if condition=='C2' else AccessController(e.B) if condition=='A1' else None
                    total=np.zeros(n);discounted=np.zeros(n);length=np.zeros(n,int);forced_count=np.zeros(n,int);trace=[];calls=0;tick=time.perf_counter();triggered=np.zeros(n,bool)
                    for t in range(0,a.max_steps,16):
                        if ctrl is None:
                            if condition=='B0':base,tr=jax.block_until_ready(e.chunk(par,base,world,jnp.int32(pr),jnp.int32(er)))
                            else:
                                carry,tr=jax.block_until_ready(gx.gated_chunk(par,sp,hp,carry,world,jnp.int32(pr),jnp.int32(er),jnp.int32(sr),condition));base=carry[0]
                            calls+=e.B*16
                        else:
                            blocks=[]
                            for step in range(16):
                                hn,prob,mask,sa,value=jax.block_until_ready(e.policy(par,base[2],base[0],world,jnp.int32(pr),base[4]))
                                alive=np.asarray(base[3]);act,ff,*_=ctrl.decide(np.asarray(base[0]),np.asarray(mask),np.asarray(prob),np.asarray(sa),alive,t+step)
                                base,tr0=jax.block_until_ready(e.transition(base,hn,jnp.asarray(act),world,jnp.int32(er),jnp.asarray(ff)))
                                ctrl.finish(np.asarray(base[0]),np.asarray(tr0['done']),alive,t+step)
                                blocks.append(tr0);calls+=e.B
                                if not bool(base[3].any()):break
                            tr=jax.tree.map(lambda *v:jnp.stack(v),*blocks)
                        # 큰 관측 배열을 저장하지 않고 유효 에피소드만 집계한다.
                        keys=['active','action','reward','done','forced','new_achievements','health','post_health','grid','mob','obs_tail']
                        if 'skill_trigger' in tr:keys+=['skill_trigger','skill_executed']
                        data={k:np.asarray(tr[k][:,:n]) for k in keys}
                        active=data['active'];rewards=data['reward']
                        expected=np.where(active,data['new_achievements'].sum(-1)+.1*(data['post_health']-data['health']),0)
                        if not np.allclose(rewards,expected,atol=1e-6,rtol=0):raise AssertionError('보상 회계 불일치')
                        total+=rewards.sum(0);discounted+=(rewards*c['discount']**(np.arange(len(rewards))+t)[:,None]).sum(0)
                        length+=active.sum(0);forced_count+=data['forced'].sum(0)
                        if 'skill_trigger' in data:triggered|=data['skill_trigger'].any(0)
                        if a.trace:trace.append(data)
                        if not bool(base[3].any()):break
                    ach=np.asarray(base[5].achievements);alive=np.asarray(base[3]);sec=time.perf_counter()-tick
                    if ctrl is not None:triggered=ctrl.started[:n]
                    for j in range(n):
                        row=dict(protocol=a.protocol,condition=condition,base_seed=seed,skill_seed=sk,world=int(world[j]),policy_root=pr,environment_root=er,skill_root=sr,raw_return=float(total[j]),discounted_return=float(discounted[j]),length=int(length[j]),completed=not bool(alive[j]),triggered=bool(triggered[j]),intervention_steps=int(forced_count[j]),checkpoint_sha256=model_hash)
                        row.update({f'achievement_{k}':bool(ach[j,k]) for k in range(22)})
                        rows.append(row)
                    if trace:
                        np.savez_compressed(out/f'trace_{condition}_{seed}_{pr}_{start+offset}.npz',**{k:np.concatenate([d[k] for d in trace]) for k in trace[0]},world=np.asarray(world[:n]))
                    costs.append(dict(condition=condition,base_seed=seed,policy_root=pr,world_start=start+offset,actual_environment_calls=calls,active_steps=int(length.sum()),seconds=sec,planner_seconds=ctrl.plan_seconds if ctrl else 0))
                    print(json.dumps(dict(condition=condition,seed=seed,episodes=n,complete=int((~alive[:n]).sum()),mean_partial_return=float(total.mean()),seconds=round(sec,3))),flush=True)
                    with (out/'episodes.csv').open('w',newline='',encoding='utf-8') as f:
                        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                    (out/'costs.json').write_text(json.dumps(costs,indent=2),encoding='utf-8')
    result=dict(arguments=vars(a),config=c,devices=str(jax.devices()),jax=jax.__version__,python=sys.version,episodes=len(rows),complete=sum(r['completed'] for r in rows),training_steps=0,paid_api_calls=0,seconds=time.perf_counter()-begin,summary={m:dict(episodes=sum(r['condition']==m for r in rows),completed=sum(r['condition']==m and r['completed'] for r in rows),mean_complete_return=float(np.mean([r['raw_return'] for r in rows if r['condition']==m and r['completed']])) if any(r['condition']==m and r['completed'] for r in rows) else None) for m in conditions})
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
