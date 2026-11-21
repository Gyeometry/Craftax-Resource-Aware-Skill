"""설치·모델·전이·공개 입력 검증. 학습 업데이트 없음."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets-only',action='store_true');p.add_argument('--out',default='runs/verification.json');a=p.parse_args()
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION','highest')
manifest=json.loads((ROOT/'ASSET_MANIFEST.json').read_text('utf-8'))
for row in manifest:
    f=ROOT/row['path'];assert f.is_file(),row['path'];assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'],row['path']
r=dict(passed=True,asset_files=len(manifest),training_updates=0,environment_calls=0)
if not a.assets_only:
    sys.path.insert(0,str(ROOT/'src'))
    import numpy as np,jax,jax.numpy as jnp
    import runtime as e
    from skill import NET as SKILL
    from masks import action_mask
    from controller import smoke as csmoke
    from ablations import smoke as asmoke
    from head import simple_rule
    from auto_gate import auto_gate
    r['controller_checks']=csmoke();r['access_checks']=asmoke()
    world=jnp.arange(56010001,56010001+e.B,dtype=jnp.int32);ob,st=e.reset(world)
    par=e.load(138101);sp=e.load_skill(153101);carry=e.initial(ob,st)
    hn,prob,mask,act,value=e.policy(par,carry[2],ob,world,jnp.int32(5609001),carry[4])
    sh,sl,sb=SKILL.apply(sp,jnp.zeros((e.B,128)),ob,jnp.zeros(e.B,jnp.int32))
    assert np.isfinite(np.asarray(prob)).all() and np.isfinite(np.asarray(sl)).all()
    np.testing.assert_allclose(np.asarray(prob).sum(-1),1,atol=1e-6)
    assert np.all(np.asarray(jnp.take_along_axis(mask,act[:,None],-1)))
    # 감사 전이와 원본 Craftax 전이의 전체 물리 상태, 보상, 종료, 관측 대응.
    keys=e.keys(jnp.int32(5609003),world,carry[4]);next_c,tr=e.transition(carry,hn,act,world,jnp.int32(5609003),jnp.zeros(e.B,bool))
    def original(k,s,ac):
        ns,rw=e.gl.craftax_step(k,s,ac,e.EP,e.ENV.static_env_params)
        return e.ENV.get_obs(ns),ns,rw,e.ENV.is_terminal(ns,e.EP)
    no,ns,rw,done=jax.jit(jax.vmap(original))(keys,st,act)
    for x,y in zip(jax.tree.leaves(next_c[1]),jax.tree.leaves(ns)):np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
    for x,y in [(next_c[0],no),(tr['reward'],rw),(tr['done'],done)]:np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
    r.update(devices=str(jax.devices()),jax=jax.__version__,python=sys.version,environment_calls=256,reset_calls=128,physical_state_exact=True,probability_sum_max_error=float(np.max(np.abs(np.asarray(prob).sum(-1)-1))),finite_networks=True,normal_reset_rule_auto_equal=bool(np.array_equal(np.asarray(simple_rule(ob)),np.asarray(auto_gate(ob)))))
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,ensure_ascii=False,indent=2),'utf-8');print(json.dumps(r,ensure_ascii=False))
