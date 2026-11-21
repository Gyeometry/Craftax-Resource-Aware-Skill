"""IR에서 추출한 하한과 독립 실행 효과 모델만 사용하는 시작 조건."""
from pathlib import Path
import json
P=Path(__file__).resolve().parent
SPEC=json.loads((P/'goal_formula_spec.json').read_text())

def decision_parts(iv,present,known,finite,candidate,xp,model=None):
    model=model if model is not None else json.loads((P/'learned_resource_model.json').read_text())
    if not model['all_identified']:return xp.zeros(candidate.shape,dtype=bool)
    costs=xp.asarray([m['cost'] for m in model['predicates']]);lower=xp.asarray(model['resource_lower'])
    need=lower+(~present).astype(costs.dtype)@costs
    return candidate&known&finite&(iv>=need).all(-1)

def auto_gate(ob):
    import jax.numpy as jnp
    from masks import extract
    from skill import facts
    v=extract(ob);candidate=facts(ob)[1]
    present=jnp.stack([v[p['helper_index']] for p in SPEC['predicates']],-1)
    return decision_parts(v[0][...,:4],present,v[SPEC['known_helper_index']],v[SPEC['valid_inventory_helper_index']],candidate,jnp)
