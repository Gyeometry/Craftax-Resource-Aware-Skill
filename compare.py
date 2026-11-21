"""저장된 에피소드와 원래 대응 평가를 비교한다. 누락과 중복을 숨기지 않는다."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('episodes');p.add_argument('--protocol',choices=['phase55','phase56'],default='phase56');p.add_argument('--out',required=True);a=p.parse_args()
got=pd.read_csv(a.episodes);ref=pd.read_csv(ROOT/f'reference/{a.protocol}_episodes.csv')
keys=['base_seed','world','policy_root','condition']
assert not got.duplicated(keys).any(),'현재 평가 키 중복'
assert not ref.duplicated(keys).any(),'원자료 키 중복'
m=got.merge(ref,on=keys,how='left',suffixes=('_new','_reference'),indicator=True,validate='one_to_one')
m['return_difference']=m.raw_return_new-m.raw_return_reference
m['return_exact_1e6']=np.isclose(m.raw_return_new,m.raw_return_reference,atol=1e-6,rtol=0)
m['length_exact']=m.length_new==m.length_reference
m['achievements_exact']=np.logical_and.reduce([m[f'achievement_{k}_new']==m[f'achievement_{k}_reference'] for k in range(22)])
out=Path(a.out);assert not out.exists(),'새 출력 경로를 지정하세요';out.mkdir(parents=True)
m.to_csv(out/'paired_episodes.csv',index=False)
by_seed=m[m._merge.eq('both')&m.completed_new].groupby(['base_seed','condition']).agg(mean_difference=('return_difference','mean'),episodes=('world','size')).reset_index()
by_seed.to_csv(out/'seed_differences.csv',index=False)
summary=dict(evaluated=len(got),matched=int(m._merge.eq('both').sum()),unmatched=int((~m._merge.eq('both')).sum()),censored=int((~got.completed).sum()),exact_return_count=int(m.return_exact_1e6.sum()),exact_length_count=int(m.length_exact.sum()),exact_achievement_count=int(m.achievements_exact.sum()),note='표본 일부의 일치가 전체 평가 재현 또는 환경 간 동등성을 보장하지 않는다.')
(out/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8');print(json.dumps(summary,ensure_ascii=False))
