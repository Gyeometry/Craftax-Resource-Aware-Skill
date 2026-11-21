"""원래 평가 CSV에서 모델 단위 평균과 대응 95% 구간을 검산한다."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out).resolve();assert not out.exists(),'새 출력 경로를 지정하세요';out.mkdir(parents=True)
main=pd.read_csv(ROOT/'reference/phase55_episodes.csv')
means=main.groupby(['base_seed','condition']).raw_return.mean().unstack();rows=[]
assert means.shape==(8,8)
for c in means:
    m=means[c];d=m-means.B0;hw=float(t.ppf(.975,7)*m.sem());dh=float(t.ppf(.975,7)*d.sem())
    rows.append(dict(condition=c,raw_return=m.mean(),mean95_low=m.mean()-hw,mean95_high=m.mean()+hw,paired_difference=d.mean(),difference95_low=d.mean()-dh,difference95_high=d.mean()+dh))
computed=pd.DataFrame(rows);ref=pd.read_csv(ROOT/'results/Table_02.csv')
merged=ref.merge(computed,on='condition',suffixes=('_reference','_computed'),validate='one_to_one')
for col in computed.columns.drop('condition'):
    np.testing.assert_allclose(merged[col+'_reference'],merged[col+'_computed'],atol=1e-9,rtol=0)
computed.to_csv(out/'phase55_recomputed.csv',index=False)
means.to_csv(out/'phase55_seed_returns.csv')
secondary=pd.read_csv(ROOT/'reference/phase56_episodes.csv')
secondary.groupby(['base_seed','condition']).raw_return.mean().unstack().to_csv(out/'phase56_seed_returns.csv')
secondary.groupby('condition').raw_return.agg(['size','mean']).to_csv(out/'phase56_summary.csv')
starts=main[main.condition.eq('RULE')&main.triggered]
fail=starts[~starts.completed]
assert len(starts)==1423 and int(starts.completed.sum())==1168
fail=fail.copy();fail['reason']=pd.to_numeric(fail['reason'],errors='raise').astype(int)
assert fail.groupby('reason').size().to_dict()=={2:176,3:5,4:74}
fail.groupby('reason').size().rename('episodes').to_csv(out/'rule_failure_reasons.csv')
result=dict(passed=True,phase55_episodes=len(main),phase56_episodes=len(secondary),model_pairs=len(means),starts=1423,completed=1168,failed=255,new_environment_steps=0,new_training_updates=0)
(out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8');print(json.dumps(result,ensure_ascii=False))
