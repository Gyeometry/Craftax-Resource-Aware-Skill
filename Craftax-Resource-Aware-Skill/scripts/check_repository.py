"""설정·자산·문서 링크·개인 경로를 오프라인으로 검사한다."""
from pathlib import Path
import ast,csv,hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'ASSET_MANIFEST.json').read_text('utf-8'))
for row in manifest:
    p=ROOT/row['path'];assert p.is_file(),row['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['path']
for name,episodes in [('phase55',32768),('phase56',6144)]:
    config=json.loads((ROOT/f'configs/{name}.json').read_text('utf-8'))
    assert config['batch']==128 and config['max_steps']==10000
    assert len(config['base_seeds'])==len(config['skill_seeds'])==8
    assert all((ROOT/f'models/PPO_{s}.msgpack').is_file() for s in config['base_seeds'])
    assert all((ROOT/f'models/Skill_{s}.msgpack').is_file() for s in config['skill_seeds'])
    with (ROOT/config['reference']).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==episodes,(name,len(rows))
    keys=[tuple(r[k] for k in ['base_seed','world','policy_root','condition']) for r in rows]
    assert len(keys)==len(set(keys)),'에피소드 키 중복'
    assert set(r['condition'] for r in rows)==set(config['conditions'])
for p in [ROOT/'run.py',ROOT/'compare.py',ROOT/'verify.py',*(ROOT/'src').glob('*.py'),*Path(__file__).parent.glob('*.py')]:
    ast.parse(p.read_text('utf-8-sig'))
links=0
for p in [ROOT/'README.md',ROOT/'AGENTS.md',ROOT/'THIRD_PARTY_NOTICES.md',*(ROOT/'docs').glob('*.md')]:
    t=p.read_text('utf-8-sig')
    assert not re.search(r'[\u3040-\u30ff]',t),p.name
    for label,link in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',t):
        if re.match(r'^[a-z]+://|^#',link):continue
        assert (p.parent/link.split('#')[0]).resolve().exists(),(p.name,link)
        links+=1
for p in ROOT.rglob('*'):
    if not p.is_file() or any(x in p.relative_to(ROOT).parts for x in ['vendor','runs','__pycache__']):continue
    assert p.name not in ['.env','id_rsa','id_ed25519'],p.name
    if p.suffix in ['.py','.md','.json','.txt','.csv','.yml','.lock']:
        t=p.read_text('utf-8-sig',errors='replace')
        assert not re.search(r'[A-Za-z]:[\\/]+Users[\\/]|/(?:root|home)/',t,re.I),p.relative_to(ROOT)
        assert not re.search(r'sk-(?:proj-)?[A-Za-z0-9_-]{24,}',t),p.relative_to(ROOT)
print(json.dumps(dict(passed=True,asset_files=len(manifest),relative_document_links=links,phase55_episodes=32768,phase56_episodes=6144,new_training_updates=0,new_environment_steps=0,remote_uploads=0),ensure_ascii=False))
