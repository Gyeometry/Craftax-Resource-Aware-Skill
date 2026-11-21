# Craftax Resource-Aware Skill

동결 PPO-RNN, 설비 공동 접근 Planner 및 학습 Skill의 정상 게임 평가를 실행하는 저장소다. 코드뿐 아니라 실제 동결 모델, 고정된 평가 설정, 원래 에피소드 결과 및 비교 도구를 포함한다. 원본 연구 폴더나 별도 모델 다운로드가 필요 없다.

## 1. 설치

Python **3.12**의 새 환경을 사용한다. Windows CPU 또는 Linux/WSL CPU를 기본 실행 경로로 둔다. 압축을 풀고 `run.py`가 있는 폴더에서 실행한다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c requirements-cpu.lock
$env:JAX_PLATFORMS = "cpu"
$env:JAX_DEFAULT_MATMUL_PRECISION = "highest"
.\.venv\Scripts\python.exe verify.py --out runs/verification.json
```

이미 Python 3.12 Conda 환경을 만들었다면 위 명령의 `.\.venv\Scripts\python.exe`를 `python`으로 바꿔 실행한다. Linux/WSL에서는 `python3.12 -m venv .venv` 후 `.venv/bin/python`을 사용한다. 새 패키지 설치에는 인터넷이 필요하지만 Craftax 소스와 모델은 이미 포함되어 있다. 기존 공유 파일로 만든 환경을 사용하더라도 버전과 검증 결과를 확인한다.

`requirements-cpu.lock`은 고정된 전체 패키지 버전이고 `requirements.txt`는 직접 의존성이다. 이번 배포 검증은 이미 설치된 Linux/WSL Python 3.12.3, JAX 0.6.2, Flax 0.10.6의 CPU 환경에서 수행했다. 완전히 새 Windows 환경의 설치를 이번에 별도로 수행한 것은 아니다.

## 2. 정상 게임 실행

먼저 실제 개입이 있었던 공개 개발 월드 한 개를 실행한다. 학습은 수행하지 않는다.

```powershell
python run.py --protocol phase56 --conditions B0 G0 RULE AUTO A1 C2 --seeds 138101 --world-start 56010003 --worlds 1 --rng-pairs 1 --out runs/check_world3
python compare.py runs/check_world3/episodes.csv --protocol phase56 --out runs/check_world3_comparison
```

이 명령의 배포 검증 결과는 다음과 같다. 단일 월드는 설치·실행 확인용이며 평균 성능의 근거가 아니다.

| 조건 | 실제 Raw Return | 에피소드 길이 | 개입 Primitive 수 |
|---|---:|---:|---:|
| B0 | 15.10 | 265 | 0 |
| G0 | 17.10 | 305 | 3 |
| RULE | 17.10 | 305 | 3 |
| AUTO | 17.10 | 305 | 3 |
| A1 | 17.10 | 375 | 4 |
| C2 | 17.10 | 375 | 5 |

Return은 비교 허용오차 1e-6 이내, 길이 및 업적은 정확히 일치했다. 개입 중 실제 시간·자원·생존 손실을 포함하며, `A1`과 학습 Skill은 철 도구 제작 자체를 강제하지 않는다.

`--max-steps 32`는 짧은 실행 확인에만 사용한다. 종료하지 못한 에피소드는 `completed=false`이며 최종 평균 Return에서 제외된다. 기본값 10000은 원래 평가 상한이다. 출력 경로는 비어 있어야 하며 기존 결과를 덮어쓰지 않는다.

## 3. 본 평가 재현

Phase 55 본 평가의 원래 모델 8개·월드 256개·정책/환경 RNG 두 쌍을 그대로 사용한다. 조건당 4096게임이다.

```powershell
python run.py --protocol phase55 --conditions B0 G0 RULE A1 C2 --out runs/phase55_main
python compare.py runs/phase55_main/episodes.csv --protocol phase55 --out runs/phase55_comparison
```

모든 시작 기준선까지 실행하려면 다음을 사용한다. 총 32768게임으로 CPU에서는 오래 걸릴 수 있다.

```powershell
python run.py --protocol phase55 --out runs/phase55_all
```

Phase 56 AUTO 평가의 원래 설정은 조건당 1024게임이다.

```powershell
python run.py --protocol phase56 --out runs/phase56_all
python compare.py runs/phase56_all/episodes.csv --protocol phase56 --out runs/phase56_comparison
```

원래 평균은 `reference/` 및 논문용 표에 보존되어 있다. Phase 55에서는 B0 **14.26577**, G0 **14.39053**, RULE **14.49502**, A1 **14.55605**, C2 **14.56704**였다. Phase 56의 다른 월드 결과를 이 평균에 섞지 않는다.

재현 설정은 `configs/`에 있다. Phase 55 확인 평가의 RNG에는 원래 worker의 +10000 offset을 이미 반영했다. 실행 배치는 원본과 같은 **128**로 고정하며 한 월드 실행도 나머지를 무효 패딩한다. 작은 배치로 바꾸면 수치 연산이 달라질 수 있다. `--seeds`, `--worlds`, `--rng-pairs`를 줄인 결과는 전체 실험 재현으로 표기하지 않는다.

## 4. 저장 결과와 비교

- `episodes.csv`: 모델·월드·RNG별 Return, 할인 Return, 길이, 개입 수, 22개 업적, 완료 여부.
- `result.json`: 실제 디바이스, 버전, 실행 설정, 조건별 완료 게임 평균.
- `costs.json`: 패딩을 포함한 실제 환경 호출 수와 유효 스텝, 계산시간.
- `--trace`: 공개 grid/mob, 실제 행동·보상 등의 축약 궤적. 전체 학습 전이나 복원용 Snapshot이 아니다.
- `compare.py`: 대응 키로 원자료와 비교한다. 일치하지 않는 행과 중도 종료를 숨기지 않는다.

다른 CPU/GPU·컴파일러에서 부동소수점 차이가 행동 샘플링과 이후 궤적을 바꿀 수 있으므로 모든 컴퓨터에서 비트 단위 일치를 보장하지 않는다. 먼저 고정 소규모 검증을 확인하고, 전체 성능은 같은 8개 모델과 동일 월드·RNG로 비교한다. 에피소드 수를 독립 학습 Seed 수로 취급하지 않는다.

## 5. 기존 결과 표 검산과 범위

```powershell
python scripts/summarize_reference.py --out runs/reference_summary
```

이 명령은 기존 CSV의 모델 단위 통계와 핵심 실패 종료 수를 검산한다. 실제 환경 실행과 별개다. 전체 160개 평가 NPZ, 학습 데이터 및 Optimizer State는 이 저장소의 즉시 실행 범위에 포함하지 않는다.

원본 출처는 PROVENANCE.json, 실행 자산 해시는 ASSET_MANIFEST.json에 있다. 구조와 수행 내역은 현재 루트 README 및 docs/를 따른다. 기본 정책의 가중치와 M2 Mask를 유지한 추론·평가 코드이며 From-scratch 학습을 자동 시작하지 않는다.
