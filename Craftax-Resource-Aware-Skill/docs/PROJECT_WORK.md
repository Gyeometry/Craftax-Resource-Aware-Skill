# 수행 내역: 재사용·수정·확장

## 기반과 직접 구현 범위

| 구분 | 내용 | 코드/근거 |
|---|---|---|
| 공식 기반 재사용 | Craftax-Classic 환경, PPO-RNN 구조 | vendor/Craftax/, vendor/Craftax_Baselines/, src/network.py |
| M2 Mask | 공개 관측으로 실제 유효 행동 지원 집합 구성 | src/masks.py |
| 선택적 Planner | 공개 국소 지도에서 설비 공동 접근을 위한 이동·설치 계획 | src/planner.py, src/controller.py |
| 접근·설치 Ablation | 최종 철 제작은 PPO에 반환 | src/ablations.py |
| 별도 Skill | CNN-GRU 모방학습으로 Planner의 상태 구성 행동 근사 | src/skill.py, models/Skill_*.msgpack |
| 시작 조건 | 공개 재료·설비 관계와 단위 설치비를 반영 | src/auto_gate.py, src/learned_resource_model.json, src/goal_formula_spec.json |
| 실행 통합 | 한 번의 Skill 호출, 최대 16 Primitive, PPO GRU 유지 | src/gated_execution.py, src/runtime.py |
| 재현 도구 | 고정 배치·RNG·모델·원래 CSV 대응 | run.py, compare.py, verify.py |

학습된 Skill은 별도 시연 기반 BC 모델이다. PPO 목적을 바꾼 새 강화학습 알고리즘으로 설명하지 않는다. 지도·제작 목표의 의미와 단위비용 합성 템플릿 등 제공된 사전지식이 있다.

## 연구 진행과 결정

| 단계 | 실제 수행 및 관찰 | 이후 결정 |
|---|---|---|
| Phase38~39-R | M2 학습과 철 제작 재료·설비 기회 감사 | 설비 공동 접근 상태 구성에 초점 |
| Phase40~42 | 동결 정책에 공개 Planner를 결합하고 정상 게임·독립 조건·Ablation 평가 | M2+C2/A1을 강한 비교 기준으로 유지 |
| Phase43~49 | 잔여 기회·사망·탐험·도구 경쟁·개입 효과의 범위 확인 | 개별 실패 사례와 전체 Return을 구분 |
| Phase50~51 | Planner 경험/시연의 직접 정책 통합 실험 | 일부 조건의 전체 Return 저하를 기록하고 기존 정책 보존 방향 검토 |
| Phase52~53 | 단일 행동과 연속 상태 구성 비교, 별도 동결 Skill 실행 | 연속 실행의 이득과 실행 실패를 함께 평가 |
| Phase54~55 | 시작 가능성·분류기·지연·자원 조건 비교 | 자원 인식형 시작이 비용과 완료율을 개선 |
| Phase56 | 설치 효과에서 자원 비용을 추출하고 시작식 구성 | 철 과제에서 AUTO/RULE 실행 동치 확인 |
| Phase57~58 | 돌 제작 목표·추가 기준선·기여 범위 점검 | 다른 목표 전체 Return 일반화는 미확정으로 유지 |

이 문서는 공동 프로젝트 전체 수행 내역이다. 개인별 작업 분담이나 저자 기여는 임의로 배정하지 않았다. 제출자의 역할 구분은 팀에서 확인해 별도로 기록한다.

## 실행 상태

현재 동결 평가 코드는 실행 가능하다. 원래 학습 데이터 전체, Optimizer 및 Runner State는 포함하지 않았다. 기존 학습 완료 결과와 이번 폴더 구성 작업을 구분하며, 새 학습 성능을 만들어 낸 것으로 보고하지 않는다.
