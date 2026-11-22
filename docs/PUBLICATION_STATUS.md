# 업로드 및 공개 상태

2026-11-22 기준 저장소 Craftax-Resource-Aware-Skill은 GitHub Public 상태로 공개되어 있다.

Repository: https://github.com/Gyeometry/Craftax-Resource-Aware-Skill

## 공개 범위

- 동결 PPO 및 Skill 평가 코드, Planner와 자원 인식형 시작 조건
- 평가용 체크포인트, 고정 평가 설정, 재현 명령
- 기존 정상 게임 평가 CSV, 주요 결과 및 검산 도구
- 기존 환경·PPO 소스와 제3자 라이선스 고지

미제출 원고 전문, 전체 내부 연구일지, 원래 전체 학습 데이터, Optimizer State, 개인 경로 및 비밀키는 포함하지 않는다.

## 라이선스

프로젝트 자체 구현 코드와 권리를 보유한 수정분은 [PolyForm Noncommercial 1.0.0](../LICENSE) 적용 대상이다. 상업적 이용에는 별도 허락이 필요하다.

Craftax, Craftax_Baselines, PureJaxRL, Gymnax 등의 원본 소스는 각 기존 MIT/Apache-2.0 라이선스를 유지한다.

모델 가중치의 별도 이용 조건은 [models/LICENSE.md](../models/LICENSE.md)를 따른다.

논문이 Accept되었다는 사실만으로 라이선스가 자동 변경되지 않는다. 이후 라이선스 확대는 별도의 명시적 수정으로 이루어진다.

## 재현 범위

기존 체크포인트를 불러와 평가 및 원본 CSV 대응 비교를 수행할 수 있다. 처음부터의 전체 정책·Skill 재학습은 이 저장소의 자동 실행 범위가 아니다.

학교 프로젝트 제출 및 교육 목적의 재현을 지원하기 위해 공개했으며, 논문 심사 완료나 게재를 주장하지 않는다.
