# 재현 조건

- 환경: Craftax-Classic-Symbolic-v1, auto_reset=False, 고정 소스 동봉.
- Python3.12, JAX0.6.2, Flax0.10.6, 나머지 버전은 requirements-cpu.lock.
- 공개 관측1345, 행동17, PPO GRU512, 별도 Skill GRU128.
- 실행 배치128 고정. 소수 월드 실행은 무효 슬롯을 패딩한다.
- 실제 환경/정책/Skill RNG는 별도 root와 월드·절대 시점으로 fold한다.
- Phase55 확인용 worker의 +10000 root offset은 configs/phase55.json에 반영되어 있다.
- 모든 조건은 동일 모델·월드·정책/환경 난수를 대응시킨다. 개입 후 궤적이 달라질 수 있다.
- 기본 상한10000, 실제 종료까지 평가. 미종료 게임은 완료 평균과 구분한다.
- 주지표 실제 Raw Return, 보조 할인율0.99. 숨겨진 GT를 정책 입력에 넣지 않는다.

## 검증 수준

2026-11-14의 기존 CPU 검증에서 두 모델·세 월드의 총18조건 에피소드는 Return(1e-6), 길이,22업적이 원래 기록과 일치했다. 근거는 validation/20261114/에 있다. 이는 전체 평가를 새로 반복한 결과나 새 Windows 설치 검증이 아니다.

이번 업로드 준비에서는 복사 자산 해시·CLI·문서·표 검산을 확인한다. 새 PPO 학습이나 전체 게임 평가를 자동 실행하지 않는다. Python/JAX/Flax·CPU/GPU에 따른 수치 차이를 고려해 실제 출력의 디바이스와 버전을 함께 보존한다.

설치·전체 실행 명령은 EXECUTION_GUIDE.md에 있다. 해당 문서의 오래된 paper/evidence 폴더 설명 대신 현재 README의 구조를 따른다. 추론용 모델 파라미터와 학습 완전 재개 체크포인트는 서로 다르다.
