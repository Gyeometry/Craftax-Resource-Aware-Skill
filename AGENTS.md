# 작업 안내

모든 사용자 보고는 한국어로 작성한다. README와 docs/PROJECT_WORK.md를 먼저 읽는다. 검증된 src/models/vendor/configs/reference를 임의로 변경하지 않는다. 고도화는 experiments/의 별도 코드와 새 출력 경로에서 수행한다.

현재는 논문 제출 전의 로컬 준비본이다. 사용자 지시 없이 원격 저장소 생성·업로드·push·Release를 실행하지 않는다. 새 학습·전체 평가·유료 API 호출도 자동 실행하지 않는다.

기본 점검: python scripts/check_repository.py, python verify.py --assets-only --out runs/asset_check.json. 실제 소규모 환경 실행은 README의 명령과 고정 개발 월드를 사용하며 전체 성능 재현과 구분한다. 기존 결과나 파일을 덮어쓰지 않는다.
