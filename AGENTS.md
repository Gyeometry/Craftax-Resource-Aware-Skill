# 작업 안내

모든 사용자 보고는 한국어로 작성한다.

먼저 README.md 및 docs/PROJECT_WORK.md를 읽는다. 검증된 src/, models/, vendor/, configs/, reference/ 파일을 요청 없이 수정하지 않는다.

GitHub 저장소는 현재 Public이다. 사용자 승인 없이 새로운 Push, Release, 라이선스 변경, 학습, 전체 Rollout 또는 유료 API 호출을 하지 않는다.

프로젝트 자체 코드의 라이선스는 PolyForm Noncommercial 1.0.0이며 외부 MIT/Apache 권리와 별도 모델 이용 조건을 구분한다.

기본 점검:
python scripts/check_repository.py
python verify.py --assets-only --out runs/asset_check.json

새 실험은 별도 코드와 새 출력 경로에서 수행한다. 기존 결과나 모델을 덮어쓰지 않는다.
