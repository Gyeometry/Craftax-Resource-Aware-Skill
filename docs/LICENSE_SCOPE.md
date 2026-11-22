# 라이선스 적용 범위

## 프로젝트 자체 코드: PolyForm Noncommercial 1.0.0

저작권자: Gyeom Kim (2026)
적용 문서: [LICENSE](../LICENSE)

프로젝트가 독립적으로 구현한 평가 실행 코드, Planner, Mask, Skill, Controller, AUTO 자원 조건, 검증 도구 및 기존 코드에 대한 자체 수정분에만 적용한다. 허용된 범위는 공식 라이선스 조항을 따른다. 상업적 이용에는 별도 허락이 필요하다.

## 기존 코드 및 파생 코드

- vendor/Craftax 및 그 원본 로직 유래 부분: Michael Matthews의 MIT 조건 유지
- vendor/Craftax_Baselines 및 PPO-RNN 원본 유래 부분: Michael Matthews의 MIT 조건 유지
- PureJaxRL 유래 부분: Apache-2.0 관련 원본 조건 및 귀속 고지 유지
- Gymnax 유래 부분: Apache-2.0 관련 원본 조건 및 귀속 고지 유지
- src/network.py 및 src/instrumented_logic.py는 원본 유래 코드와 프로젝트 수정분을 구분하여 각각의 적용 권리를 존중한다.

자체 코드에 부여한 제한으로 기존 제3자 MIT/Apache 권리를 취소하거나 축소하지 않는다. 기존에 적법하게 부여된 MIT 권리도 소급 철회하는 것으로 해석하지 않는다.

## 별도 자산

학습된 PPO/Skill/Head 체크포인트는 자동으로 소스 코드 라이선스에 포함되지 않는다. 교육·연구 목적의 모델 이용 및 재배포에 관한 설명은 [모델 사용 조건](../models/LICENSE.md)을 따른다.

평가 CSV, 이미지, 논문 원고 등은 별도 권리 대상이다.

## 기타

[제3자 출처와 고지](../THIRD_PARTY_NOTICES.md)
[공개 상태](PUBLICATION_STATUS.md)

이 문서는 파일별 권리 관계에 관한 프로젝트 설명이며, 모든 역사적 저작권 관계를 보증하는 법률 의견은 아니다.
