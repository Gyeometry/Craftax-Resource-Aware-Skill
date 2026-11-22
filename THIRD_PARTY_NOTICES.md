# 제3자 소스 및 라이선스

본 저장소의 자체 구현 부분은 PolyForm Noncommercial License 1.0.0을 따른다. 이 제한은 제3자 원본 코드의 기존 허가를 변경하지 않는다.

## Craftax
- Source: https://github.com/MichaelTMatthews/Craftax
- Copyright (c) 2024 Michael Matthews
- License: MIT (vendor/Craftax/LICENSE)
- Includes Craftax game/environment sources and original logic underlying src/instrumented_logic.py.

## Craftax_Baselines
- Source: https://github.com/MichaelTMatthews/Craftax_Baselines
- Copyright (c) 2024 Michael Matthews
- License: MIT (vendor/Craftax_Baselines/LICENSE)
- src/network.py derives from its recurrent PPO architecture with project-specific modifications.

## PureJaxRL ancestry
- Source: https://github.com/luchris429/purejaxrl
- Upstream contribution: Copyright 2023 Chris Lu
- License: Apache License 2.0
- The PPO baseline explicitly credits PureJaxRL. See LICENSES/PureJaxRL-Apache-2.0.txt and LICENSES/Apache-2.0.txt.

## Gymnax ancestry
- Source: https://github.com/RobertTLange/gymnax
- Upstream attribution: Copyright 2020 Rémi Louf as contained in the provided license copy
- License: Apache License 2.0
- The Craftax environment base explicitly credits Gymnax. See LICENSES/Gymnax-Apache-2.0.txt and LICENSES/Apache-2.0.txt.

## 프로젝트 코드와 모델
Independent Planner, Skill, control, evaluation, and resource-initiation code belongs to the project and is licensed within the ownership scope of Gyeom Kim under PolyForm Noncommercial 1.0.0.

Model weights have a distinct permission notice in models/LICENSE.md. Training provenance is in models/PROVENANCE.json and the root provenance records.

The exact historical revisions of the upstream PureJaxRL and Gymnax components were not established. These notices preserve documented ancestry without asserting a verified upstream commit.

For component-specific scope see [LICENSE_SCOPE.md](docs/LICENSE_SCOPE.md).
