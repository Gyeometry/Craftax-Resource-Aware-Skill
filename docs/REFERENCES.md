# 원본과 관련 연구

## 선정 논문과 직접 실행 기반

1. Matthews et al., Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning (2024). https://arxiv.org/abs/2402.16801
2. 환경 소스: https://github.com/MichaelTMatthews/Craftax
3. 공식 PPO-RNN 기반: https://github.com/MichaelTMatthews/Craftax_Baselines

공식 기반의 네트워크·환경과 프로젝트가 추가한 Mask·Planner·Skill·시작 조건은 PROJECT_WORK.md에 구분했다. 동봉 소스는 고정본이며 원본 저장소 최신 결과를 재현했다고 주장하지 않는다. 원본 논문의 장기 학습 예산과 본 평가 체크포인트 예산을 혼합하지 않는다.

## 계산 구조상 가까운 연구

- Sutton et al., Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning (1999). https://doi.org/10.1016/S0004-3702(99)00052-1
- Bacon et al., The Option-Critic Architecture (2017). https://arxiv.org/abs/1609.05140
- Graves et al., LISPR: An Options Framework for Policy Reuse with Reinforcement Learning (2020). https://arxiv.org/abs/2012.14942
- Bagaria et al., Effectively Learning Initiation Sets in Hierarchical Reinforcement Learning (2023). https://proceedings.neurips.cc/paper_files/paper/2023/hash/e8da56eb93676e8f60ed2b696e44e7dc-Abstract-Conference.html

시간 확장 실행·시작 조건·고정 정책 재사용은 기존 계열이다. 본 프로젝트는 그 구조를 제작 준비 상태와 자원 예약에 적용해 검증한 구현·실증으로 설명한다. 위 알고리즘 전부를 직접 재현한 비교로 표기하지 않는다.
