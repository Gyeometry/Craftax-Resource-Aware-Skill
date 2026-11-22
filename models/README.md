# Frozen evaluation checkpoints

Included evaluation assets:
- 8 frozen PPO-RNN checkpoint files named PPO_138101.msgpack through PPO_138108.msgpack.
- 8 separately trained state-construction skill files named Skill_153101.msgpack through Skill_153108.msgpack.
- 1 auxiliary initiation classifier checkpoint named head_model.msgpack.

The PPO exports originate from locally trained policies at 10,485,760 environment steps and were frozen for evaluation. The skill models were trained using behavior cloning on A1 planner demonstrations. The classifier was trained on public observations and planning-feasibility labels.

The learned termination head is not used as the final execution termination rule for the main proposed configuration.

For model-level hashes and origin records, see [PROVENANCE.json](PROVENANCE.json). The root [PROVENANCE.json](../PROVENANCE.json) records the broader project provenance. Rights and permissions are described in [LICENSE.md](LICENSE.md).

These files enable evaluation of existing frozen checkpoints, not full from-scratch training reproduction.
