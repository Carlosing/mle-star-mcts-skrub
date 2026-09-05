# Mechanism comparison

| Axis | MCTS-skrub (ours) | AutoGluon | MLE-STAR |
|---|---|---|---|
| Search mechanism | MCTS over a skrub-DataOps space (structure + HPs) | Ensemble/stacking of many pretrained model configs | LLM writes + iteratively refines Python code |
| LLM call count | O(1) per task (2 + N_PROPOSES), fixed & known up front | 0 (no LLM) | Unbounded: ~26 best case, 1000+ with debug cascade |
| Cost scales with | CV rollouts / wall-clock (pure code) — LLM cost constant | Wall-clock (pure code) | LLM calls (data-dependent debug retries) |
| Leakage handling | skrub DataOps: transforms fit inside CV folds by construction | AutoGluon internal CV / bagging | Depends on the code the LLM generates (no guarantee) |
| Adaptivity | Space authored once; search adapts within it (Optional Feature 1/3) | Fixed AutoML pipeline | Fully adaptive per step (at unbounded token cost) |
| Relational tables | Yes — AggJoiner over aux_*.csv is a searchable stage | No — flat table only | Possible if the LLM writes the join |
