# Future work

Deliberately unimplemented improvements, recorded at project close (2026-07-13).
None of these change the load-bearing invariant — **the LLM proposes the space,
pure code searches it, LLM calls stay ≤ O(outer steps)** — they change *which*
pure-code searcher runs inside it and how the space evolves between slices.
Status/context: [PROJECT_STATE.md](PROJECT_STATE.md).

---

## 1. Replace the MCTS engine with an off-the-shelf HPO framework

### Why

`mcts.py` was built to search structure + HPs jointly, but at our budgets
(40–100 rollouts over spaces with dozens of dimensions) the tree rarely
develops past depth 2–3 — the engine operates closer to guided sampling than
to tree search. Mature HPO frameworks (Optuna/TPE, SMAC3/ConfigSpace) are
model-based samplers designed exactly for this budget/dimensionality regime,
and they come with the analysis machinery (parameter importances, trial
dataframes) that our tree-mined ablation approximates by hand.

The architecture already permits the swap cleanly: the searcher's inputs are
`get_action_space(plan)` (flat dict of named discrete choices),
`get_choice_gating(plan)` (model → active-HP conditionals), and a
deterministic bounded-[0,1] rollout function. Nothing outside
`search_loop.run_search_loop` knows the searcher is a tree.

### Mapping

| Ours | Framework equivalent |
|---|---|
| `get_action_space` dict | Optuna define-by-run `suggest_categorical` / ConfigSpace hyperparameters |
| `get_choice_gating` + `canonicalize` (inactive HPs dropped) | Optuna conditional suggests / ConfigSpace `EqualsCondition` — native; delete our gating code |
| `mcts.score_cache` (exact, deterministic rollouts) | Study/run history storage; keep our cache as a memo layer so a repeated config never re-rolls |
| Persistent tree across outer slices | Persistent study + **ask-and-tell** interface (both frameworks support it) |
| `tree_action_values` (variance ledger) | fANOVA / MDI parameter importances + `study.best_trials` |
| `start_node` focused-refinement bonus phase | Local search around incumbent (SMAC does this natively) or a seeded neighborhood enqueue |

Constraints to preserve, verbatim from the current design:

- **Determinism** — seeded subsamples + seeded estimators; the sampler itself
  must be seeded too, or the score cache stops being exact.
- **Bounded higher-is-better reward**; the 60s wall-clock cap → 0.0 stays.
- **Dynamic space growth** — Optional Feature 3 injections extend categorical choice
  sets mid-run. Optuna's define-by-run space handles a growing option list
  naturally (old trials remain valid `tell`s); ConfigSpace does **not** like a
  mutating space — with SMAC you'd rebuild the space and re-`tell` the cached
  history. This asymmetry probably decides the framework: **Optuna
  ask-and-tell is the path of least resistance.**

### Reporting back to the LLM (the part that must not be lost)

The proposer prompt currently carries the cross-stage tree ledger + incumbent
config/score. The framework replacement is strictly richer and should feed the
same ≤-one-call-per-slice proposer:

- **Top-k trials** (config + reward) from the study — replaces the incumbent
  line, generalizes it.
- **Parameter importances** (fANOVA) — replaces `pick_target_node`'s
  variance ledger as the `target_stage` hint. Notably, this may fix the
  known null result: the ledger elected `model` on essentially every pick
  (see PROJECT_STATE, Optional Feature 1); fANOVA importance over the study history is
  a far better-calibrated signal for "which stage is worth extending".
- **Marginal distributions per choice** (e.g. "GapEncoder won 7/9 trials it
  appeared in") — cheap to compute from the trial dataframe, compact to
  serialize, and much more informative to the proposer than raw Q/N.

Keep the serialization compact (a state-dict-like summary, never trial
transcripts) — same rule as today.

### Validation order

Before adopting any framework, run the missing baseline: **seeded random
search over the same resolved space, same rollout fn, same budget, same top-k
read-off.** It's ~30 lines against the existing plumbing. If random search
matches MCTS (plausible at these budgets), the framework swap is justified by
the analysis/reporting machinery alone; if MCTS wins, that's the figure the
current engine never got.

---

## 2. Revamp the warm start

The current `prior_fn` mechanism (plan options carry `"prior": 0.0–1.0`;
`_make_prior_fn` seeds fresh children, neutral 0.5 default) is UCT-specific
and dies with the engine. Two replacement designs, not mutually exclusive:

### 2a. Framework-native prior encoding

- **Optuna:** `study.enqueue_trial(config)` — the LLM's plan already implies
  its favorite configs (default-first ordering exists precisely because of
  this). Have `resolve_spec` emit the plan's top-prior full configurations and
  enqueue them as the first trials. Zero new LLM contract; the `prior` field
  is reinterpreted rather than removed.
- **SMAC3:** πBO-style user priors — actual prior *distributions* over the
  space, the theoretically clean version of what `prior_fn` approximated.
  More faithful, but inherits the static-ConfigSpace problem above.

The enqueue approach is strictly simpler and preserves the current semantics:
the search starts at skrub's robust root + the LLM's best guesses, then the
sampler takes over.

### 2b. LLM proposes space **cuts**, not only additions

`_merge_raw_plans` is strictly additive today because persisted *tree states*
must stay appliable — removing an option would orphan nodes. With a
study-based engine that constraint dissolves: cached trials referencing a
removed option simply remain as history (`tell`s); they never need to be
re-applied. That unlocks a two-sided proposer contract:

```json
{"extend": {<current raw-plan shape>},
 "prune":  {"model": ["sklearn.svm.SVR"],
            "scope_geo_cluster": "*"}}
```

Safety rules for `prune` (code-owned, like everything structural):

- Never prune an operator present in the current incumbent or any enqueued
  top-k state.
- Never prune a backbone default (the robust root must stay reachable).
- Prunes go through the same resolve → rebuild path as extensions; a prune
  that would empty a stage falls back to the stage's default entry.

This is the missing half of Optional Feature 3: at small budgets, *shrinking* a bloated
LLM-authored space is plausibly worth more rollouts than extending it — every
pruned dead option redirects budget to live ones.

---

## 3. Minor improvements

### 3a. Ablate the token-heavy prompt components

The pipeline now logs tokens per run (`result["tokens"]`), and the replay
harness (`scripts/claude_agents.py`, `run_pipeline(spec_raw=...)`) makes plan
A/Bs quota-free. Nobody has measured whether the expensive prompt parts earn
their tokens:

- **Data digest size:** full `make_data_summary` vs a truncated digest vs
  schema-only, measured by downstream *plan quality* (holdout of the resulting
  run), not by plan plausibility. Observed run-to-run variance is already
  dominated by plan authorship (toxicity: 0.78–0.95 across authors at equal
  budget), so this ablation is measuring the thing that actually matters.
- **Proposer context:** full ledger + digest + column names vs incumbent-only.
  The proposer prompt is the second-largest token sink (country-happiness
  runs: ~48k tokens with 4 calls vs ~15k with 2).

### 3b. Score plan extensions by the difference they add

Injections are currently accepted wholesale (strictly additive merge) and
their value is only visible indirectly. Two cheap upgrades:

- **Attribution:** log, per injected option, whether it was ever selected in
  a rollout, entered the top-k pool, or survived into the incumbent /
  ensemble members. An injection hit-rate per task is one dict-comprehension
  over the score cache and turns "Optional Feature 3 helps" from an anecdote (the
  country-happiness QuantileTransformer rescue) into a measured rate.
- **Diff-gated injection:** before merging, diff the proposed plan against
  the current space and rank additions by novelty along dimensions the
  importance analysis (§1) marks as high-value; drop additions that are
  near-duplicates of existing options (same estimator class, overlapping HP
  range). The proposer prompt can also be told *what already exists* more
  aggressively — today near-duplicate proposals cost budget to disprove.

### 3c. Housekeeping carried over from PROJECT_STATE

- `llm_calls` miscounts offline replay proposers as real calls (known, owed).
- `scoped_encodings` was dropped as malformed in every country-happiness run —
  the plan-schema failure is surfaced (`dropped_sections`) but its cause
  (planner emits a shape the resolver rejects) was never chased down; a
  schema example in the planner prompt is probably a one-line fix.
- AggTarget as a guarded assemble option (fold-safe by construction; explicit
  Week-2 stretch, never taken).
