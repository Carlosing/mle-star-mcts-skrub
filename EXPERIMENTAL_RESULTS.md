# EXPERIMENTAL_RESULTS.md

Benchmark results for the MCTS-skrub extension against AutoGluon and the revived
MLE-STAR, on 10 tasks. Underlying data: `figures/comparison.csv`, the per-run
`result.json` artifacts under `results/`, and the figures under `figures/`. To
reproduce, see [`EXPERIMENTS.md`](EXPERIMENTS.md).

---

## Status of the numbers

**A change to the train/test split invalidated the basis of every benchmark run,
and the three methods were not re-run afterwards.**

On **2026-07-14 17:13**, commit `517f954` moved the train/holdout split *onto
disk*, drawn once by `scripts/stage_tasks.py` before any method reads the data —
giving all three methods one shared evaluation set, referred to below as *the
bench*. Before that commit there was no shared bench: each method carved its own
25% holdout out of `train.csv` at run time. Every archived benchmark run — all 10
extension runs, all 11 AutoGluon runs, all 13 MLE-STAR runs — was produced
**before** that commit.

**All three methods are therefore biased in the same direction**, for overlapping
reasons:

| Method | Why its archived number is optimistic |
|---|---|
| **Extension** | The search cross-validated over rows that later became its own eval set, so it *selected* its pipeline with the reported rows visible. It also refit on 75% of a `train.csv` that was, for two tasks, larger than today's. |
| **AutoGluon** | Never saw its holdout — clean with respect to selection — but it was **never run against the on-disk split**. It carved its own validation split from a larger `train.csv` than the current bench provides. |
| **MLE-STAR** | Reports a **validation score it carved out of `train.csv` itself**, not a holdout score. Its scripts are explicitly instructed not to touch `test.csv` (teammate's report, [`docs/mle_star_implementation_report.pdf`](docs/mle_star_implementation_report.pdf) §2.7, §5.4). |

No number in the archived comparison sits on the bench that ships in this
repository today, and the three columns do not share a scoring basis.

**The only numbers measured on the current split** are four zero-quota *replays* —
a captured plan re-run through the fixed pipeline at no API cost (§1).

Scope of the damage: **the score comparison is indicative only; the capability
findings (§3) and the cost model (§4) are unaffected.**

---

## What the benchmark compares

Three approaches to the same problem — *given a dataset and a task description,
produce a trained model* — on 10 tasks.

| Method | Mechanism | LLM role |
|---|---|---|
| **Extension** (ours) | MCTS over a skrub DataOps pipeline space — structure **and** hyperparameters | Authors the search *space*, once. Never runs the search. |
| **AutoGluon** | Ensembling/stacking of many pretrained model configurations | None |
| **MLE-STAR** | An LLM writes and iteratively debugs Python code | Writes every line, in a debug cascade |

### What was held constant, and what was not

| | Held constant? |
|---|---|
| The 10 tasks | ✅ yes |
| The task descriptions and metrics | ✅ yes |
| **The evaluation rows** | ❌ **no** — three different bases (see above) |
| **The training frame** | ❌ **no** — `train.csv` shrank by 20% for two benchmark tasks at `517f954` |
| **The compute budget** | ❌ **no** — the three methods expose different knobs; see below |
| Relational aux tables | ❌ by design — only the extension consumes `aux_*.csv`; this is the capability under test |

**The three methods do not accept the same kind of budget**, so none was held
constant:

| Method | Run length controlled by |
|---|---|
| **AutoGluon** | a wall-clock budget passed directly (`time_limit`), plus `presets` and `num_cpus` |
| **Extension** | the CV-rollout budget (40/60/100 per *slice* — one stretch of search between two proposal calls), the number of slices, and the number of proposal steps |
| **MLE-STAR** | the LLM-call cap, plus a per-script execution timeout |

Only AutoGluon takes time as an *input*. For the extension and MLE-STAR, run
length is an *outcome* of their respective budget parameters. One hour was used as
a rough target: the extension and MLE-STAR were configured at settings that took
roughly that long, and AutoGluon was then given 3600 s to match. That is an
approximate alignment, not a controlled variable.

`figures/comparison.csv` carries a `time_budget_s` column, but the three methods fill
it with different quantities — AutoGluon's run budget, MLE-STAR's per-script
timeout, nothing for the extension. Read it as a provenance field, never as a
comparison.

> **Where the extension's time actually goes.** The search is the cheap part: each
> rollout cross-validates on a subsample (capped at 2000 rows), so adding rollouts
> costs far less than the row count suggests. Most of the wall clock is the final
> stage — fitting the ensemble members and refitting the incumbent on the full
> training set — which is fixed work that does not scale with the search budget.

---

## 1. The numbers measured on the current split

Four tasks' captured plans were replayed through the fixed pipeline
(`scripts/replay_from_run.py`) — same plan, same budget, same proposals, at zero
API cost. They are the only benchmark artifacts on today's bench. Three carry the
post-fix stamp `ensemble.selection == "oof_3fold"`; bike-sharing's ran with a
single ensemble member, so it has no `ensemble` block to stamp and is identified as
clean by its post-fix date instead.

| Task | Metric | Clean score | Artifact |
|---|---|---|---|
| credit-fraud | roc_auc | **0.8070** | `results/replay_credit-fraud_np2_20260715-1325/` |
| traffic-violations | accuracy | **0.8876** | `results/replay_traffic-violations_np2_20260715-1401/` |
| bike-sharing | neg RMSE | **−35.26** | `results/replay_bike-sharing_np2_20260715-1358/` |
| country-happiness | neg RMSE | **−568.70** | `results/replay_country-happiness_np2_20260715-1321/` |

### How much did the fix move the score?

Comparing like with like — the same replay configuration (`np2`), before and after
the fix:

| Task | pre-fix `np2` | clean `np2` | Δ |
|---|---|---|---|
| credit-fraud | 0.8236 | 0.8070 | **−0.0166** |
| traffic-violations | 0.8843 | 0.8876 | +0.0033 |
| country-happiness | −753.87 | −568.70 | +185.2 |
| bike-sharing | *(no pre-fix `np2` archived)* | −35.26 | vs live −38.00: +2.74 |

Only credit-fraud moved in the predicted direction; the other three moved *better*
when clean.

That comparison is confounded. The fix also changed how the final model is fit:
post-fix, `evaluate_top_k` refits on **all** of `train.csv`; pre-fix it fit on the
75% left after carving the holdout. Each clean replay therefore trained on more
rows than the run it is compared against:

| Task | archived fit rows | clean fit rows | change |
|---|---|---|---|
| bike-sharing | 10 427 | 13 903 | **+33%** |
| country-happiness | 88 | 117 | **+33%** |
| traffic-violations | 9 001 | 12 001 | **+33%** |
| credit-fraud | 11 250 | 12 000 | **+6.7%** |

The three probes that improved are exactly the three that received a 33% larger
training set. credit-fraud — the only near-matched comparison, because its
`train.csv` was simultaneously shrunk — is the only one that shows optimism.

**The replays do not establish that the bias is small — only that it is not large
enough to survive a confounded comparison.** On the one probe where training-set
size is roughly controlled, the archived number was optimistic by ~0.017 roc_auc.
Elsewhere it is unresolved, and country-happiness (117 rows total) is
variance-dominated either way.

**Why a replay is a controlled comparison.** The LLM's plan is the only
non-deterministic input; everything downstream of it is seeded — subsamples,
estimators, and an exact record of every configuration already scored — so
`scripts/replay_from_run.py` reproduces a run bit-for-bit from the plan stored in
its `result.json`. That is what makes the pre-fix/clean pairs above differ in the
split and nothing else. It also means reproducing a *live* result requires
reproducing its plan: quote a live number together with its stored plan, and expect
fresh live runs to vary.

---

## 2. The archived comparison

Every number below predates the split fix. **The three columns do not share a
scoring basis**, so no winner is marked — a maximum across them would be
meaningless. Lower-is-better metrics are negated so higher is always better.

Basis: **E** = extension's own 25% carve-out, *selected on the rows it reports* →
optimistic. **A** = AutoGluon's own 25% carve-out, unseen during fitting → clean
selection, wrong bench. **M** = MLE-STAR's self-carved validation split →
self-reported, different rows again.

| Task | Metric | Extension (E) | AutoGluon (A) | MLE-STAR (M) | Note |
|---|---|---|---|---|---|
| bike-sharing | neg RMSE | −38.00 | −124.96 | −46.59 | clean replay: −35.26 |
| country-happiness ⛓ | neg RMSE | −753.87 | **cannot run** ‡ | −734.26 | clean replay: −568.70 |
| credit-fraud ⛓ | roc_auc | 0.826 ⚠ | 0.602 ⚠ | 0.500 | ⚠ **void** — see below |
| flight-delays ⛓ | neg RMSE | −36.31 | −35.24 | −35.08 | |
| medical-charge | neg RMSE | −2003.1 | −1694.9 | −2755.7 | |
| movielens ⛓ | neg RMSE | −1.006 | −0.954 | −0.943 | |
| open-payments | accuracy | 0.941 ⚠ | 0.941 ⚠ | 0.955 | ⚠ **void** — see below |
| toxicity *(text)* | accuracy | 0.945 † | 0.690 | 0.681 | 200-row holdout; † better of two runs |
| traffic-violations | accuracy | 0.885 † | 0.883 | 0.894 | clean replay: 0.8876 |
| videogame-sales | neg RMSE | −1.278 | −1.163 | **no runnable script** § | |

⛓ = relational (ships `aux_*.csv` the extension can aggregate-join; the flat-table
baselines never see them).

**⚠ credit-fraud and open-payments are void, not merely biased.** Commit `517f954`
could not recover a labelled holdout for these two, so it carved a fresh one out of
`train.csv` — which **shrank the training frame**:

| Task | `train.csv` before | after |
|---|---|---|
| credit-fraud | 15 000 | **12 000** |
| open-payments | 6 400 | **5 120** |

Both the extension *and* AutoGluon numbers for those tasks were measured on a
dataset that no longer exists in this repository. The commit itself states:
*"results previously measured on them are void."* This includes the
extension-vs-AutoGluon credit-fraud gap (0.826 vs 0.602), which is the largest
apparent margin in the table and should not be read as a result. The clean replay
(0.807) *is* on today's bench — but **there is no clean AutoGluon number to compare
it to.**

The other eight tasks re-staged byte-identically, so their archived numbers are
biased-but-mutually-comparable.

**† Both toxicity and traffic-violations are reported from the better of two live
runs.** toxicity's other run scored **0.845**; traffic-violations' other run, at a
different budget, scored **0.837**. Both pairs are listed in
[`EXPERIMENTS.md § Provenance`](EXPERIMENTS.md#provenance-of-the-archived-runs).
The spread is authored-plan variance, not search noise — toxicity's two runs used
the same budget, and the 0.10 gap turns on whether the plan offered `TextEncoder`
at all. This is the largest single source of run-to-run variance in the extension.

**toxicity's holdout is 200 rows**, so every reported accuracy is a multiple of
0.005 and a one-row change moves the score by that much. Its archived numbers were
never replayed on the fixed split — not because anything blocks it, but because the
clean-replay probe covered four other tasks. A toxicity replay is available at zero
API cost and would settle its number.

**‡ AutoGluon cannot run country-happiness at all.** It exits with "No models were
trained". The main table is `(Country, happiness_score)` — a string identifier and
the target, 117 rows. All predictive signal lives in the aggregate joins over three
World-Bank aux tables. This is an *absence* in `comparison.csv`, not a value:
`make_figures.py` drops rows whose `holdout` is null.

**§ MLE-STAR produced no runnable script for videogame-sales** before hitting its
call cap. Also an absence, not a value. It is reported rather than omitted because
a bounded-budget method that returns nothing executable *is* a result about the
method.

![Quality at cost — one panel per task](figures/quality_at_cost.png)

> **Reading this figure.** Bars are keyed by **(method, bench)**, never averaged
> across the split fix: the plain `extension` bar is pre-fix, the hatched
> `extension (clean)` bar — present on the four replayed tasks — is on today's
> bench. Each bar is the **mean of the runs in its group** with min–max error bars,
> so a bar marked `n>1` sits below the single value tabled above (which quotes one
> run). The baselines have one run each, so their bars match the table exactly.

---

## 3. Capability findings

Statements about what an approach can and cannot do. No scoring basis,
training-frame change, or selection bias affects them.

1. **AutoGluon cannot run a task whose signal is entirely relational.** On
   country-happiness it produces no model at all. The extension's `AggJoiner` stage
   constructs the aux features and fits them. This is not a lost race — it is a
   capability the flat-table baseline does not have.

2. **The extension cannot emit a non-runnable pipeline.** Operators are named by
   dotted import path and resolved through an import allow-list; nothing is
   `eval`'d, hallucinated paths are dropped, and every node in the search tree is a
   complete, evaluable configuration. MLE-STAR, which generates code, returned no
   runnable script on videogame-sales within its budget — a failure mode the
   structured-plan design cannot produce.

3. **The mechanism comparison** ([`figures/mechanism_table.md`](figures/mechanism_table.md))
   is qualitative and holds — with one correction: its "1000+ calls" figure for
   MLE-STAR is an estimate, not a measurement (see §4).

Two further extension wins have an identifiable mechanism, though their *scores*
carry the caveats of §2:

- **toxicity** — the plan offered skrub's `TextEncoder` (a sentence-embedding
  backbone) as a scoped per-column option and the search selected it
  (`scope_text_encoder: TextEncoder` + `LinearSVC` in the winning state).
  AutoGluon's default pipeline never reaches an embedding encoder for a free-text
  column. The mechanism is visible in the winning configuration; the 0.945 score
  carries the §2 caveats and has not been re-measured on the fixed split.
- **bike-sharing** — the winning state carries `DatetimeEncoder` with
  `add_weekday=True`, in both the vectorizer's datetime slot and a scoped
  `date_expansion` group. Weekday is large signal for hourly rental demand.
  This one *was* re-confirmed on the fixed split (−35.26).

Where the flat table already carries the signal — flight-delays, movielens,
medical-charge, videogame-sales — AutoGluon's mature bagging and stacking is ahead
in the archived set. Relational and authored-space capability raises a ceiling; it
does not guarantee a win.

---

## 4. Cost model

The archived data supports the *structural* argument, not an *empirical* one.

**What the architecture guarantees:**

| Method | LLM calls per task | Scales with search budget? |
|---|---|---|
| Extension | `2 + N_PROPOSES` — **known before the run** | **No.** A budget-20 run and a budget-200 run cost identical tokens. |
| AutoGluon | 0 | n/a |
| MLE-STAR | one per code-generation and per debug retry — **data-dependent** | Yes, and unbounded without a cap. |

That asymmetry is the point of keeping the LLM out of the search loop: quality
is bought with CPU time, at a token cost fixed in advance.

**What the archived data shows:**

| Method | Tokens per task (archive) |
|---|---|
| Extension | 32 k – 65 k (`llm_calls: 4`) |
| MLE-STAR | 15.9 k – 69.9 k |

**These ranges overlap**, so the archived runs do not demonstrate a token-cost gap.
The reason is that MLE-STAR's archived runs were themselves capped — a capped
cascade lands in the same range as a fixed one. What the data shows is that **the
bound holds**, not that the unbounded method is more expensive in practice.

Two limits on the cost data itself:

- MLE-STAR's `llm_calls: 0` and `wall_clock_s: 0.0` in every `result.json` are
  **hardcoded by the ingestion script** (`convert_mlestar_final_state.py`), not
  measured. There is no call count or wall clock for that method.
- The extension's `llm_calls` is a *computed constant* (`2 + n_proposes`), not a
  counter either — though for that method the constant is the guarantee.

Demonstrating the cost gap empirically requires running MLE-STAR uncapped, or at
several caps, and measuring calls. That was not done.

The one axis where the extension's own token cost *does* vary is the number of
proposal steps — measured in §5.

---

## 5. Mid-search plan injection (Optional Feature 3)

The controlled measurement is the replay series: one task's captured plan re-run at
N = 0, 1, 2 of its *real logged* proposals, everything else identical. All rows are
pre-fix except where noted.

| Task | holdout n | N=0 | N=1 | N=2 |
|---|---|---|---|---|
| credit-fraud (roc_auc) | 3000 | 0.809 | 0.828 | 0.824 · **0.807** (clean) |
| toxicity (accuracy) | **200** | 0.940 | 0.950 | 0.950 |
| traffic-violations (accuracy) | 2999 | 0.885 | 0.883 | 0.884 · **0.888** (clean) |
| country-happiness (neg RMSE) | 29 | −753.9 | −753.9 | −753.9 · **−568.7** (clean) |

![Proposal scaling](figures/proposal_scaling.png)

> **Reading this figure.** The two benches are drawn as separate series — grey
> for the pre-fix runs, blue diamonds for the current bench — because a score on
> one is not comparable to a score on the other. Every clean replay so far sits at
> `n=2`, so the blue series is a single marker per task rather than a curve;
> toxicity has no clean replay at all and shows only the grey series. The
> `n=0`/`n=1` clean replays needed to complete it cost only CPU time.

The *return* on a proposal is high-variance, and the series above are too small to
read as trends:

- **toxicity's holdout is 200 rows.** The 0.940 → 0.950 step is **two rows**. It is
  not evidence of anything.
- **country-happiness's holdout is 29 rows**, and the flat pre-fix series simply
  means no injected operator ever displaced the incumbent.
- **credit-fraud** is the only series with a holdout large enough to carry a signal,
  and it improves at N=1 then regresses at N=2.

### What a proposal step costs

Each proposal is one extra LLM call, so the call *count* stays O(1). The token
cost is not small, and this is the axis on which the extension's spend actually
varies.

![Token cost](figures/token_cost.png)

From `tokens_by_agent` across the 13 archived live runs on benchmark tasks:

| | Tokens |
|---|---|
| Initial plan (`data_analyst` + `plan_author`) | median **15.8 k** |
| **Each proposal call** | median **11.8 k** (range 8.0 k – 15.8 k) |

**A proposal call costs a median 0.79× the entire initial plan authoring**, and on
three runs — bike-sharing, movielens and one of the two toxicity runs — a single
proposal cost *more* than authoring the plan from scratch (1.28–1.35×). At
`n_proposes=2`, the setting every benchmark run used, proposals account for a
median **61% of the run's total token spend**.

The reason is structural. A proposal re-authors the *whole* plan: the current plan is
sent to the LLM and an extended plan comes back, which is merged by adding entries
only. Merging additively is what lets the search tree built so far stay valid — every
configuration already explored remains applicable to the extended plan — but it means
the existing plan is re-sent and re-emitted on every call, and the plan is the bulk of
the payload. Cost per proposal therefore grows with the size of the plan rather than
with the size of the addition.

### The trade-off

Set the cost against the score series above: proposals buy a **minor and
inconsistent** improvement at a **large and predictable** token cost. Meanwhile CV
rollouts cost **zero** tokens, which is the point of keeping the LLM out of the
search loop.

That asymmetry suggests spending the budget differently: **a larger search over the
initial plan is likely more token-efficient than buying additional proposal steps**,
because the extra rollouts are free in tokens while each proposal is not. The
archived runs cannot settle this — no task was run at a high rollout budget with
`n_proposes=0` for comparison — but it is the cheap experiment to run next, and it
costs no API quota at all.

**Monotonic improvement should not be expected** from more proposals, and this data
neither establishes nor refutes a positive average return on them.

---

## 6. What a clean comparison would require

1. **Replay the extension** from captured plans on the fixed split — zero API cost,
   minutes per task. Four of ten are already done; the remaining six, toxicity
   included, need only CPU time.
2. **Re-run AutoGluon** on the on-disk split for all 10 tasks — CPU only, no API
   cost, ~10 CPU-hours.
3. **Run MLE-STAR through `scripts/run_mlestar.py`**, which *does* score against
   the shared `test.csv`/`test_answer.csv` and *does* record calls and wall clock.
   This is the only step that costs real API budget. It has never been run
   end-to-end; all shipped MLE-STAR numbers were ingested from a teammate's earlier
   runs.
4. **Report each method's own budget parameters** rather than claiming a matched time
   budget. A shared `TIME_BUDGET` is not achievable in the strict sense: only
   AutoGluon consumes wall clock directly, the extension's `--time-budget-s` caps
   the search but not the ensemble fit or full-data refit, and MLE-STAR is bounded
   by call count.

---

## Summary

- **Every archived benchmark number predates the on-disk split fix, and all three
  methods are optimistic** — the extension by selecting on its eval rows, AutoGluon
  and MLE-STAR by training on a larger frame than today's bench, MLE-STAR
  additionally by reporting a self-carved validation score rather than a holdout
  score. There is no clean three-way comparison in this repository.
- **The only numbers on the current bench are four zero-quota replays** (§1). On
  the one probe where training-set size is roughly controlled, the archived score
  was optimistic by ~0.017 roc_auc. Elsewhere the comparison is confounded by a 33%
  larger training set, leaving the bias unbounded.
- **credit-fraud and open-payments are void, not biased** — their training frame
  shrank by 20% at the fix, so both the extension and AutoGluon numbers were
  measured on a dataset that no longer exists.
- **The archived table quotes the better of repeated runs** on toxicity (0.945 vs
  0.845) and traffic-violations (0.885 vs 0.837) — a second selection effect on top
  of the split bias. toxicity's holdout is 200 rows, so differences under ~0.01
  there are not meaningful.
- **The capability findings survive everything** (§3): AutoGluon cannot run a task
  whose signal is purely relational, and the extension cannot emit a non-runnable
  pipeline — a failure MLE-STAR produced on one task.
- **The cost model's structural argument holds; its empirical demonstration does
  not.** The extension's token cost is constant in search depth by construction —
  but the archived MLE-STAR runs were capped, so the two methods' measured token
  ranges overlap and no gap was demonstrated (§4).
- **Proposal steps are the expensive part of the extension's token bill, and buy
  little** (§5). Each one re-emits the whole plan, costing a median 11.8 k tokens —
  0.79× the entire initial authoring — for a minor, inconsistent score change. At
  `n_proposes=2` that is 61% of the run's tokens. Since CV rollouts cost no tokens
  at all, **a larger search on the initial plan is the more token-efficient use of
  the budget**, though the archived runs cannot confirm it.
