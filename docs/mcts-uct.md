# MCTS & UCT — team primer

A short guide to the Monte Carlo Tree Search engine we use for pipeline
configuration search.
Read this as a guideline for
[`machine_learning_engineering/mcts.py`](../machine_learning_engineering/mcts.py).

## What problem MCTS solves for us

A skrub pipeline exposes several `choose_from` / `choose_int` / `choose_float`
nodes — encoder, model, number of trees, learning rate, … Each combination of
choices is one **configuration**. We can't try them all (the space is large and
each evaluation is a cross-validation that costs seconds). MCTS spends a fixed
**budget** of evaluations searching that space, steering toward promising
configurations instead of trying them at random.

The compact configuration dict (`{"encoder": "GapEncoder()", "model": "GBM",
"n_trees": 100, ...}`) is the **MCTS state** — never the full Python script.

## UCT in one formula

Selection at every node uses the **Upper Confidence bound applied to Trees**
([`mcts.py`](../machine_learning_engineering/mcts.py) `MCTSNode.uct`):

```text
UCT(node) = Q/N  +  c · √(ln N_parent / N)
            └───┘    └──────────────────┘
        exploitation       exploration
```

- **Q/N** — the node's average reward so far (its mean CV score). High means
  "this region of configs has been good."
- **exploration term** — large when a node has been visited rarely relative to
  its parent (`N` small, `N_parent` large), and shrinks as we sample it more.
  High means "we haven't looked here much; worth a try."
- **c** — the knob trading the two off. We use **0.5**, not the textbook
  √2 ≈ 1.41, because our rewards are *noisy subsampled* CV scores, so we lean
  toward exploiting estimates we're more confident in.
- **Unvisited nodes return ∞**, so every option is tried at least once before
  any is re-explored.

## The four phases (one MCTS iteration)

The loop in `mcts_search` repeats `budget` times:

1. **Select** — from the root, repeatedly descend into the child with the
   highest UCT until reaching a leaf (`select`).
2. **Expand** — grow new children from that leaf by swapping **one** choice
   value at a time. The action space comes from the skrub plan's choice nodes
   (`skrub_ops.get_action_space`, built on `_evaluation.choices`), **never from
   an LLM**. Already-seen states are skipped via `tried_states` (`expand`).
3. **Rollout** — evaluate the chosen configuration: cross-validate it on a
   seeded subsample, producing a reward in `[0, 1]`
   (`skrub_ops.make_rollout_fn`). Failed configs score `0.0` rather than
   crashing.
4. **Backpropagate** — add the reward to the `Q` and `N` of every ancestor up
   to the root (`backpropagate`).

**Why a tree and not a flat search:** backpropagation makes a parent's `Q/N`
the running average of its entire subtree, so UCT naturally steers the budget
toward promising *regions* of the config space, not just individual points.

## Things specific to our implementation

- **The tree persists across outer-loop steps.** Pass the returned `root` and
  `tried_states` back into the next `mcts_search` call so statistics
  accumulate — never reset the tree (brief anti-pattern #7).
- **Determinism is mandatory.** Rollouts must be reproducible (seeded
  subsample + seeded estimators) or `Q/N` becomes noise and UCT never
  converges. See the `skrub-api-corrections` note for why skrub's own
  `subsample(how="random")` is *not* safe here.
- **The LLM never does the search.** UCT selection, action enumeration,
  rollout, and backprop are all pure code. The optional `prior_fn` hook is the
  only place an LLM can influence the tree (warm-starting child `Q`/`N`),
  following the AlphaZero "policy prior + UCT search" pattern (brief §6).
- **Targeting narrows what `expand` grows, not where selection starts.**
  `mcts_search` accepts `target_key` — a single choice name, or a *set* of
  them — so `expand` only edits those stages while UCT selection and backprop
  are untouched. (The outer loop no longer uses it: since 2026-07-13 the
  Option-1 pick is only a proposer hint — the variance ledger elects `model`
  on essentially every pick, and locking expansion to it starved off-target
  injected options until the bonus phase.) The **focused-refinement bonus
  phase** instead
  passes `start_node` so selection *descends from the incumbent node* rather
  than the root, spending a final `ceil(budget/4)` rollouts locally on ALL of
  the incumbent's single-edit neighbors — structural stages and gated HPs
  alike (backprop still walks the full path to the root, so global `Q/N`
  stays consistent; gating still keeps a non-selected model's HPs out). This
  is the deterministic form of the "biased exploration around the best
  config" the stage-targeting was originally conceived for.

## Optimization-MCTS, not game-MCTS

This is the most important conceptual point for understanding why our engine
looks the way it does. Classic MCTS (chess, Go) has two properties our problem
does **not**:

- **Only terminal states have a value.** Interior board positions are not
  scorable; their value is *estimated* by a random rollout/playout to the end
  of the game.
- **The objective is cumulative regret** — every move played counts toward the
  final outcome, so you want to minimize average regret over the search.

In pipeline-configuration search, **every node is a complete, cross-validatable
configuration with a directly available value**. Three consequences:

1. **Our "rollout" is a direct value-function query, not a playout.**
   `skrub_ops.make_rollout_fn` just cross-validates the config. This is exactly
   what AlphaZero does when it replaces random playouts with a value network
   `v(s)` — we are in the "value at every node, no playout" regime. The
   optional `prior_fn` is the policy-prior half of the same pattern.

2. **The real objective is simple regret (best-arm identification), not
   cumulative regret.** We don't care how good the *average* config we tried
   was — we want the single best one. This is *why* the brief drops `c` from
   √2 to 0.5 (lean toward exploitation); the principled version would use a
   pure-exploration objective (see references below).

3. **We return the global incumbent, not the most-visited root child.**
   `mcts_search` tracks `best_state`/`best_score` independently of tree
   position, so a good config is never lost because UCT wandered elsewhere —
   UCT only decides *where to spend the next evaluation*. (Game-MCTS returns
   the most-visited child; doing that here would be a bug.)

### Two possible tree topologies

There is a design fork worth being explicit about:

- **Partial-config-per-level** (what the brief's "max depth = one per
  `choose_from` node" implies, and what SELA does): the root is empty, each
  level commits one more choice, only leaves are fully specified, and interior
  nodes need a rollout. Depth = number of parameters.
- **Edit-distance / local search** (what our `expand` actually builds): every
  node is a *complete* config — the root is the plan's default config
  (`skrub_ops.get_default_state`) and `expand` changes one key — edges are
  single-parameter edits, and depth = number of edits from the seed. Every
  node is evaluable.

Our code currently uses the second. "Descending the subtree" therefore means
"making more single-parameter edits to the config," and there are no terminal
states in the game sense.

### Loops and transpositions

Because edges are single-parameter edits, the same configuration can be reached
by different edit orders (change encoder then model = change model then
encoder). In a naive tree this produces **transpositions** (the same state as
multiple nodes) and, with reversible edits, **cycles** (edit a value, then edit
it back).

How this is handled in general:

- **Games tolerate transpositions** by treating them as separate nodes (a true
  tree — correct but wasteful), or share statistics across them with a
  **transposition table / DAG-MCTS** (Childs et al. 2008; Saffidine et al.
  2012, "UCD").
- **True cycles** are usually ruled out by game rules (threefold repetition in
  chess, the ko rule in Go) or by detecting a repeated state on the current
  path and refusing to descend into it.

How **we** handle it: the global `tried_states` set is a graph-search *closed
set*. `expand` skips any candidate whose `state_key` is already present, so
**each distinct configuration is instantiated as a node at most once in the
whole tree**. That single mechanism prevents both cycles (a config can't be
re-created by editing a value back) and duplicate transposition nodes.

The trade-off versus a transposition table: we do **not** merge statistics from
the multiple paths that could reach a config — the first parent to propose it
wins and fixes its position in the tree. For optimization that is fine (we want
each config evaluated once, not its `Q/N` averaged over arrival routes), and it
keeps the engine pure. The only residual "revisit" is the exhausted-leaf edge
case, where a leaf whose single-edit neighbours are all already tried is
re-selected and its state re-evaluated; the persisted **score cache**
(`mcts_search`'s `score_cache`) makes that a free lookup rather than a repeated
rollout, so those iterations cost nothing but a UCT bookkeeping update. (The
focused-refinement bonus phase leans on exactly this: once the incumbent's
untested single-edit neighbors are exhausted, remaining bonus rollouts are
cache hits.)

### Extensions worth knowing (optimization-specific)

- **Optimistic optimization** — Munos's HOO / DOO / SOO and the 2014 monograph
  below: tree search where *any* point is evaluable. The closest-fitting theory
  for our setting.
- **Best-arm identification** — Successive Rejects, Sequential Halving,
  simple-regret MCTS (BRUE): allocate a fixed budget to *find the max* rather
  than minimize average regret.
- **Progressive widening** (Couëtoux et al. 2011) — expand a node's children
  gradually as its visit count grows, so promising nodes earn more children and
  weak ones stay narrow. The principled alternative to hard-terminating
  non-improving branches.
- **Restart-enabled MCTS** — the brief's **ArchPilot**: when a region plateaus,
  restart from the incumbent instead of marking nodes terminal (escapes local
  optima that single-parameter edits cannot cross).
- **Bayesian optimization** (SMAC / TPE / GP-BO) — the classical non-MCTS tool
  for expensive black-box functions; worth citing as the baseline our approach
  competes against.

## Resources (in reading order)

Read #1 alone and you can follow our code; #2–#3 give the theory for the
report; #4 is a lookup reference; #5 ties it to our project.

1. **Int8 — "Monte Carlo Tree Search: beginners guide"**
   (https://int8.io/monte-carlo-tree-search-beginners-guide/). Visual walk
   through the four phases and the UCT formula. **Start here.**
2. **Auer, Cesa-Bianchi & Fischer (2002), "Finite-time Analysis of the
   Multiarmed Bandit Problem."** The origin of **UCB1**, the
   `mean + c·√(ln N_parent / N)` formula. Read §2 (the UCB1 statement). UCT is
   "apply UCB1 at every tree node."
3. **Kocsis & Szepesvári (2006), "Bandit based Monte-Carlo Planning."** The
   paper that introduced **UCT** (UCB1 extended to trees). Short — cite this as
   the source.
4. **Browne et al. (2012), "A Survey of Monte Carlo Tree Search Methods,"**
   IEEE TCIAIG. The reference handbook; keep it for §3 (the algorithm) and the
   exploration-constant discussion. Not an end-to-end read.
5. **SELA (MetaGPT/FoundationAgents, 2024).** MCTS over ML pipeline configs —
   the closest prior work to this project. Read last, once the mechanics are
   clear.
6. **Munos (2014), "From Bandits to Monte-Carlo Tree Search: The Optimistic
   Principle Applied to Optimization and Planning,"** Foundations & Trends in
   ML. The reference for tree search where every point is evaluable — the
   theory behind the "Optimization-MCTS" section above. Optional / for the
   report.
