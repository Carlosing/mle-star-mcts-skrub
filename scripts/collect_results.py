"""Mirror runs/'s directory structure into results/, copying every run artifact.

``runs/`` holds full run artifacts; ``results/`` is the small, git-shareable
mirror. Each run directory (identified by holding a ``result.json``) is copied
whole — ``result.json``, ``summary.md``, ``ensemble.pkl`` and the raw agent I/O
(``data_analyst_*.json``, ``plan_author_*.json``, ``proposer_*.json``) — so a
run's captured plan and proposals travel with its scores. That is what makes an
archived run replayable: ``scripts/replay_from_run.py`` reads ``spec_raw`` from
``result.json`` and the real proposals from ``proposer_<k>_response.json``.

The exception is AutoGluon's ``ag_models/`` (multi-GB serialized models per
task, written by ``run_autogluon.py``). Those directories are pruned from the
walk entirely — never read, never copied. ``PRUNE_DIRS`` holds that list.

A per-file size ceiling (``--max-file-mb``, default 25) is a backstop against
committing something pathological by accident; skipped files are reported, not
silently dropped. Pass ``--max-file-mb 0`` to disable it.

Safe to re-run: only copies a file that's missing or newer than what's already
in ``results/``; never deletes anything.

Run:
    uv run python scripts/collect_results.py
    uv run python scripts/collect_results.py --runs runs --out results
"""

import argparse
import os
import shutil

# Directories never descended into: AutoGluon's serialized model store (GBs per
# task), rendered figures (output, not a run artifact — moved separately), and
# the usual caches.
PRUNE_DIRS = frozenset(
    {"ag_models", "figures", "__pycache__", ".ipynb_checkpoints"}
)

# Filenames never copied.
SKIP_FILES = frozenset({".DS_Store", "Thumbs.db"})

# Per-file ceiling in MB; a file above it is skipped and reported.
DEFAULT_MAX_FILE_MB = 25.0


def collect_results(
    runs_dir: str,
    out_dir: str,
    max_file_mb: float = DEFAULT_MAX_FILE_MB,
) -> tuple[list[str], list[tuple[str, float]]]:
    """Mirror every run directory under ``runs_dir`` into ``out_dir``.

    A "run directory" is any directory containing ``result.json``; all of its
    files are copied, `PRUNE_DIRS` are never entered, and files larger than
    ``max_file_mb`` (0 = no limit) are skipped. Returns
    ``(copied_paths, [(skipped_path, size_mb), ...])``.

    Example:
        collect_results("runs", "results") -> (["results/toxicity_.../result.json", ...], [])
    """
    copied: list[str] = []
    skipped: list[tuple[str, float]] = []
    limit_bytes = max_file_mb * 1024 * 1024 if max_file_mb > 0 else None

    for root, dirs, files in os.walk(runs_dir):
        dirs[:] = [d for d in dirs if d not in PRUNE_DIRS]
        if "result.json" not in files:
            continue

        rel = os.path.relpath(root, runs_dir)
        dest_dir = os.path.join(out_dir, rel) if rel != "." else out_dir

        for name in sorted(files):
            if name in SKIP_FILES:
                continue
            src = os.path.join(root, name)
            if limit_bytes is not None:
                size = os.path.getsize(src)
                if size > limit_bytes:
                    skipped.append((src, size / (1024 * 1024)))
                    continue
            dst = os.path.join(dest_dir, name)
            if os.path.exists(dst) and os.path.getmtime(
                dst
            ) >= os.path.getmtime(src):
                continue
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(dst)

    return copied, skipped


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Mirror run artifacts (folder structure preserved) from "
        "runs/ to results/, excluding AutoGluon's ag_models/."
    )
    parser.add_argument("--runs", default="runs", help="source runs directory")
    parser.add_argument(
        "--out", default="results", help="destination directory"
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=DEFAULT_MAX_FILE_MB,
        help=f"skip files larger than this many MB (default: "
        f"{DEFAULT_MAX_FILE_MB:g}; 0 disables the limit)",
    )
    args = parser.parse_args()

    copied, skipped = collect_results(
        args.runs, args.out, max_file_mb=args.max_file_mb
    )
    print(f"copied {len(copied)} file(s) into {args.out}/")
    for path in copied:
        print(f"  {path}")
    if skipped:
        print(
            f"\nskipped {len(skipped)} file(s) over "
            f"{args.max_file_mb:g} MB (raise --max-file-mb to include):"
        )
        for path, size_mb in skipped:
            print(f"  {path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    _main()
