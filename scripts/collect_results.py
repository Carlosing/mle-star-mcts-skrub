"""Mirror runs/'s directory structure into results/, copying only result.json.

``runs/`` holds full run artifacts, including AutoGluon's ``ag_models/``
(multi-GB serialized models per task) — far too large to share or commit.
``results/`` mirrors each run directory's name but keeps only its
``result.json`` (the small slice ``scripts/make_figures.py`` actually reads),
so the comparison data can be pushed/shared without the multi-GB baggage.

Safe to re-run: only copies a file that's missing or newer than what's already
in ``results/``; never deletes anything.

Run:
    uv run python scripts/collect_results.py
    uv run python scripts/collect_results.py --runs runs --out results
"""

import argparse
import os
import shutil


def collect_results(runs_dir: str, out_dir: str) -> list[str]:
    """Copy every ``result.json`` under ``runs_dir`` into the same relative
    path under ``out_dir``. Skips any ``figures`` directory (rendered output,
    not a run artifact — moved separately). Returns the paths copied.
    """
    copied = []
    for root, dirs, files in os.walk(runs_dir):
        if os.path.basename(root) == "figures":
            dirs[:] = []
            continue
        if "result.json" not in files:
            continue
        rel = os.path.relpath(root, runs_dir)
        dest_dir = os.path.join(out_dir, rel) if rel != "." else out_dir
        src = os.path.join(root, "result.json")
        dst = os.path.join(dest_dir, "result.json")
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(
            src
        ):
            continue
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy result.json artifacts (folder structure preserved) "
        "from runs/ to results/."
    )
    parser.add_argument("--runs", default="runs", help="source runs directory")
    parser.add_argument(
        "--out", default="results", help="destination directory"
    )
    args = parser.parse_args()

    copied = collect_results(args.runs, args.out)
    print(f"copied {len(copied)} result.json file(s) into {args.out}/")
    for path in copied:
        print(f"  {path}")


if __name__ == "__main__":
    _main()
