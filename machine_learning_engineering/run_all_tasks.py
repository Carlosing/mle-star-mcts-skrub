"""Batch runner that executes the MLE-STAR pipeline for every task folder.

Each task runs in its own child process so that the parent can enforce a hard
kill switch (Ctrl+C, token budget or per-task timeout) without blocking the rest
of the batch. Progress is persisted after every task in a registry file so the
batch can be resumed if it is interrupted.
"""

import argparse
import datetime
import fcntl
import json
import multiprocessing
import os
import signal
import sys
import time
import traceback
from typing import Any

import openai

from machine_learning_engineering.agent import run_pipeline
import machine_learning_engineering.runner as runner


DEFAULT_DATA_DIR = "./machine_learning_engineering/tasks/"
DEFAULT_WORKSPACE_DIR = "./machine_learning_engineering/workspace/"
DEFAULT_MAX_TOKENS_PER_TASK = 200_000
DEFAULT_MAX_TOTAL_TOKENS = 0  # 0 means unlimited.
DEFAULT_MAX_TASK_TIME = 3600  # seconds.
DEFAULT_MAX_ATTEMPTS = 1
DEFAULT_MAX_LLM_RETRIES = 3
DEFAULT_LLM_RETRY_DELAY = 5.0  # seconds.
DEFAULT_MAX_CALLS_PER_MINUTE = 9  # Keep under Academic Cloud's 10/min limit.
REGISTRY_FILENAME = "batch_registry.json"
REGISTRY_LOCK_FILENAME = ".batch_registry.lock"
POLL_INTERVAL = 1.0  # seconds.

# Set by the SIGINT handler in the parent process.
_shutdown_requested = multiprocessing.Event()


def _acquire_registry_lock(workspace_dir: str):
    """Acquire an exclusive lock for registry read/write operations."""
    lock_path = os.path.join(workspace_dir, REGISTRY_LOCK_FILENAME)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release_registry_lock(fd) -> None:
    """Release the registry lock."""
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


class _RateLimiter:
    """Simple in-process sliding-window rate limiter for LLM calls."""

    def __init__(self, max_calls_per_minute: int):
        self.max_calls = max_calls_per_minute
        self.window_seconds = 60.0
        self._call_times: list[float] = []

    def acquire(self) -> None:
        """Block until a call slot is available."""
        now = time.time()
        # Drop calls outside the window.
        cutoff = now - self.window_seconds
        self._call_times = [t for t in self._call_times if t > cutoff]

        if len(self._call_times) >= self.max_calls:
            # Wait until the oldest call slides out of the window.
            oldest = self._call_times[0]
            sleep_for = oldest + self.window_seconds - now
            if sleep_for > 0:
                print(
                    f"[rate limiter] Sleeping {sleep_for:.1f}s to stay under "
                    f"{self.max_calls} calls/minute..."
                )
                time.sleep(sleep_for)
            # Refresh window after sleeping.
            now = time.time()
            cutoff = now - self.window_seconds
            self._call_times = [t for t in self._call_times if t > cutoff]

        self._call_times.append(time.time())


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _load_registry_locked(workspace_dir: str) -> dict[str, Any]:
    """Read the registry without acquiring the lock (caller must hold it)."""
    path = os.path.join(workspace_dir, REGISTRY_FILENAME)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
        "total_tokens_spent": 0,
        "tasks": [],
    }


def _load_registry(workspace_dir: str) -> dict[str, Any]:
    fd = _acquire_registry_lock(workspace_dir)
    try:
        return _load_registry_locked(workspace_dir)
    finally:
        _release_registry_lock(fd)


def _save_registry(registry: dict[str, Any], workspace_dir: str) -> None:
    os.makedirs(workspace_dir, exist_ok=True)
    path = os.path.join(workspace_dir, REGISTRY_FILENAME)
    tmp_path = path + ".tmp"
    fd = _acquire_registry_lock(workspace_dir)
    try:
        # Re-read the registry under the lock to avoid losing updates from
        # other parallel processes.
        current = _load_registry_locked(workspace_dir)
        # Merge task entries: overwrite with the entries from this process.
        existing = {e["task_name"]: e for e in current.get("tasks", [])}
        for entry in registry.get("tasks", []):
            existing[entry["task_name"]] = entry
        current["tasks"] = sorted(
            existing.values(), key=lambda e: e.get("task_name", "")
        )
        # Recompute the global total from the merged entries.
        current["total_tokens_spent"] = sum(
            e.get("tokens_used", 0) or 0 for e in current["tasks"]
        )
        current["updated_at"] = _now_iso()
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        _release_registry_lock(fd)


def _find_task_entry(registry: dict[str, Any], task_name: str) -> dict[str, Any] | None:
    for entry in registry["tasks"]:
        if entry.get("task_name") == task_name:
            return entry
    return None


def _summarize_final_state(workspace_dir: str, task_name: str) -> dict[str, Any]:
    """Read the final_state.json produced by agent.py, if it exists."""
    path = os.path.join(workspace_dir, task_name, "final_state.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        submission = state.get("submission_code_exec_result", {}) or {}
        return {
            "best_score": state.get("best_score_1"),
            "submission_returncode": submission.get("returncode"),
            "tokens_used": state.get("total_tokens_spent", 0),
        }
    except Exception:
        return {}


def _task_worker(
    task_name: str,
    data_dir: str,
    workspace_dir: str,
    token_counter: multiprocessing.Value,
    result_queue: multiprocessing.Queue,
    max_llm_retries: int = DEFAULT_MAX_LLM_RETRIES,
    llm_retry_delay: float = DEFAULT_LLM_RETRY_DELAY,
    max_calls_per_minute: int = DEFAULT_MAX_CALLS_PER_MINUTE,
) -> None:
    """Run a single task inside a child process.

    Monkeypatches runner.llm_call so every LLM call updates the shared
    token_counter and retries on transient timeouts/connection errors.
    The final result (or exception) is placed in result_queue.
    """
    # Let the parent handle Ctrl+C.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    debug_log_path = os.path.join(workspace_dir, f"llm_debug_{task_name}.log")
    debug_log = open(debug_log_path, "w", encoding="utf-8")

    def _log(msg: str) -> None:
        line = f"[{datetime.datetime.now().isoformat()}] {msg}"
        print(line)
        debug_log.write(line + "\n")
        debug_log.flush()

    original_llm_call = runner.llm_call
    rate_limiter = _RateLimiter(max_calls_per_minute)
    call_index = [0]

    def _format_messages(messages) -> str:
        try:
            total_len = sum(len(m.get("content", "")) for m in messages)
            preview = json.dumps(messages, ensure_ascii=False)[:2000]
            return f"messages_count={len(messages)}, total_chars={total_len}, preview={preview}"
        except Exception as e:
            return f"could_not_format_messages: {e}"

    def _extract_response_body(exc) -> str:
        """Try to extract the raw response body from an OpenAI exception."""
        body = ""
        # Common locations for the response body in openai exceptions.
        for attr in ("response", "body"):
            obj = getattr(exc, attr, None)
            if obj is None:
                continue
            if hasattr(obj, "text"):
                body += f"\n{attr}.text: {getattr(obj, 'text', '')}"
            if hasattr(obj, "json"):
                try:
                    body += f"\n{attr}.json: {obj.json()}"
                except Exception:
                    pass
            if hasattr(obj, "status_code"):
                body += f"\n{attr}.status_code: {obj.status_code}"
        return body or "no response body extracted"

    def _monitored_llm_call(messages, temperature=0.0, model=None):
        call_index[0] += 1
        call_id = call_index[0]
        _log(
            f"CALL {call_id} START | model={model} | temperature={temperature} | "
            f"{_format_messages(messages)}"
        )
        last_error = None
        for attempt in range(max_llm_retries):
            try:
                rate_limiter.acquire()
                _log(f"CALL {call_id} ATTEMPT {attempt + 1}/{max_llm_retries} | calling LLM")
                response = original_llm_call(
                    messages, temperature=temperature, model=model
                )
                usage = getattr(response, "usage", None)
                tokens = getattr(usage, "total_tokens", 0) if usage else 0
                with token_counter.get_lock():
                    token_counter.value += int(tokens)
                content_preview = ""
                try:
                    content_preview = response.choices[0].message.content[:500]
                except Exception:
                    pass
                _log(
                    f"CALL {call_id} SUCCESS | tokens={tokens} | "
                    f"content_preview={content_preview!r}"
                )
                return response
            except (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.RateLimitError,
                openai.APIStatusError,
            ) as exc:
                status_code = getattr(exc, "status_code", None)
                _log(
                    f"CALL {call_id} ATTEMPT {attempt + 1}/{max_llm_retries} FAILED | "
                    f"type={type(exc).__name__} | status_code={status_code} | "
                    f"error={exc} | response_body={_extract_response_body(exc)}"
                )
                # Retry only on transient server errors (5xx) or rate limits (429).
                if status_code is not None and not (
                    status_code >= 500 or status_code == 429
                ):
                    raise
                last_error = exc
                if attempt < max_llm_retries - 1:
                    delay = llm_retry_delay * (2 ** attempt)
                    _log(f"CALL {call_id} RETRYING in {delay:.1f}s...")
                    time.sleep(delay)
        _log(f"CALL {call_id} GIVING UP after {max_llm_retries} attempts")
        raise last_error

    runner.llm_call = _monitored_llm_call

    try:
        # Force absolute paths into the global config so that agent.py's
        # save_state writes to the correct location regardless of the child
        # process's current working directory.
        import machine_learning_engineering.shared_libraries.config as config_module

        config_module.CONFIG.task_name = task_name
        config_module.CONFIG.workspace_dir = workspace_dir
        config_module.CONFIG.data_dir = data_dir

        state = run_pipeline(
            task_name=task_name,
            data_dir=data_dir,
            workspace_dir=workspace_dir,
        )
        submission = state.get("submission_code_exec_result", {}) or {}
        result_queue.put(
            {
                "status": "success",
                "best_score": state.get("best_score_1"),
                "submission_returncode": submission.get("returncode"),
                "tokens_used": state.get("total_tokens_spent", 0),
            }
        )
    except Exception as exc:
        traceback.print_exc()
        _log(f"WORKER EXCEPTION: {exc}")
        result_queue.put(
            {
                "status": "failed",
                "error": str(exc),
                "tokens_used": token_counter.value,
            }
        )
    finally:
        debug_log.close()


def _terminate_process(process: multiprocessing.Process) -> None:
    """Kill a child process and wait for it to exit."""
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def _run_single_task(
    task_name: str,
    data_dir: str,
    workspace_dir: str,
    max_tokens_per_task: int,
    max_total_tokens: int,
    max_task_time: int,
    total_tokens_so_far: int,
    max_llm_retries: int = DEFAULT_MAX_LLM_RETRIES,
    llm_retry_delay: float = DEFAULT_LLM_RETRY_DELAY,
    max_calls_per_minute: int = DEFAULT_MAX_CALLS_PER_MINUTE,
) -> dict[str, Any]:
    """Run one task under the parent's supervision.

    Returns a result dict with status, score, returncode, tokens_used and
    optional error. The parent can enforce timeout and token budgets by
    terminating the child process.
    """
    token_counter = multiprocessing.Value("L", 0)
    result_queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=_task_worker,
        args=(
            task_name,
            data_dir,
            workspace_dir,
            token_counter,
            result_queue,
            max_llm_retries,
            llm_retry_delay,
            max_calls_per_minute,
        ),
    )
    process.start()

    start_time = time.time()
    terminated = False
    reason = ""

    try:
        while process.is_alive():
            time.sleep(POLL_INTERVAL)

            elapsed = time.time() - start_time
            current_tokens = token_counter.value
            projected_total = total_tokens_so_far + current_tokens

            if _shutdown_requested.is_set():
                reason = "shutdown requested"
                terminated = True
                break

            if max_task_time > 0 and elapsed > max_task_time:
                reason = f"timeout ({elapsed:.0f}s > {max_task_time}s)"
                terminated = True
                break

            if max_tokens_per_task > 0 and current_tokens > max_tokens_per_task:
                reason = f"token limit per task ({current_tokens} > {max_tokens_per_task})"
                terminated = True
                break

            if max_total_tokens > 0 and projected_total > max_total_tokens:
                reason = f"total token limit ({projected_total} > {max_total_tokens})"
                terminated = True
                break

        if terminated:
            print(f"Killswitch activated for {task_name}: {reason}. Terminating task...")
            _terminate_process(process)
    except KeyboardInterrupt:
        # Should be handled by the signal handler, but keep a fallback.
        _shutdown_requested.set()
        reason = "keyboard interrupt"
        terminated = True
        _terminate_process(process)

    # Drain the queue if the child managed to report a result.
    result: dict[str, Any] | None = None
    if not terminated:
        try:
            result = result_queue.get(timeout=5)
        except Exception:
            pass
        process.join(timeout=5)
        if process.is_alive():
            _terminate_process(process)

    # If we killed the task, report the token count from our live counter.
    # Do not read final_state.json here because it may contain stale data
    # from a previous run.
    if terminated:
        return {
            "status": "killed",
            "reason": reason,
            "best_score": None,
            "submission_returncode": None,
            "tokens_used": token_counter.value,
            "error": None,
        }

    if result is None:
        summary = _summarize_final_state(workspace_dir, task_name)
        return {
            "status": "failed",
            "reason": "worker did not return a result",
            "best_score": summary.get("best_score"),
            "submission_returncode": summary.get("submission_returncode"),
            "tokens_used": summary.get("tokens_used", 0),
            "error": "worker process exited without a result",
        }

    return result


def _signal_handler(signum, frame) -> None:
    """Set the shutdown flag on Ctrl+C; the parent will kill the current task."""
    print("\nShutdown requested. Stopping current task and continuing to next...")
    _shutdown_requested.set()


def discover_task_names(data_dir: str) -> list[str]:
    """Return the names of subdirectories that contain a task description."""
    data_dir = os.path.abspath(data_dir)
    task_names = []
    for entry in sorted(os.listdir(data_dir)):
        entry_path = os.path.join(data_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if not os.path.exists(os.path.join(entry_path, "task_description.txt")):
            print(f"Skipping '{entry}' (no task_description.txt found)", file=sys.stderr)
            continue
        task_names.append(entry)
    return task_names


def run_all_tasks(
    data_dir: str | None = None,
    workspace_dir: str | None = None,
    max_tokens_per_task: int = DEFAULT_MAX_TOKENS_PER_TASK,
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
    max_task_time: int = DEFAULT_MAX_TASK_TIME,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    requested_task_names: list[str] | None = None,
    max_llm_retries: int = DEFAULT_MAX_LLM_RETRIES,
    llm_retry_delay: float = DEFAULT_LLM_RETRY_DELAY,
    max_calls_per_minute: int = DEFAULT_MAX_CALLS_PER_MINUTE,
) -> None:
    """Run the MLE-STAR pipeline once per task folder.

    Each task runs in a separate child process. If a task exceeds its token
    budget, its time limit, or the user requests a shutdown, the child is
    terminated and the batch continues with the next task. Progress is saved
    to a registry file after every task.
    """
    data_dir = os.path.abspath(data_dir or DEFAULT_DATA_DIR)
    workspace_dir = os.path.abspath(workspace_dir or DEFAULT_WORKSPACE_DIR)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    os.makedirs(workspace_dir, exist_ok=True)
    registry = _load_registry(workspace_dir)
    total_tokens_spent = registry.get("total_tokens_spent", 0)

    all_task_names = discover_task_names(data_dir)
    if not all_task_names:
        print(f"No task folders found in {data_dir}", file=sys.stderr)
        return

    # If specific tasks were requested, filter to those.
    if requested_task_names:
        requested = set(requested_task_names)
        available = set(all_task_names)
        missing = requested - available
        if missing:
            print(
                f"Warning: requested task(s) not found: {sorted(missing)}",
                file=sys.stderr,
            )
        all_task_names = [name for name in all_task_names if name in requested]

    # Build the list of tasks to run, skipping successful ones and respecting
    # the attempt limit for failed/killed tasks.
    task_names = []
    for name in all_task_names:
        entry = _find_task_entry(registry, name)
        if entry is None:
            task_names.append(name)
            continue
        if entry.get("status") == "success":
            print(f"Skipping '{name}' (already completed).")
            continue
        attempts = entry.get("attempts", 0)
        if attempts >= max_attempts:
            print(
                f"Skipping '{name}' (reached {attempts}/{max_attempts} attempts, "
                f"last status: {entry.get('status')})."
            )
            continue
        task_names.append(name)

    print(f"Found {len(all_task_names)} task(s), {len(task_names)} remaining to run.")
    print(f"Workspace root: {workspace_dir}")
    print(
        f"Limits: {max_tokens_per_task} tokens/task, {max_task_time}s/task, "
        f"{max_llm_retries} LLM retries, {max_calls_per_minute} calls/minute"
    )
    if max_total_tokens > 0:
        print(f"Total token budget: {max_total_tokens}")
    print()

    signal.signal(signal.SIGINT, _signal_handler)

    for task_name in task_names:
        if _shutdown_requested.is_set():
            print("Shutdown flag is set; stopping batch.")
            break

        print(f"=== Running task: {task_name} ===")
        task_start = time.time()

        result = _run_single_task(
            task_name=task_name,
            data_dir=data_dir,
            workspace_dir=workspace_dir,
            max_tokens_per_task=max_tokens_per_task,
            max_total_tokens=max_total_tokens,
            max_task_time=max_task_time,
            total_tokens_so_far=total_tokens_spent,
            max_llm_retries=max_llm_retries,
            llm_retry_delay=llm_retry_delay,
            max_calls_per_minute=max_calls_per_minute,
        )

        elapsed = time.time() - task_start
        tokens_used = result.get("tokens_used") or 0
        total_tokens_spent += tokens_used

        # Update registry.
        entry = _find_task_entry(registry, task_name)
        if entry is None:
            entry = {"task_name": task_name}
            registry["tasks"].append(entry)
        entry["status"] = result["status"]
        entry["attempts"] = entry.get("attempts", 0) + 1
        entry["best_score"] = result.get("best_score")
        entry["submission_returncode"] = result.get("submission_returncode")
        entry["tokens_used"] = tokens_used
        entry["elapsed_seconds"] = round(elapsed, 1)
        entry["timestamp"] = _now_iso()
        if result.get("reason"):
            entry["reason"] = result["reason"]
        if result.get("error"):
            entry["error"] = result["error"]

        registry["total_tokens_spent"] = total_tokens_spent
        _save_registry(registry, workspace_dir)

        # Print per-task summary.
        print(
            f"--- {task_name}: {result['status']} | "
            f"score={result.get('best_score')} | "
            f"returncode={result.get('submission_returncode')} | "
            f"tokens={tokens_used} | "
            f"time={elapsed:.1f}s | "
            f"total_tokens={total_tokens_spent} ---\n"
        )

    # Final summary.
    print("\n=== Batch summary ===")
    for entry in sorted(registry["tasks"], key=lambda e: e.get("task_name", "")):
        line = (
            f"{entry['task_name']}: {entry.get('status')}"
            f" | attempts={entry.get('attempts', 0)}"
            f" | tokens={entry.get('tokens_used', 0)}"
            f" | score={entry.get('best_score')}"
            f" | returncode={entry.get('submission_returncode')}"
        )
        if entry.get("reason"):
            line += f" | reason={entry['reason']}"
        print(line)

    print(f"\nTotal tokens spent: {total_tokens_spent}")

    failed_or_killed = [
        e for e in registry["tasks"] if e.get("status") in ("failed", "killed")
    ]
    if failed_or_killed:
        print(f"{len(failed_or_killed)} task(s) failed or were killed.")
        sys.exit(1)
    print("All tasks completed successfully.")


if __name__ == "__main__":
    # Ensure spawn is used so the child process gets a clean interpreter state
    # (avoids thread/connection-pool issues from forking).
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Run the MLE-STAR pipeline for every task folder."
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing task folders (default: %(default)s).",
    )
    parser.add_argument(
        "--workspace-dir",
        default=DEFAULT_WORKSPACE_DIR,
        help="Directory for outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tokens-per-task",
        type=int,
        default=DEFAULT_MAX_TOKENS_PER_TASK,
        help="Maximum tokens per task before killing it (default: %(default)s).",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=DEFAULT_MAX_TOTAL_TOKENS,
        help="Maximum total tokens for the whole batch. 0 means unlimited (default: %(default)s).",
    )
    parser.add_argument(
        "--max-task-time",
        type=int,
        default=DEFAULT_MAX_TASK_TIME,
        help="Maximum seconds per task before killing it (default: %(default)s).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum attempts for failed/killed tasks on resume (default: %(default)s).",
    )
    parser.add_argument(
        "--task-name",
        action="append",
        default=None,
        help="Run only this specific task. Can be used multiple times.",
    )
    parser.add_argument(
        "--max-llm-retries",
        type=int,
        default=DEFAULT_MAX_LLM_RETRIES,
        help="Maximum retries for LLM timeout/connection errors (default: %(default)s).",
    )
    parser.add_argument(
        "--llm-retry-delay",
        type=float,
        default=DEFAULT_LLM_RETRY_DELAY,
        help="Base delay in seconds between LLM retries (default: %(default)s).",
    )
    parser.add_argument(
        "--max-calls-per-minute",
        type=int,
        default=DEFAULT_MAX_CALLS_PER_MINUTE,
        help="Maximum LLM calls per minute to respect rate limits (default: %(default)s).",
    )
    args = parser.parse_args()

    run_all_tasks(
        data_dir=args.data_dir,
        workspace_dir=args.workspace_dir,
        max_tokens_per_task=args.max_tokens_per_task,
        max_total_tokens=args.max_total_tokens,
        max_task_time=args.max_task_time,
        max_attempts=args.max_attempts,
        requested_task_names=args.task_name,
        max_llm_retries=args.max_llm_retries,
        llm_retry_delay=args.llm_retry_delay,
        max_calls_per_minute=args.max_calls_per_minute,
    )
