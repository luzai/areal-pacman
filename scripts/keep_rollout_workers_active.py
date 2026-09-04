#!/usr/bin/env python3
"""Keep local rollout vLLM workers active during long PPO compute windows."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


SERVER_RE = re.compile(r"Starting vLLM server on (http://[^\s]+)")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def server_urls(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    urls = SERVER_RE.findall(log_path.read_text(errors="replace"))
    return list(dict.fromkeys(urls))


def exercise_worker(base_url: str, model: str, timeout: float) -> tuple[bool, str]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": "Return one letter: U",
            "max_tokens": 1,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return response.status == 200, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.read(300).decode(errors='replace')}"
    except Exception as exc:  # monitoring must not terminate training
        return False, repr(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-pid", type=int, required=True)
    parser.add_argument("--rollout-log", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()

    while process_alive(args.main_pid):
        urls = server_urls(args.rollout_log)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if not urls:
            print(f"[{stamp}] waiting for rollout server addresses", flush=True)
        for url in urls:
            ok, detail = exercise_worker(url, args.model, args.timeout)
            print(f"[{stamp}] {url} ok={ok} {detail}", flush=True)
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
