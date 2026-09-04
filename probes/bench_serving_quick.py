#!/usr/bin/env python3
"""Short stable-prompt decode and cold-prefill experiments; no server mutation."""
import argparse
import concurrent.futures
from contextlib import contextmanager
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from bench_prefill_mixed import health, prefill_once, stream_chat


PROMPTS = {
    "prose": "Explain how a hash table works, including collisions, resizing, and the tradeoffs versus a balanced search tree. Give concrete examples.",
    "code": "Implement a thread-safe token bucket rate limiter in Python. Include a monotonic clock, blocking and nonblocking acquisition, and unit tests. Explain the design.",
}


@contextmanager
def profile_session(url, enabled):
    def request(action):
        with urllib.request.urlopen(urllib.request.Request(
            url + "/" + action, data=b"", method="POST"), timeout=180) as response:
            if response.status != 200:
                raise RuntimeError(f"{action} returned {response.status}")
    if enabled:
        request("start_profile")
    try:
        yield
    finally:
        if enabled:
            request("stop_profile")


def metrics(url):
    with urllib.request.urlopen(url + "/metrics", timeout=20) as response:
        text = response.read().decode()
    names = ("spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
             "spec_decode_num_accepted_tokens_total", "num_preemptions_total")
    result = {}
    for name in names:
        result[name] = sum(float(m.group(1)) for m in re.finditer(
            rf"^vllm:{name}(?:\{{[^\n]*\}})? ([^\n]+)$", text, re.M))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=("decode", "prefill"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--levels", default="1,6")
    parser.add_argument("--prompts", default="prose,code")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prefill-tokens", default="32768,131072")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--profile", action="store_true",
                        help="Capture bounded worker traces; timings are diagnostic only")
    args = parser.parse_args()
    health(args.url)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": 1, "config": vars(args),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "cases": []}
    # Unique salt occurs before the large reference; never reset shared APC.
    salt = str(time.time_ns())
    for round_index in range(args.rounds):
        if args.mode == "prefill":
            for count in map(int, args.prefill_tokens.split(",")):
                with profile_session(args.url, args.profile):
                    result = prefill_once(args.url, count, case_id=f"{round_index}-{count}",
                        corpus_seed=salt, phrase_index=round_index)
                result["round"] = round_index
                result["cold"] = result["cached_prompt_tokens"] == 0
                report["cases"].append(result)
                print(json.dumps(result), flush=True)
                output.write_text(json.dumps(report, indent=2) + "\n")
        else:
            for c in map(int, args.levels.split(",")):
                for kind in args.prompts.split(","):
                    payload = {"model": "glm-5.3-flash", "messages": [
                        {"role": "user", "content": PROMPTS[kind]}],
                        "temperature": 0, "seed": 42, "ignore_eos": True,
                        "max_tokens": args.max_tokens,
                        "chat_template_kwargs": {"enable_thinking": False}}
                    before = metrics(args.url)
                    with profile_session(args.url, args.profile):
                        started = time.perf_counter()
                        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                            streams = list(pool.map(lambda _: stream_chat(args.url, payload), range(c)))
                    wall = max(s["finished_monotonic"] for s in streams) - started
                    after = metrics(args.url)
                    result = {"round": round_index, "concurrency": c, "kind": kind,
                              "wall_seconds": wall,
                              "aggregate_tokens_per_second": sum(s["completion_tokens"] for s in streams) / wall,
                              "streams": streams,
                              "metrics_delta": {k: after[k] - before[k] for k in before}}
                    report["cases"].append(result)
                    print(json.dumps(result), flush=True)
                    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
