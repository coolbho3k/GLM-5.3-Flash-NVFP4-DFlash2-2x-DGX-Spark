#!/usr/bin/env python3
"""Build a public mixed/long-context JSONL corpus for NVFP4 KV calibration.

The source rows are cached, and only prompts are used; reference assistant
answers are deliberately excluded. Long prompts are sized with vLLM's
``/tokenize`` endpoint so the target lengths match the deployed tokenizer.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import urllib.parse
import urllib.request


DATASETS = {
    "cnn": ("abisee/cnn_dailymail", "3.0.0", "train", 100),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "default", "train_sft", 48),
    "humaneval": ("openai/openai_humaneval", "openai_humaneval", "test", 32),
    "gsm8k": ("openai/gsm8k", "main", "train", 32),
    "xnli": ("facebook/xnli", "all_languages", "train", 30),
}
LONG_TARGETS = (32768, 131072, 262144)
LANGUAGES = ("ar", "bg", "de", "el", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh")


def _fetch_rows(cache: Path, name: str, *, offline: bool) -> list[dict]:
    dataset, config, split, count = DATASETS[name]
    path = cache / f"{name}.json"
    if not path.exists():
        if offline:
            raise FileNotFoundError(f"offline source cache is missing {path}")
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": 0,
                "length": count,
            }
        )
        request = urllib.request.Request(
            f"https://datasets-server.huggingface.co/rows?{query}",
            headers={"User-Agent": "glm53-nvfp4-calibration/1"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    result = [entry["row"] for entry in json.loads(path.read_text())["rows"]]
    if len(result) < count:
        raise ValueError(f"{name}: expected {count} rows, received {len(result)}")
    return result[:count]


def _token_count(url: str, model: str, prompt: str) -> int:
    request = urllib.request.Request(
        url,
        data=json.dumps({"model": model, "prompt": prompt}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return len(json.load(response)["tokens"])


def _long_prompt(kind: str, count: int, articles: list[str]) -> str:
    needle = "CALIBRATION-NEEDLE-7B3D9A"
    records = []
    needle_index = max(1, (count * 3) // 5)
    for index in range(count):
        text = articles[(index * 37 + count * 11) % len(articles)]
        if kind == "retrieval" and index == needle_index:
            text += (
                f"\nThe validation key for archive segment {index:06d} "
                f"is {needle}."
            )
        category = (index * 17 + len(text)) % 13
        records.append(
            f"ARCHIVE RECORD {index:06d} CATEGORY {category:02d}\n"
            f"{text}\nEND RECORD {index:06d}\n"
        )
    body = "\n".join(records)
    if kind == "retrieval":
        prefix = "Read the archive records and return the exact validation key. Do not guess.\n\n"
        suffix = "\n\nQuestion: What exact validation key appears in the archive?"
    else:
        prefix = "Read the archive records. Each header contains CATEGORY followed by an integer.\n\n"
        suffix = "\n\nQuestion: How many records have CATEGORY 07? State the count and method briefly."
    return prefix + body + suffix


def _sized_long(
    kind: str,
    target: int,
    articles: list[str],
    *,
    tokenize_url: str,
    model: str,
) -> tuple[int, str]:
    count = max(8, target // 700)
    best: tuple[int, str] | None = None
    for _ in range(5):
        prompt = _long_prompt(kind, count, articles)
        measured = _token_count(tokenize_url, model, prompt)
        if measured <= target and (best is None or measured > best[0]):
            best = (measured, prompt)
        if 0.97 * target <= measured <= target:
            return measured, prompt
        next_count = max(1, int(count * (target * 0.985) / measured))
        if next_count == count:
            next_count += -1 if measured > target else 1
        count = next_count
    if best is None:
        prompt = _long_prompt(kind, max(1, count - 1), articles)
        best = (_token_count(tokenize_url, model, prompt), prompt)
    return best


def _build_short(sources: dict[str, list[dict]]) -> list[dict]:
    result = [
        {"bucket": "natural-chat", "prompt": row["prompt"], "max_tokens": 64}
        for row in sources["ultrachat"]
    ]
    result.extend(
        {
            "bucket": "summarization",
            "prompt": (
                "Summarize the following article accurately and concisely. "
                "Preserve important names, numbers, and qualifications.\n\n"
                + row["article"]
            ),
            "max_tokens": 64,
        }
        for row in sources["cnn"][:48]
    )
    result.extend(
        {
            "bucket": "code",
            "prompt": (
                "Complete this Python function correctly. Return the implementation "
                "and a short explanation.\n\n" + row["prompt"]
            ),
            "max_tokens": 64,
        }
        for row in sources["humaneval"]
    )
    result.extend(
        {
            "bucket": "math-reasoning",
            "prompt": (
                "Solve this problem carefully, showing the essential reasoning and "
                "final answer.\n\n" + row["question"]
            ),
            "max_tokens": 64,
        }
        for row in sources["gsm8k"]
    )
    for index, row in enumerate(sources["xnli"]):
        language = LANGUAGES[index % len(LANGUAGES)]
        translations = dict(
            zip(row["hypothesis"]["language"], row["hypothesis"]["translation"])
        )
        result.append(
            {
                "bucket": "multilingual-nli",
                "prompt": (
                    f"Language: {language}. Determine whether the hypothesis is "
                    "entailed by, neutral to, or contradicts the premise. Give the "
                    "label and a brief explanation in the same language.\n\n"
                    f"Premise: {row['premise'][language]}\n"
                    f"Hypothesis: {translations[language]}"
                ),
                "max_tokens": 64,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--tokenize-url", default="http://127.0.0.1:8000/tokenize")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-long", action="store_true")
    args = parser.parse_args()

    sources = {
        name: _fetch_rows(args.cache_dir, name, offline=args.offline)
        for name in DATASETS
    }
    result = _build_short(sources)
    long_lengths = []
    if not args.skip_long:
        articles = [row["article"] for row in sources["cnn"]]
        for kind in ("retrieval", "aggregation"):
            for target in LONG_TARGETS:
                measured, prompt = _sized_long(
                    kind,
                    target,
                    articles,
                    tokenize_url=args.tokenize_url,
                    model=args.model,
                )
                print(f"{kind}: target={target} measured={measured}")
                result.append(
                    {
                        "bucket": f"ruler-{kind}",
                        "prompt": prompt,
                        "max_tokens": 32,
                        "target_prompt_tokens": target,
                        "measured_prompt_tokens": measured,
                    }
                )
                long_lengths.append(measured)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in result:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    manifest = {
        "schema": "glm-nvfp4-mla-corpus-v1",
        "rows": len(result),
        "buckets": dict(sorted(Counter(row["bucket"] for row in result).items())),
        "long_prompt_tokens": long_lengths,
        "reference_answers_included": False,
        "source_cache": str(args.cache_dir),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} and {manifest_path}")


if __name__ == "__main__":
    main()
