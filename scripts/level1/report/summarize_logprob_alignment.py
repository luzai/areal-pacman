from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit rollout old log-probabilities against actor log-probabilities "
            "recomputed before the PPO optimizer update."
        )
    )
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-abs-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--mean-abs-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return math.nan
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def main() -> None:
    args = parse_args()
    files = sorted(args.audit_dir.glob("rank-*-call-*.json"))
    if not files:
        raise SystemExit(f"no alignment audit files found in {args.audit_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    action_token_ids: dict[int, tuple[str, int]] = {}
    for index, action in enumerate(("U", "D", "L", "R")):
        encoded = tokenizer.encode(action, add_special_tokens=False)
        if len(encoded) != 1:
            raise SystemExit(f"action {action!r} encoded as {encoded}, not one token")
        action_token_ids[encoded[0]] = (action, 1 << index)
    if len(action_token_ids) != 4:
        raise SystemExit("U/D/L/R do not map to four distinct tokenizer tokens")

    rows: list[dict[str, object]] = []
    all_deltas: list[float] = []
    mask_violations = 0
    version_mismatches = 0
    unknown_action_tokens = 0
    nonfinite_values = 0
    action_counts = {action: 0 for action in ("U", "D", "L", "R")}
    mask_counts: dict[str, int] = {}

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fields = (
            payload["old_logprobs"],
            payload["recomputed_logprobs"],
            payload["deltas"],
            payload["action_token_ids"],
            payload["action_mask_bits"],
            payload["versions"],
        )
        lengths = {len(field) for field in fields}
        if lengths != {payload["count"]}:
            raise SystemExit(f"inconsistent field lengths in {path}")

        current_version = int(payload["current_version"])
        for old, recomputed, delta, token_id, mask_bits, version in zip(
            *fields, strict=True
        ):
            old = float(old)
            recomputed = float(recomputed)
            delta = float(delta)
            mask_bits = int(mask_bits)
            version = int(version)
            token_id = int(token_id)
            if not all(math.isfinite(value) for value in (old, recomputed, delta)):
                nonfinite_values += 1
            if abs((recomputed - old) - delta) > 1.0e-6:
                raise SystemExit(f"stored delta mismatch in {path}")
            all_deltas.append(delta)
            mask_counts[str(mask_bits)] = mask_counts.get(str(mask_bits), 0) + 1

            action_entry = action_token_ids.get(token_id)
            if action_entry is None:
                unknown_action_tokens += 1
            else:
                action, action_bit = action_entry
                action_counts[action] += 1
                if not mask_bits & action_bit:
                    mask_violations += 1
            if version != current_version:
                version_mismatches += 1

        rows.append(
            {
                "file": path.name,
                "rank": payload["rank"],
                "call_index": payload["call_index"],
                "current_version": current_version,
                "count": payload["count"],
            }
        )

    absolute = sorted(abs(value) for value in all_deltas)
    signed_mean = statistics.fmean(all_deltas)
    mean_abs = statistics.fmean(absolute)
    rms = math.sqrt(statistics.fmean(value * value for value in all_deltas))
    max_abs = absolute[-1]
    report = {
        "aligned": (
            nonfinite_values == 0
            and unknown_action_tokens == 0
            and mask_violations == 0
            and version_mismatches == 0
            and max_abs <= args.max_abs_tolerance
            and mean_abs <= args.mean_abs_tolerance
        ),
        "files": len(files),
        "tokens": len(all_deltas),
        "signed_mean_delta": signed_mean,
        "mean_abs_delta": mean_abs,
        "rms_delta": rms,
        "p50_abs_delta": percentile(absolute, 0.50),
        "p90_abs_delta": percentile(absolute, 0.90),
        "p95_abs_delta": percentile(absolute, 0.95),
        "p99_abs_delta": percentile(absolute, 0.99),
        "max_abs_delta": max_abs,
        "max_abs_tolerance": args.max_abs_tolerance,
        "mean_abs_tolerance": args.mean_abs_tolerance,
        "nonfinite_values": nonfinite_values,
        "unknown_action_tokens": unknown_action_tokens,
        "mask_violations": mask_violations,
        "version_mismatches": version_mismatches,
        "action_counts": action_counts,
        "mask_counts": mask_counts,
        "records": rows,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if not report["aligned"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
