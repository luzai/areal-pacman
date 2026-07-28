from __future__ import annotations

import argparse
import json
import os

import torch
from torch_memory_saver import torch_memory_saver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise torch-memory-saver pause/resume on one CUDA GPU."
    )
    parser.add_argument("--allocation-mib", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.allocation_mib < 1:
        raise ValueError("allocation-mib must be positive")
    if os.environ.get("TMS_INIT_ENABLE") != "1":
        raise RuntimeError("TMS_INIT_ENABLE=1 is required before Python starts")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    elements = args.allocation_mib * 1024 * 1024 // 4
    tensor = torch.full(
        (elements,),
        2.0,
        dtype=torch.float32,
        device="cuda",
    )
    torch.cuda.synchronize()
    expected_sum = float(elements * 2)

    torch_memory_saver.pause()
    torch_memory_saver.resume()
    torch.cuda.synchronize()
    actual_sum = float(tensor.sum().item())
    if actual_sum != expected_sum:
        raise RuntimeError(
            f"tensor changed across offload: {actual_sum} != {expected_sum}"
        )

    print(
        json.dumps(
            {
                "allocation_mib": args.allocation_mib,
                "cuda_device": torch.cuda.get_device_name(),
                "status": "ok",
                "tensor_sum": actual_sum,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
