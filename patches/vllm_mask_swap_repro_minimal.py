#!/usr/bin/env python3
"""Prove the allowed_token_ids_mask row-swap bug in vLLM's InputBatch.swap_states.

vllm/v1/worker/gpu_input_batch.py (0.22.1), lines 669-676:

    if self.allowed_token_ids_mask_cpu_tensor is not None:
        (
            self.allowed_token_ids_mask_cpu_tensor[i1],
            self.allowed_token_ids_mask_cpu_tensor[i2],
        ) = (
            self.allowed_token_ids_mask_cpu_tensor[i2],
            self.allowed_token_ids_mask_cpu_tensor[i1],
        )

For a 2-D torch tensor, t[i] returns a VIEW, not a copy. The tuple-swap idiom
therefore does NOT swap the rows: the first assignment overwrites row i1, and
the second assignment then reads that already-overwritten row.
"""
import torch

VOCAB = 8
ALLOWED_A = [1, 2]          # request A may only emit tokens 1,2
ALLOWED_B = [5, 6]          # request B may only emit tokens 5,6


def build_mask_rows():
    """Mimic add_request(): row=True everywhere, False for allowed ids."""
    t = torch.zeros(4, VOCAB, dtype=torch.bool)
    t[0] = True
    t[0][ALLOWED_A] = False   # row 0 -> request A
    t[1] = True
    t[1][ALLOWED_B] = False   # row 1 -> request B
    return t


def allowed_of(row):
    return (~row).nonzero().flatten().tolist()


print("=" * 66)
print("CASE 1 -- two constrained requests swap rows")
print("=" * 66)
t = build_mask_rows()
print(f"before: row0 allows {allowed_of(t[0])}   row1 allows {allowed_of(t[1])}")

# verbatim vLLM idiom
t[0], t[1] = t[1], t[0]

print(f"after : row0 allows {allowed_of(t[0])}   row1 allows {allowed_of(t[1])}")
print(f"expected (a correct swap): row0 {ALLOWED_B}   row1 {ALLOWED_A}")
buggy_1 = allowed_of(t[0]) == ALLOWED_B and allowed_of(t[1]) == ALLOWED_B
print(f"==> BUG REPRODUCED: {buggy_1}  (row1's mask clobbered row0; A's mask LOST)")

print()
print("=" * 66)
print("CASE 2 -- constrained request swaps with an UNCONSTRAINED one")
print("        (this is the 'mask silently dropped' symptom)")
print("=" * 66)
t = torch.zeros(4, VOCAB, dtype=torch.bool)
t[0] = True
t[0][ALLOWED_A] = False   # row 0 -> constrained request A
# row 1 -> unconstrained request (add_request never touches the row: all False)
print(f"before: row0 allows {allowed_of(t[0])}   row1 allows {allowed_of(t[1])} (= no constraint)")

t[0], t[1] = t[1], t[0]

print(f"after : row0 allows {allowed_of(t[0])}   row1 allows {allowed_of(t[1])}")
lost = allowed_of(t[0]) == list(range(VOCAB)) and allowed_of(t[1]) == list(range(VOCAB))
print(f"==> BUG REPRODUCED: {lost}")
print("    Both rows now say 'every token allowed'. The constrained request")
print("    samples UNCONSTRAINED -> emits its natural top-1 token -> the")
print("    caller sees a token outside allowed_token_ids. Exactly the")
print("    ObjectiveParseError symptom.")

print()
print("=" * 66)
print("THE FIX -- clone the source rows before assigning")
print("=" * 66)
t = build_mask_rows()
print(f"before: row0 allows {allowed_of(t[0])}   row1 allows {allowed_of(t[1])}")
tmp = t[0].clone()
t[0].copy_(t[1])
t[1].copy_(tmp)
print(f"after : row0 allows {allowed_of(t[0])}   row1 allows {allowed_of(t[1])}")
fixed = allowed_of(t[0]) == ALLOWED_B and allowed_of(t[1]) == ALLOWED_A
print(f"==> FIX CORRECT: {fixed}")

print()
print("NOTE: the neighbouring swaps in swap_states() are fine because")
print("      temperature_cpu/top_p_cpu/... are 1-D numpy arrays, where")
print("      arr[i] yields a scalar COPY. Only allowed_token_ids_mask")
print("      is 2-D, where t[i] yields a VIEW. That is why only this")
print("      one is broken.")
