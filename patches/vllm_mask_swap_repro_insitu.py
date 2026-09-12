#!/usr/bin/env python3
"""In-situ proof: drive the REAL vllm InputBatch.swap_states() and show that
a request's allowed_token_ids mask is destroyed.

Runs on CPU only. Does not touch the installed vLLM package.
"""
import torch
from vllm.v1.worker.gpu_input_batch import InputBatch

VOCAB = 64
ALLOWED_A = [11, 12]   # request A may only emit 11,12
ALLOWED_B = [41, 42]   # request B may only emit 41,42


def new_batch():
    return InputBatch(
        max_num_reqs=8,
        max_model_len=256,
        max_num_batched_tokens=1024,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=VOCAB,
        block_sizes=[16],
        kernel_block_sizes=[16],
    )


def install_mask(batch, row, allowed):
    """Exactly what add_request() does for a constrained request."""
    if batch.allowed_token_ids_mask_cpu_tensor is None:
        batch.allowed_token_ids_mask = torch.zeros(
            batch.max_num_reqs, VOCAB, dtype=torch.bool)
        batch.allowed_token_ids_mask_cpu_tensor = torch.zeros(
            batch.max_num_reqs, VOCAB, dtype=torch.bool)
    if allowed is not None:
        batch.allowed_token_ids_mask_cpu_tensor[row] = True
        batch.allowed_token_ids_mask_cpu_tensor[row][allowed] = False


def allowed_of(batch, row):
    return (~batch.allowed_token_ids_mask_cpu_tensor[row]).nonzero().flatten().tolist()


def register(batch, row, req_id):
    """Minimal bookkeeping so swap_states() can run."""
    while len(batch._req_ids) <= row:
        batch._req_ids.append(None)
        batch.req_output_token_ids.append(None)
        batch.spec_token_ids.append([])
    batch._req_ids[row] = req_id
    batch.req_output_token_ids[row] = []
    batch.req_id_to_index[req_id] = row
    batch.num_tokens_no_spec[row] = 1
    batch.num_prompt_tokens[row] = 1
    batch.num_computed_tokens_cpu[row] = 1


def scenario(name, allowed_row1, expect_desc):
    print("=" * 70)
    print(name)
    print("=" * 70)
    b = new_batch()
    register(b, 0, "req-A")
    register(b, 1, "req-B")
    install_mask(b, 0, ALLOWED_A)
    install_mask(b, 1, allowed_row1)

    before0, before1 = allowed_of(b, 0), allowed_of(b, 1)
    print(f"before swap: row0(req-A) allows {before0}")
    print(f"             row1(req-B) allows {before1}")

    # The real vLLM call, as invoked from
    # vllm/v1/attention/backends/utils.py:703 (reorder_batch...)
    b.swap_states(0, 1)

    after0, after1 = allowed_of(b, 0), allowed_of(b, 1)
    print(f"after swap : row0 allows {after0}")
    print(f"             row1 allows {after1}")
    print(f"expected   : row0 {before1}")
    print(f"             row1 {before0}")
    ok = (after0 == before1 and after1 == before0)
    print(f"==> correct swap? {ok}")
    if not ok:
        print(f"==> BUG CONFIRMED IN REAL swap_states(): {expect_desc}")
    print()
    return ok


print("vLLM InputBatch.swap_states() -- allowed_token_ids_mask integrity\n")

ok1 = scenario(
    "SCENARIO 1: two constrained requests are reordered",
    ALLOWED_B,
    "req-A lost its own mask and inherited req-B's allowed set "
    f"({ALLOWED_B}). req-A can now emit tokens it must never emit.",
)

ok2 = scenario(
    "SCENARIO 2: a constrained request is reordered against an "
    "unconstrained one",
    None,
    "req-A's constraint vanished entirely -> it samples over the FULL "
    "vocabulary. This is the 'mask silently dropped' symptom.",
)

print("=" * 70)
print(f"RESULT: swap_states preserves masks = {ok1 and ok2}")
print("=" * 70)
