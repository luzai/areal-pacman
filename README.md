# Historical C1 / Iter25 lineage — DRAFT

This branch is a clean, single-root source snapshot for historical reconstruction.
It is **not a runnable or validated training release**. No previous private Git
history, weights, raw training logs, machine environments, or investigation
archives are included. The release branches are unchanged.

## Target recipe

Qwen3.5-9B base -> 16 updates at horizon 256 -> 1 update at horizon 700 ->
8 updates at horizon 512 (historical Iter25 / global step 24).

| Stage | Actor initialization | Optimizer | Cumulative stop |
| --- | --- | --- | --- |
| 1 | Base model | Fresh | 16 |
| 2 | This reproduction's Iter16 HF export | Fresh | 17 |
| 3 | This reproduction's Iter17 DCP state | Restore | 25 |

Validation is always OFF. Preserve historical LR, reward, GC and offload settings.
The reference model remains the base. Historical scheduler reconstruction and
fresh RNG seeding mean this is not bit-exact uninterrupted checkpoint recovery.
Do not substitute historical trained weights for newly produced checkpoints.

## Included source

- `areal_pacman/`, `train_areal.py`, startup shims and packaging metadata:
  the selected third-stage source snapshot, excluding backup files.
- `legacy/MaaPacman/maapacman/`: the 13-file third-stage environment subset.
- `configs/repro/`: three draft configurations with portable path placeholders.

**The included trainer is third-stage source, not a universal runtime for all
three stages.** The fail-closed orchestrator at
`scripts/repro/run_c1_iter25_lineage.sh` therefore requires three separately
verified recipe roots, matching AReaL roots, and per-stage MaaPacman roots. It
runs 16 + 1 + 8 updates,
exports the Iter16 HF boundary, shares only the Stage 2/3 recovery namespace,
checks that the DCP contains optimizer state, and requires Stage 3 to report the
model+optimizer recovery path. Run `--preflight-only` before allocating GPUs and
`--smoke` for a one-update-per-stage handoff test.

The historical packaging dependency on `maapacman` is retained; this draft does
not supply a complete install recipe.
Do not install an unrelated package of the same name to fill that gap.

## Remaining gates

1. Establish the exact first-stage MaaPacman identity and per-stage source selection.
2. Reconstruct fixed data/seeds, runtime versions and startup environment.
3. Test the automated Iter16 export, Iter17 optimizer transfer and stopping
   boundaries against the real Qwen3.5-9B workload.
4. Integrate the matching AReaL snapshots; no new AReaL runtime is published here.
5. Complete real-model/GPU training acceptance and disk-retention checks.

See [test status and bug decisions](docs/test-status.md) and
[source attribution](THIRD_PARTY_NOTICES.md). This draft does not grant a new
blanket license over historical or third-party code.
