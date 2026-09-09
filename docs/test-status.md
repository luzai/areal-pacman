# Evidence and limitations

Source archive SHA256 for the selected third-stage snapshot:
`f0fdf3961fd2bab89ef5f69d9e9cf65f95095d1cacd83d4842ef4e73aa0e508f`.
Included historical files were checked against that snapshot's original manifest.
Publication excludes `.orig` backups and preserves source text with LF normalization.

The external game version checked locally was pacman-python
`d258122eecf6e0dc0a04d6fb8ff57a9b43f0c1d8`. The game is not bundled here.
This does not establish the exact first-stage game/environment identity.

## Completed diagnostics (not full training acceptance)

- Real headless third-stage legacy game: 256/700/512 horizons, two resets each,
  2,936 steps; image shape, truncation and worker cleanup passed.
- Those two seeds gave identical frames. Old reset records seed rather than
  transmitting it to the game worker; not independent randomized scenarios.
- Four-process CPU Gloo dispatch/packing: three synthetic cases for each of
  three historical sources passed. Variable row counts 42/48/41/49 produced
  matching planned forward counts across ranks and preserved loss-mask tokens.
  This did not execute GPU forward/backward or test gradients/RTensor dispatch.
- Historical serializer + local scheduler's orjson codec: all three sources
  turn inf/-inf/NaN into null/None; finite 20.0 is preserved. This proves a codec
  bug, not that original C1 parameters triggered it.

## Bug decisions

No new fixes below have been applied to the published historical source:

| Candidate | Status |
| --- | --- |
| a15a66a0 nonfinite RPC encoding | Bug reproduced; candidate for separately tested backport |
| ee872bae grouped PPO row counting | New interface differs; no direct cherry-pick justified by current tests |
| 1b31d73d Qwen3.5 row/GDN isolation | Missing from old source; real-model numerical comparison pending |
| 98028b2b Qwen3.5 multimodal position IDs | Missing from old source; real-image position/logprob verification pending |
| Image encoding reuse | Optional optimization; legacy input equivalence must be tested |

The inspected old game treats ghost-door tile 1 as traversable; its wall range
is 100-199 and action masking uses the same IsWall predicate. Changing that
behavior changes the historical environment. A local positioned-fixture test
passed with real game movement and unchanged worker masks: (10,10) -> (11,10)
-> (12,10) using D, and the reverse using U (zero-based row,column). Both masks
allowed the direction before entry and actual coordinates crossed the door.
This was not a learned-policy rollout or a remote doorway test.

Keep an unmodified historical reference; label any later correctness-fixed
variant explicitly. Neither version currently has completed reproduction training.
