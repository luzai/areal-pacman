# Historical MaaPacman runtime subset

This is the 13-file Python environment / bundled Level 1 subset of the immutable
Iter25 512-step source snapshot. It is not a nested Git repository and contains
no Windows input-control agent, model weights, or Python environment.

Source archive SHA256:
`f0fdf3961fd2bab89ef5f69d9e9cf65f95095d1cacd83d4842ef4e73aa0e508f`.
The source files were checked against the snapshot's historical SHA256 manifest;
vendored contents retain the same text, with LF line endings.

Do not assume this subset is also the exact first-stage environment.
This subset is published as a historical draft at the repository owner's request.
See the root THIRD_PARTY_NOTICES.md; no new blanket license is asserted.
All-stage dependency closure and final-release review remain pending.
Set the explicit game-root environment variable
when testing; the old sibling-directory fallback is not valid at this location.
