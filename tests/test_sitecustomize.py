from __future__ import annotations

import pytest

from sitecustomize import _tms_binary_from_ld_preload


TMS = "/env/site-packages/torch_memory_saver_hook_mode_preload.abi3.so"
STDBUF = "/usr/libexec/coreutils/libstdbuf.so"


@pytest.mark.parametrize(
    "value",
    [
        TMS,
        f"{TMS}:{STDBUF}",
        f"{TMS}:{STDBUF}:{STDBUF}",
        f"{TMS} {STDBUF}",
    ],
)
def test_tms_binary_is_selected_from_multi_library_preload(value: str) -> None:
    assert _tms_binary_from_ld_preload(value) == TMS


def test_tms_binary_is_required() -> None:
    with pytest.raises(RuntimeError, match="does not contain"):
        _tms_binary_from_ld_preload(STDBUF)
