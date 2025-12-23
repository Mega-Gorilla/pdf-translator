# SPDX-License-Identifier: Apache-2.0
"""Progress callback protocol for translation pipeline."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProgressCallback(Protocol):
    """Progress callback protocol."""

    def __call__(
        self,
        stage: str,
        current: int,
        total: int,
        message: str = "",
    ) -> None: ...
