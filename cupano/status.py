from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CudaStatus:
    code: int = 0
    message: str = ""

    def ok(self) -> bool:
        return self.code == 0


class CudaStatusError(RuntimeError):
    def __init__(self, status: CudaStatus):
        self.status = status
        super().__init__(status.message or f"CudaStatus(code={status.code})")
