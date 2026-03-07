from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpatialTiff:
    xpos: float
    ypos: float


@dataclass
class Rect:
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    @property
    def empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def intersect(self, other: "Rect") -> "Rect":
        x0 = max(self.x, other.x)
        y0 = max(self.y, other.y)
        x1 = min(self.right, other.right)
        y1 = min(self.bottom, other.bottom)
        return Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))


@dataclass
class CanvasInfo:
    width: int
    height: int
    positions: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Remapper:
    width: int = 0
    height: int = 0
    xpos: int = 0
