"""Per-stage search timing utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SearchTiming:
    """Accumulates wall-clock millisecond measurements for each search stage.

    Usage::

        t = SearchTiming()
        t.start("embed")
        embedding = model.encode(query)
        t.stop("embed")
        print(t.stages)   # {"embed": 12.4}
        print(t.total_ms) # 12.4
    """

    stages: dict[str, float] = field(default_factory=dict)
    _starts: dict[str, float] = field(default_factory=dict, repr=False)

    def start(self, stage: str) -> None:
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> None:
        t = self._starts.pop(stage, None)
        if t is not None:
            self.stages[stage] = round((time.perf_counter() - t) * 1000, 2)

    @property
    def total_ms(self) -> float:
        return round(sum(self.stages.values()), 2)

    def as_dict(self) -> dict:
        return {"stages_ms": dict(self.stages), "total_ms": self.total_ms}
