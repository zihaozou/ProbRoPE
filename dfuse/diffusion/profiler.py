"""Per-step timing profiler for training.

Two implementations share an interface:
    - NullProfiler: no-op, used when profiling is disabled.
    - StepProfiler: records CUDA events at stage boundaries and returns
      elapsed milliseconds at the end of each step.

Usage:
    profiler = StepProfiler(device=...) if enabled else NullProfiler()
    profiler.mark_step_start()
    with profiler.stage("forward"):
        ...
    with profiler.stage("backward"):
        ...
    profiler.end_step()
    readings = profiler.readout_ms()   # dict[str, float]
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Optional

import torch


class NullProfiler:
    """No-op profiler. All methods return immediately."""

    @contextmanager
    def stage(self, name: str):
        yield

    def mark_step_start(self) -> None:
        pass

    def end_step(self) -> None:
        pass

    def readout_ms(self) -> Dict[str, float]:
        return {}


class StepProfiler:
    """Active profiler using CUDA events for GPU stages.

    One instance lives for the entire training run. `mark_step_start` resets
    per-step state; `stage()` records enter/exit events; `end_step()` syncs
    once and finalizes readings.

    Dataload timing is the perf_counter gap between previous `end_step` and
    this `mark_step_start`. This is a CPU wall-clock measurement and does not
    use CUDA events.

    If CUDA is unavailable or `device.type != 'cuda'`, all stages fall back
    to perf_counter. This allows unit-testing on CPU-only machines.
    """

    def __init__(self, device: Optional[torch.device] = None, enabled: bool = True):
        self.enabled = enabled
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._use_cuda = self.device.type == "cuda" and torch.cuda.is_available()


        self._spans: Dict[str, tuple] = {}

        self._event_pool: list = []

        self._prev_end_perf: Optional[float] = None
        self._step_start_perf: Optional[float] = None
        self._dataload_ms: Optional[float] = None

        self._readings: Dict[str, float] = {}

    def _borrow_event(self) -> "torch.cuda.Event":
        if self._event_pool:
            return self._event_pool.pop()
        return torch.cuda.Event(enable_timing=True)

    def _return_events(self, *events) -> None:
        self._event_pool.extend(events)

    @contextmanager
    def stage(self, name: str):
        if self._use_cuda:
            start = self._borrow_event()
            end = self._borrow_event()
            start.record()
            try:
                yield
            finally:
                end.record()
                self._spans[name] = (start, end)
        else:
            start = time.perf_counter()
            try:
                yield
            finally:
                end = time.perf_counter()
                self._spans[name] = (start, end)

    def mark_step_start(self) -> None:
        now = time.perf_counter()
        if self._prev_end_perf is not None:
            self._dataload_ms = (now - self._prev_end_perf) * 1000.0
        else:
            self._dataload_ms = 0.0
        self._step_start_perf = now
        self._spans = {}

    def end_step(self) -> None:


        if self._use_cuda and self._spans:
            last_end = next(reversed(self._spans.values()))[1]
            last_end.synchronize()

        readings: Dict[str, float] = {}
        if self._dataload_ms is not None:
            readings["dataload"] = self._dataload_ms

        for name, (a, b) in self._spans.items():
            if self._use_cuda:
                readings[name] = a.elapsed_time(b)
                self._return_events(a, b)
            else:
                readings[name] = (b - a) * 1000.0

        self._readings = readings
        self._spans = {}
        self._prev_end_perf = time.perf_counter()

    def readout_ms(self) -> Dict[str, float]:
        return dict(self._readings)
