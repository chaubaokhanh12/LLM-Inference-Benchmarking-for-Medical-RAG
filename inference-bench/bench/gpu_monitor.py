"""Background samplers that run next to the load generator.

GpuMonitor   nvidia-smi every second -> gpu.csv (memory, utilisation, power)
MetricsPoller vLLM's Prometheus /metrics every second -> vllm_metrics.csv
              (running / waiting requests, KV-cache usage). Metric names change between
              vLLM versions, so every `vllm:*` gauge is kept and the analysis picks columns.
"""

from __future__ import annotations

import csv
import re
import subprocess
import threading
import time
from pathlib import Path

import httpx

NVIDIA_QUERY = "timestamp,memory.used,memory.total,utilization.gpu,power.draw"


class GpuMonitor:
    def __init__(self, out: Path, interval_ms: int = 1000):
        self.out = out
        self.interval_ms = interval_ms
        self._proc: subprocess.Popen | None = None
        self._file = None

    def __enter__(self) -> GpuMonitor:
        self._file = open(self.out, "w", encoding="utf-8", newline="")
        self._file.write("unix_time," + NVIDIA_QUERY.replace(".", "_") + "\n")
        self._file.flush()
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={NVIDIA_QUERY}", "--format=csv,noheader,nounits",
                 f"-lms={self.interval_ms}"],
                stdout=subprocess.PIPE, text=True,
            )
        except FileNotFoundError:
            self._proc = None  # no NVIDIA GPU here; the run still works
            return self
        threading.Thread(target=self._pump, daemon=True).start()
        return self

    def _pump(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            if line.strip():
                self._file.write(f"{time.time():.3f},{line.strip()}\n")
                self._file.flush()

    def __exit__(self, *exc) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        if self._file:
            self._file.close()


_PROM_LINE = re.compile(r"^(vllm:[a-zA-Z0-9_:]+)(\{[^}]*\})?\s+([-0-9.eE+]+|NaN)$")


def parse_prometheus(text: str) -> dict[str, float]:
    """vllm:* samples, summed over label sets; histogram/summary sub-series are skipped."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _PROM_LINE.match(line.strip())
        if not m or m.group(1).endswith(("_bucket", "_sum", "_count", "_created")):
            continue
        try:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
        except ValueError:
            continue
    return out


class MetricsPoller:
    def __init__(self, url: str | None, out: Path, interval_s: float = 1.0):
        self.url = url
        self.out = out
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.rows: list[dict] = []

    def __enter__(self) -> MetricsPoller:
        if self.url:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        with httpx.Client(timeout=5) as client:
            while not self._stop.is_set():
                try:
                    self.rows.append({"unix_time": round(time.time(), 3),
                                      **parse_prometheus(client.get(self.url).text)})
                except httpx.HTTPError:
                    pass
                self._stop.wait(self.interval_s)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        if self.rows:
            keys = sorted({k for r in self.rows for k in r} - {"unix_time"})
            with open(self.out, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["unix_time", *keys])
                w.writeheader()
                w.writerows(self.rows)
