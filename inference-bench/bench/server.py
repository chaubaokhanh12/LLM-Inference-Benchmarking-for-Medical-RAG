"""Start an inference server from its command line, wait until it answers, stop it."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx


class ManagedServer:
    """`with ManagedServer(cmd, health_url, log): ...` — no-op when cmd is None (server already up)."""

    def __init__(self, command: str | None, health_url: str, log_path: Path,
                 startup_timeout_s: float = 900.0):
        self.command = command
        self.health_url = health_url
        self.log_path = log_path
        self.startup_timeout_s = startup_timeout_s
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> ManagedServer:
        if self.command:
            log = open(self.log_path, "w", encoding="utf-8")
            kwargs = {"start_new_session": True} if os.name != "nt" else {
                "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            args = self.command if os.name == "nt" else shlex.split(self.command)
            self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                         shell=os.name == "nt", **kwargs)
        self.wait_ready()
        return self

    def wait_ready(self) -> None:
        deadline = time.time() + self.startup_timeout_s
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"server exited with code {self.proc.returncode}; see {self.log_path}")
            try:
                if httpx.get(self.health_url, timeout=5).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(3)
        raise TimeoutError(f"server not ready after {self.startup_timeout_s}s: {self.health_url}")

    def __exit__(self, *exc) -> None:
        if not self.proc or self.proc.poll() is not None:
            return
        if os.name == "nt":
            self.proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        print(f"server stopped (log: {self.log_path})", file=sys.stderr)
