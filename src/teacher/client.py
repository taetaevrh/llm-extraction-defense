"""Trusted query accountant and text-only process transport."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from src.task import normalize_label


class BudgetExceeded(RuntimeError):
    pass


class BudgetedTeacher:
    def __init__(self, text_backend, ledger_path, limits):
        self._backend = text_backend
        self.limits = dict(limits)
        self.attempts = Counter()
        self.records = []
        self._ledger = Path(ledger_path)
        self._ledger.touch(exist_ok=False)

    def query(self, text, phase):
        if phase not in self.limits or self.attempts[phase] >= self.limits[phase]:
            raise BudgetExceeded(f"Teacher budget exhausted or unknown phase: {phase}")
        self.attempts[phase] += 1
        record = {
            "phase": phase, "attempt": self.attempts[phase],
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        started = time.perf_counter()
        try:
            raw = self._backend(text)
            if not isinstance(raw, str):
                raise TypeError("Teacher interface must return text only")
            record.update({"output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                           "normalized_label": normalize_label(raw),
                           "status": "valid" if normalize_label(raw) else "invalid"})
            return raw
        except Exception as error:
            record.update({"status": "error", "error": str(error)})
            raise
        finally:
            record["latency_seconds"] = time.perf_counter() - started
            self.records.append(record)
            with self._ledger.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")


class TeacherProcess:
    """The model lives in a separate process; transport returns no internals."""
    def __init__(self, config_path, resource_path, stderr_path):
        self._stderr = Path(stderr_path).open("x", encoding="utf-8")
        self._process = subprocess.Popen(
            [sys.executable, "-u", "-m", "src.teacher.worker", str(config_path), str(resource_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, encoding="utf-8", bufsize=1,
        )
        ready = self._process.stdout.readline()
        if not ready or not json.loads(ready).get("ready"):
            self.close()
            raise RuntimeError("Teacher startup failed; inspect teacher stderr")

    def __call__(self, text):
        self._process.stdin.write(json.dumps({"text": text}) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("Teacher worker terminated")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["text"]

    def close(self):
        if self._process.poll() is None:
            self._process.stdin.close()
            try:
                self._process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
                raise RuntimeError("Teacher process did not exit cleanly")
        self._stderr.close()
        if self._process.returncode:
            raise RuntimeError(f"Teacher exited with code {self._process.returncode}")
