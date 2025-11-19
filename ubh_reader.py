from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class UBHFrame:
    timestamp: int
    logtime: str
    ranges_mm: np.ndarray


def read_ubh_header(path: Path) -> Dict[str, str]:
    header: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            label = line.strip()
            if not label:
                continue
            if label == "[timestamp]":
                break
            value = fh.readline()
            if not value:
                break
            header[label.strip("[]")] = value.strip()
    return header


def iter_ubh_frames(path: Path) -> Iterator[UBHFrame]:
    def next_nonempty() -> Optional[str]:
        for raw in fh:
            stripped = raw.strip()
            if stripped:
                return stripped
        return None

    def read_value(expected: str) -> Optional[str]:
        while True:
            label = next_nonempty()
            if label is None:
                return None
            if label == expected:
                break
        return next_nonempty()

    with path.open("r", encoding="utf-8") as fh:
        timestamp_str = read_value("[timestamp]")
        if timestamp_str is None:
            return
        timestamp = int(timestamp_str)
        while True:
            logtime = read_value("[logtime]")
            if logtime is None:
                break
            scan_line = read_value("[scan]")
            if scan_line is None:
                break
            ranges = np.array(
                [int(v) for v in scan_line.split(";") if v],
                dtype=np.float32,
            )
            yield UBHFrame(timestamp=timestamp, logtime=logtime, ranges_mm=ranges)
            next_timestamp = read_value("[timestamp]")
            if next_timestamp is None:
                break
            timestamp = int(next_timestamp)


def compute_angles(header: Dict[str, str]) -> np.ndarray:
    total_steps = int(header["totalSteps"])
    start_step = int(header["startStep"])
    end_step = int(header["endStep"])
    front_step = int(header["frontStep"])
    step_angle = (2 * math.pi) / total_steps
    steps = np.arange(start_step, end_step + 1)
    return (steps - front_step) * step_angle
