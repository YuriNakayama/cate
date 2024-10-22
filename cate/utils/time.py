import csv
from _collections_abc import dict_keys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclass
class Event:
    time: datetime
    name: str
    category: Literal["start", "stop"]

    @classmethod
    def keys(cls) -> dict_keys:
        return cls.__dataclass_fields__.keys()


class Timer:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def start(self, name: str) -> None:
        event = Event(datetime.now(), name, "start")
        self.events.append(event)

    def stop(self, name: str) -> None:
        event = Event(datetime.now(), name, "stop")
        self.events.append(event)

    def to_csv(self, path: Path) -> None:
        with path.open(mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(Event.keys()))
            writer.writerows([asdict(event) for event in self.events])
