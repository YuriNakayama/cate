import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import KeysView, Literal


@dataclass
class Event:
    time: datetime
    model: str
    phase: str
    category: Literal["start", "stop"]
    description: str | int = ""

    @classmethod
    def keys(cls) -> KeysView[str]:
        return cls.__dataclass_fields__.keys()


# TODO: 実行ファイルがwith句などでstart, stopできるように
class Timer:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def start(
        self, model: str, phase: str, description: str | int | None = None
    ) -> None:
        description = "" if description is None else description
        event = Event(datetime.now(), model, phase, "start", description)
        self.events.append(event)

    def stop(
        self, model: str, phase: str, description: str | int | None = None
    ) -> None:
        description = "" if description is None else description
        event = Event(datetime.now(), model, phase, "stop", description)
        self.events.append(event)

    def to_csv(self, path: Path) -> None:
        with path.open(mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(Event.keys()))
            writer.writerows([asdict(event) for event in self.events])
