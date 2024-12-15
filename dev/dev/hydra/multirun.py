from dataclasses import dataclass

import hydra
from omegaconf import DictConfig


@dataclass
class Model:
    name: str


class Train:
    max_epochs: int
    gpus: int
    precision: int
    max_epochs_2: int


@dataclass
class TrainerConfig:
    model: Model
    train: Train


@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def my_app(cfg_dict: DictConfig) -> None:
    cfg = TrainerConfig(**cfg_dict)  # type: ignore
    print(type(cfg.train))
    print(cfg)


if __name__ == "__main__":
    my_app()
