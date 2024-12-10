from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
import hydra


@dataclass
class TrainerConfig:
    max_epochs: int
    gpus: int
    precision: int


cs = ConfigStore.instance()
cs.store(name="trainer1", node=TrainerConfig(max_epochs=10, gpus=1, precision=16))
cs.store(name="trainer2", node=TrainerConfig(max_epochs=20, gpus=2, precision=32))
cs.store(name="trainer3", node=TrainerConfig(max_epochs=30, gpus=3, precision=64))


@hydra.main(config_path="conf", config_name="config.yaml")
def my_app(cfg: TrainerConfig) -> None:
    print(type(cfg))
    print(cfg)


if __name__ == "__main__":
    my_app()
