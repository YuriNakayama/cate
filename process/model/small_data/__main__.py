import hydra
from omegaconf import DictConfig

from cate.infra.mlflow import MlflowClient
from cate.utils import get_logger, path_linker, send_message
from process.model.small_data.scripts.train import train


@hydra.main(config_name="config.yaml", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    client = MlflowClient(cfg.mlflow.experiment_name)
    logger = get_logger("trainer")
    pathlink = path_linker(cfg.data.name)

    for sample_ratio in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        train(cfg, client, logger, pathlink, sample_ratio=sample_ratio, random_state=42)

    send_message(
        f"Training Finished small_data {cfg.model.name}, {cfg.data.name}, {sample_ratio}"
    )


if __name__ == "__main__":
    main()
