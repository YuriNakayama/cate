import hydra
from omegaconf import DictConfig

from cate.infra.mlflow import MlflowClient
from cate.utils import get_logger, path_linker, send_messages
from process.model.base_pattern.scripts.train import train


@hydra.main(config_name="config.yaml", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    client = MlflowClient(cfg.mlflow.experiment_name)
    logger = get_logger("trainer")
    pathlink = path_linker(cfg.data.name)
    
    
    train(
        cfg,
        pathlink,
        client,
        logger,
        sample_ratio=0.1,
    )
    send_messages([f"Training Finished biased_data {cfg.model.name}"])


if __name__ == "__main__":
    main()
