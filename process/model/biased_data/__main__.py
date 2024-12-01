import hydra
from omegaconf import DictConfig
from scripts.train import setup_dataset, train

from cate.infra.mlflow import MlflowClient
from cate.utils import get_logger, path_linker


@hydra.main(config_name="config.yaml", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    client = MlflowClient(cfg.mlflow.experiment_name)
    logger = get_logger("trainer")
    pathlink = path_linker(cfg.data.name)
    train_ds, test_ds = setup_dataset(cfg, logger, pathlink)
    train(cfg, client, logger, rank=1, train_ds=train_ds, test_ds=test_ds)


if __name__ == "__main__":
    main()
