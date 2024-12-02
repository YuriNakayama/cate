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
    train_ds, test_ds, rank_df = setup_dataset(cfg, logger, pathlink)
    for rank in range(cfg.model.num_rank):
        train(cfg, client, logger, rank=rank, train_ds=train_ds, test_ds=test_ds, rank_df=rank_df)


if __name__ == "__main__":
    main()
