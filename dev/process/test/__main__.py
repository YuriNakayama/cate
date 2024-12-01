import hydra
from omegaconf import DictConfig
from mlflow import MlflowClient


def setup_mlflow(cfg: DictConfig) -> MlflowClient:

    client =  MlflowClient(cfg.mlflow.tracking_uri)
    return client

@hydra.main(config_name="config.yaml", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    print(cfg)


if __name__ == "__main__":
    main()
