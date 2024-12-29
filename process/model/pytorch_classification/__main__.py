import hydra
from omegaconf import DictConfig

from cate.infra.mlflow import MlflowClient
from cate.utils import get_logger, path_linker, send_message
from process.model.base_pattern.scripts.train import train


@hydra.main(config_name="config.yaml", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    client = MlflowClient(cfg.mlflow.experiment_name)
    logger = get_logger("trainer")
    pathlink = path_linker(cfg.data.name)

    tags = {
        "dataset": cfg.data.name,
        "package": "pytorch",
        "layer": "parent",
    }
    run_ids = client.search_runs_by_tags(tags)
    if not run_ids:
        parent_run = client.start_run(
            run_name=f"{cfg.data.name}",
            tags=tags,
            description=f"Pytorch classification model training for {cfg.data.name}",  # noqa: E501
        )
        client.end_run()
        parent_run_id = parent_run.info.run_id
    else:
        parent_run_id = run_ids[0]

    train(
        cfg,
        pathlink,
        client,
        logger,
        parent_run_id=parent_run_id,
    )
    send_message("Training Finished pytorch classification")


if __name__ == "__main__":
    main()
