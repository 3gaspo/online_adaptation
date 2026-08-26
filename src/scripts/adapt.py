"""Hydra entry point for proposal adaptors and external-model evaluation."""

import hydra
from omegaconf import DictConfig

from src.pipeline.workflow import run_stage


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_stage(cfg, "adapt")


if __name__ == "__main__":
    main()
