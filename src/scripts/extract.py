"""Hydra entry point for reusable causal feature extraction."""

import hydra
from omegaconf import DictConfig

from src.pipeline.workflow import run_stage


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_stage(cfg, "extract")


if __name__ == "__main__":
    main()
