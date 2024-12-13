import logging
import os
import tempfile
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf, open_dict

from src.scmap_client_server.benchmarks import BenchmarkCollector
from src.utils.scmap_applicator import Applicator

logger = logging.getLogger("main")

@hydra.main(config_path="scenario_configs", config_name="base", version_base=None)
def main(cfg: DictConfig):

    artifact_path = Path(cfg.eval.artifacts_base) / cfg.scenario_identifier / cfg.mode / str(cfg.run)
    artifact_path.mkdir(parents=True, exist_ok=True)

    try:
        with open(cfg.dataset.dataset_config, "r") as f:
            datasets_precomputed = dict()
            datasets_precomputed['datasets_precomputed'] = yaml.safe_load(f)
            datasets_precomputed = OmegaConf.create(datasets_precomputed)
            cfg = OmegaConf.merge(cfg, datasets_precomputed)
    except:
        pass

    # setup logging
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(artifact_path / "run.log")
    fileHandler.setFormatter(rootLogger.handlers[0].formatter)
    rootLogger.addHandler(fileHandler)

    # persist config
    cfg_serialized = artifact_path / "config.yaml"

    with open(cfg_serialized, mode="w") as f:
        OmegaConf.save(config=cfg, f=f)

    with open_dict(cfg):
        cfg.artifact_path = artifact_path

    # Start scE(match)
    with tempfile.TemporaryDirectory() as write_directory:
        benchmark_collector = BenchmarkCollector(cfg, cfg.artifact_path)

        if cfg.benchmark.skip_existing_runs and benchmark_collector.run_exists():
            logger.info("Run already exists, skipping...")
            return

        benchmark_collector.start_consumer()

        applicator = Applicator(benchmark_collector)
        applicator.apply_scmap(
            cfg,
            client1_ref_data_file=cfg.dataset.client1_ref_data_file,
            client2_ref_data_file=cfg.dataset.client2_ref_data_file,
            client1_query_data_file=cfg.dataset.client1_query_data_file,
            client2_query_data_file=cfg.dataset.client2_query_data_file,
            cluster_type_key_1=cfg.dataset.cluster_type_key_1,
            cluster_type_key_2=cfg.dataset.cluster_type_key_2,
            mode=cfg.mode,
            ckks_params={
                'n_exp': cfg.he_params.n_exp,
                'scale_exp': cfg.he_params.scale_exp,
                'qi_sizes': cfg.he_params.qi_sizes,
            },
            sent_data_dir=Path(write_directory)
        )
        benchmark_collector.finalize()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)7s - %(name)s - %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=os.getenv("LOGLEVEL", "DEBUG"))
    main()

