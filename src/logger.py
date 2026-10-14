import time

from loguru import logger

sampling_logger = logger.bind(module="feature_extractor.sampling")
rendering_logger = logger.bind(module="feature_extractor.rendering")
backprojection_logger = logger.bind(module="feature_extractor.backprojection")
pipeline_logger = logger.bind(module="feature_extractor.pipeline")
ingestion_logger = logger.bind(module="data_loaders.ingest")
dataload_logger = logger.bind(module="data_loaders.dataload")
metrics_logger = logger.bind(module="metrics")
transformations_logger = logger.bind(module="transformations")
registrate_logger = logger.bind(module="pose6d.registrate")
pose6d_preprocessing_logger = logger.bind(module="pose6d.preprocessing")
scripts_extraction_logger = logger.bind(module="scripts.extract_pT")
scripts_propagation_logger = logger.bind(module="scripts.propagate_features")
pose6d_dataset_logger = logger.bind(module="pose6d.dataset")
notebook_logger = logger.bind(module="Notebooks")
experiments_logger = logger.bind(module="experiments")
training_logger = logger.bind(module="experiments.training")


class TimedLogger:
    """Logs at most once every `every_s` seconds (e.g. training progress)."""

    def __init__(self, log, every_s: float = 60.0):
        self.log = log
        self.every_s = every_s
        self._last = time.monotonic()

    def __call__(self, msg: str, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last >= self.every_s:
            self.log.info(msg)
            self._last = now
