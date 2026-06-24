from loguru import logger

sampling_logger = logger.bind(module="feature_extractor.sampling")
rendering_logger = logger.bind(module="feature_extractor.rendering")
backprojection_logger = logger.bind(module="feature_extractor.backprojection")
pipeline_logger = logger.bind(module="feature_extractor.pipeline")
ingestion_logger = logger.bind(module="data_loaders.ingest")
dataload_logger = logger.bind(module="data_loaders.dataload")
metrics_logger = logger.bind(module="metrics")
transformations_logger = logger.bind(module="transformations")
