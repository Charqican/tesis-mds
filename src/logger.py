from loguru import logger

sampling_logger = logger.bind(module="feature_extractor.sampling")
rendering_logger = logger.bind(module="feature_extractor.rendering")
backprojection_logger = logger.bind(module="feature_extractor.backprojection")
pipeline_logger = logger.bind(module="feature_extractor.pipeline")
