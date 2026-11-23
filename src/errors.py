class PipelineError(Exception):
    """Base class for all expected errors."""


class ConfigError(PipelineError):
    """config.yaml is missing, malformed, or has wrong values."""


class DataError(PipelineError):
    """The dataset is missing, unreadable, or not in the expected shape."""