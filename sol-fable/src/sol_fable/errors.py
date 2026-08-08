"""Project-specific exceptions."""


class SolFableError(Exception):
    """Base exception for expected Sol-Fable failures."""


class ConfigurationError(SolFableError):
    """Raised when project configuration is invalid."""


class DocumentParseError(SolFableError):
    """Raised when an input document cannot be parsed safely."""


class PipelineStateError(SolFableError):
    """Raised when a stage is run without its required prior state."""


class PipelineRunError(SolFableError):
    """A full run failed after its auditable run ID had already been allocated."""

    def __init__(self, run_id: str, cause: Exception):
        self.run_id = run_id
        self.cause = cause
        super().__init__(f"Run {run_id} failed ({type(cause).__name__}): {cause}")


class LLMAssessmentError(SolFableError):
    """Raised when a live LLM backend cannot produce a schema-valid response after retries."""
