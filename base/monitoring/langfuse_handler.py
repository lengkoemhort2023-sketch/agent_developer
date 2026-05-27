import logging
import os

logger = logging.getLogger(__name__)

_handler = None
_initialized = False


def get_langfuse_handler():
    """
    Return a LangChain-compatible Langfuse CallbackHandler, or None if keys are absent.
    Initialises once and caches the result.
    """
    global _handler, _initialized
    if _initialized:
        return _handler

    _initialized = True
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

    if not public_key or not secret_key:
        logger.info("[Langfuse] Keys not set — LLM tracing disabled")
        return None

    try:
        from langfuse.callback import CallbackHandler

        _handler = CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=os.environ.get("LANGFUSE_HOST", "http://localhost:3002"),
        )
        logger.info("[Langfuse] LLM observability initialised (host=%s)",
                    os.environ.get("LANGFUSE_HOST", "cloud.langfuse.com"))
    except ImportError:
        logger.info("[Langfuse] Package not installed — LLM tracing disabled")
    except Exception as exc:
        logger.warning("[Langfuse] Init failed (non-fatal): %s", exc)

    return _handler
