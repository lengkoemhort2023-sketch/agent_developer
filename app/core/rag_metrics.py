"""
RAG Pipeline Metrics and Tracing

Provides decorators and context managers for tracking RAG operations.
"""

import time
import logging
import asyncio
import inspect
from functools import wraps
from contextlib import contextmanager
from typing import Optional, Any, Dict
from opentelemetry import trace, metrics
from app.core.observability import observability

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


@contextmanager
def track_rag_retrieval(retriever_type: str = "vector"):
    """
    Context manager to track RAG retrieval metrics.
    
    Usage:
        with track_rag_retrieval("vector") as metrics:
            # do retrieval
            metrics["success"] = True
    """
    start_time = time.time()
    context = {"success": False, "error": None}
    
    # Create span for retrieval
    with tracer.start_as_current_span("rag.retrieval") as span:
        span.set_attribute("retriever_type", retriever_type)
        
        try:
            yield context
        except Exception as e:
            context["error"] = str(e)
            span.set_attribute("error", True)
            logger.error(f"Retrieval error: {e}", exc_info=True)
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            
            metrics_obj = observability.metrics
            if metrics_obj:
                try:
                    # Duration histogram
                    metrics_obj["rag_retrieval_duration"].labels(
                        retriever_type=retriever_type
                    ).observe(duration)
                    
                    # Counter
                    status = "success" if context["success"] else "failure"
                    metrics_obj["rag_retrieval_count"].labels(
                        retriever_type=retriever_type,
                        status=status
                    ).inc()
                except Exception as e:
                    logger.warning(f"Error recording retrieval metrics: {e}")
            
            span.set_attribute("duration_ms", duration * 1000)


@contextmanager
def track_rag_llm_call(model_name: str = "default"):
    """
    Context manager to track LLM call metrics.
    
    Usage:
        with track_rag_llm_call("ollama/mistral") as metrics:
            # call LLM
            metrics["tokens_used"] = 150
    """
    start_time = time.time()
    context = {"tokens_used": 0}
    
    with tracer.start_as_current_span("rag.llm") as span:
        span.set_attribute("model_name", model_name)
        
        try:
            yield context
        except Exception as e:
            span.set_attribute("error", True)
            logger.error(f"LLM error: {e}", exc_info=True)
            raise
        finally:
            duration = time.time() - start_time
            
            metrics_obj = observability.metrics
            if metrics_obj:
                try:
                    metrics_obj["rag_llm_duration"].labels(
                        model_name=model_name
                    ).observe(duration)
                except Exception as e:
                    logger.warning(f"Error recording LLM metrics: {e}")
            
            span.set_attribute("duration_ms", duration * 1000)
            span.set_attribute("tokens_used", context.get("tokens_used", 0))


@contextmanager
def track_rag_pipeline(pipeline_name: str = "default"):
    """
    Context manager to track entire RAG pipeline.
    
    Usage:
        with track_rag_pipeline("generative") as metrics:
            # full RAG pipeline
            metrics["retrieval_count"] = 5
            metrics["used_documents"] = 3
    """
    start_time = time.time()
    context = {}
    
    with tracer.start_as_current_span("rag.pipeline") as span:
        span.set_attribute("pipeline_name", pipeline_name)
        
        try:
            yield context
        except Exception as e:
            span.set_attribute("error", True)
            span.record_exception(e)
            raise
        finally:
            duration = time.time() - start_time

            metrics_obj = observability.metrics
            if metrics_obj:
                try:
                    metrics_obj["rag_total_duration"].observe(duration)
                except Exception as e:
                    logger.warning(f"Error recording pipeline metrics: {e}")

            logger.info(
                "RAG pipeline completed",
                extra={
                    "pipeline": pipeline_name,
                    "duration_ms": duration * 1000,
                    **context
                }
            )

            span.set_attribute("duration_ms", duration * 1000)


def track_rag_operation(operation_type: str):
    """
    Decorator for RAG operations.
    
    Usage:
        @track_rag_operation("query_expansion")
        def expand_query(query: str) -> str:
            # implementation
            pass
    """
    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(f"rag.{operation_type}") as span:
                    span.set_attribute("operation", operation_type)
                    start_time = time.time()
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        span.set_attribute("error", True)
                        span.record_exception(e)
                        logger.error(f"Operation {operation_type} failed: {e}", exc_info=True)
                        raise
                    finally:
                        span.set_attribute("duration_ms", (time.time() - start_time) * 1000)
            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(f"rag.{operation_type}") as span:
                span.set_attribute("operation", operation_type)
                start_time = time.time()
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    span.set_attribute("error", True)
                    span.record_exception(e)
                    logger.error(f"Operation {operation_type} failed: {e}", exc_info=True)
                    raise
                finally:
                    duration = time.time() - start_time
                    span.set_attribute("duration_ms", duration * 1000)
                    logger.debug(f"Operation {operation_type} completed in {duration*1000:.2f}ms")

        return sync_wrapper
    return decorator


def track_database_query(table: str, operation: str):
    """
    Decorator for database queries.
    
    Usage:
        @track_database_query("chat_messages", "insert")
        def save_message(message: dict):
            # implementation
            pass
    """
    def _record(span, duration):
        span.set_attribute("duration_ms", duration * 1000)
        metrics_obj = observability.metrics
        if metrics_obj:
            try:
                metrics_obj["db_query_duration"].labels(
                    table=table, operation=operation
                ).observe(duration)
            except Exception as exc:
                logger.warning(f"Error recording DB metrics: {exc}")

    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span("db.query") as span:
                    span.set_attribute("table", table)
                    span.set_attribute("operation", operation)
                    start_time = time.time()
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        span.set_attribute("error", True)
                        span.record_exception(e)
                        raise
                    finally:
                        _record(span, time.time() - start_time)
            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span("db.query") as span:
                span.set_attribute("table", table)
                span.set_attribute("operation", operation)
                start_time = time.time()
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    span.set_attribute("error", True)
                    span.record_exception(e)
                    raise
                finally:
                    _record(span, time.time() - start_time)

        return sync_wrapper
    return decorator
