"""
Query Queue Manager for Agentic RAG.

Provides a queue system to handle multiple user queries with:
- Priority-based ordering
- Worker thread pool for parallel processing
- Job status tracking
- Result polling via job IDs
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue, Empty
from typing import Any, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future
import queue

from .config import (
    QUEUE_MAX_WORKERS,
    QUEUE_TIMEOUT_SECONDS,
    QUEUE_MAX_QUEUE_SIZE,
)

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(order=True)
class QueueJob:
    priority: int
    job_id: str = field(compare=False)
    question: str = field(compare=False)
    file_type: Optional[str] = field(compare=False)
    query_language: Optional[str] = field(compare=False)
    user_token: Optional[str] = field(compare=False)
    session_id: Optional[str] = field(compare=False)
    doc_id: Optional[str] = field(compare=False)
    language: str = field(compare=False, default="en")
    created_at: float = field(compare=False, default_factory=time.time)
    started_at: Optional[float] = field(compare=False, default=None)
    completed_at: Optional[float] = field(compare=False, default=None)
    status: JobStatus = field(compare=False, default=JobStatus.PENDING)
    result: Any = field(compare=False, default=None)
    error: Optional[str] = field(compare=False, default=None)
    callback: Optional[Callable] = field(compare=False, default=None)


class QueryQueueManager:
    """
    Thread-safe queue manager for processing RAG queries.
    
    Usage:
        queue_manager = QueryQueueManager()
        
        # Submit a query
        job_id = queue_manager.enqueue(
            question="What is the policy on...",
            file_type="policy",
            priority=5
        )
        
        # Poll for result
        job = queue_manager.get_job_status(job_id)
        if job.status == JobStatus.COMPLETED:
            result = job.result
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._job_store: Dict[str, QueueJob] = {}
        self._job_lock = threading.RLock()
        
        self._task_queue: PriorityQueue = PriorityQueue(maxsize=QUEUE_MAX_QUEUE_SIZE)
        
        self._executor = ThreadPoolExecutor(
            max_workers=QUEUE_MAX_WORKERS,
            thread_name_prefix="rag_worker_"
        )
        
        self._shutdown_event = threading.Event()
        self._workers_started = False
        
        self._initialized = True
        logger.info(
            f"[QueryQueueManager] Initialized with {QUEUE_MAX_WORKERS} workers, "
            f"max queue size={QUEUE_MAX_QUEUE_SIZE}"
        )

    def _start_workers(self):
        """Start background worker threads."""
        if self._workers_started:
            return
            
        for i in range(QUEUE_MAX_WORKERS):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"rag_queue_worker_{i}",
                daemon=True
            )
            thread.start()
        
        self._workers_started = True
        logger.info(f"[QueryQueueManager] Started {QUEUE_MAX_WORKERS} worker threads")

    def _worker_loop(self):
        """Main worker loop that processes jobs from the queue."""
        while not self._shutdown_event.is_set():
            try:
                job = self._task_queue.get(timeout=1.0)
                self._process_job(job)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"[QueryQueueManager] Worker error: {e}")

    def _process_job(self, job: QueueJob):
        """Process a single job."""
        job.status = JobStatus.PROCESSING
        job.started_at = time.time()
        
        logger.info(
            f"[QueryQueueManager] Processing job {job.job_id}: "
            f"'{job.question[:50]}...'"
        )
        
        try:
            result = self._execute_query(job)
            job.result = result
            job.status = JobStatus.COMPLETED
            logger.info(f"[QueryQueueManager] Job {job.job_id} completed successfully")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            logger.error(f"[QueryQueueManager] Job {job.job_id} failed: {e}")
            
        finally:
            job.completed_at = time.time()
            
            if job.callback:
                try:
                    job.callback(job)
                except Exception as e:
                    logger.error(f"[QueryQueueManager] Callback error for {job.job_id}: {e}")

    def _execute_query(self, job: QueueJob) -> Dict[str, Any]:
        """
        Execute the actual RAG query.
        This is where the RAG pipeline is invoked.
        """
        from .services.generative import ResponseGenerationService
        from .services.vector_store import VectorStoreService
        
        service = ResponseGenerationService()
        
        result = service.response(
            question=job.question,
            file_type=job.file_type,
            user_token=job.user_token,
            session_id=job.session_id,
            doc_id=job.doc_id,
        )
        
        return result

    def enqueue(
        self,
        question: str,
        file_type: Optional[str] = None,
        query_language: Optional[str] = None,
        user_token: Optional[str] = None,
        session_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        language: str = "en",
        priority: int = 5,
        callback: Optional[Callable] = None,
    ) -> str:
        """
        Add a query to the processing queue.
        
        Args:
            question: User's question
            file_type: Optional document type filter
            query_language: Optional language filter ("km", "en", None)
            user_token: User authentication token
            session_id: Chat session ID
            doc_id: Specific document ID to search
            language: Detected language code
            priority: Lower number = higher priority (1 = highest)
            callback: Optional async callback when job completes
            
        Returns:
            job_id: String that can be used to poll for results
        """
        self._start_workers()
        
        job_id = str(uuid.uuid4())
        
        job = QueueJob(
            priority=priority,
            job_id=job_id,
            question=question,
            file_type=file_type,
            query_language=query_language,
            user_token=user_token,
            session_id=session_id,
            doc_id=doc_id,
            language=language,
            callback=callback,
        )
        
        with self._job_lock:
            self._job_store[job_id] = job
        
        try:
            self._task_queue.put(job, timeout=QUEUE_TIMEOUT_SECONDS)
            logger.info(
                f"[QueryQueueManager] Enqueued job {job_id} "
                f"(priority={priority}, queue_size={self._task_queue.qsize()})"
            )
        except queue.Full:
            job.status = JobStatus.FAILED
            job.error = "Queue is full, please try again later"
            logger.warning(f"[QueryQueueManager] Queue full, rejected job {job_id}")
            
        return job_id

    def get_job_status(self, job_id: str) -> Optional[QueueJob]:
        """
        Get the current status of a job.
        
        Returns:
            QueueJob object if found, None otherwise
        """
        with self._job_lock:
            return self._job_store.get(job_id)

    def get_job_result(self, job_id: str, timeout: float = 0) -> Optional[Any]:
        """
        Get job result, optionally blocking until completion.
        
        Args:
            job_id: The job ID returned from enqueue()
            timeout: Maximum time to wait in seconds (0 = return immediately)
            
        Returns:
            Job result if completed, None otherwise
        """
        start_time = time.time()
        
        while True:
            job = self.get_job_status(job_id)
            if job is None:
                return None
                
            if job.status == JobStatus.COMPLETED:
                return job.result
                
            if job.status == JobStatus.FAILED:
                raise Exception(f"Job failed: {job.error}")
                
            if timeout <= 0:
                return None
                
            if time.time() - start_time >= timeout:
                return None
                
            time.sleep(0.1)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending job (cannot cancel processing/completed jobs).
        
        Returns:
            True if cancelled, False if job not found or already processing/completed
        """
        with self._job_lock:
            job = self._job_store.get(job_id)
            if job is None:
                return False
                
            if job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                logger.info(f"[QueryQueueManager] Cancelled job {job_id}")
                return True
                
            return False

    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get current queue statistics.
        
        Returns:
            Dict with queue stats
        """
        with self._job_lock:
            pending = sum(1 for j in self._job_store.values() if j.status == JobStatus.PENDING)
            processing = sum(1 for j in self._job_store.values() if j.status == JobStatus.PROCESSING)
            completed = sum(1 for j in self._job_store.values() if j.status == JobStatus.COMPLETED)
            failed = sum(1 for j in self._job_store.values() if j.status == JobStatus.FAILED)
            
        return {
            "queue_size": self._task_queue.qsize(),
            "pending_jobs": pending,
            "processing_jobs": processing,
            "completed_jobs": completed,
            "failed_jobs": failed,
            "total_jobs": len(self._job_store),
            "max_workers": QUEUE_MAX_WORKERS,
        }

    def get_pending_jobs(self, limit: int = 50) -> list:
        """Get list of pending jobs."""
        with self._job_lock:
            jobs = [
                {
                    "job_id": j.job_id,
                    "question": j.question[:100],
                    "priority": j.priority,
                    "created_at": j.created_at,
                    "status": j.status.value,
                }
                for j in sorted(
                    self._job_store.values(),
                    key=lambda x: (x.priority, x.created_at)
                )[:limit]
            ]
        return jobs

    def shutdown(self, wait: bool = True):
        """
        Shutdown the queue manager.
        
        Args:
            wait: If True, wait for pending jobs to complete
        """
        logger.info("[QueryQueueManager] Shutting down...")
        self._shutdown_event.set()
        
        if wait:
            time.sleep(2)
            
        self._executor.shutdown(wait=wait)
        logger.info("[QueryQueueManager] Shutdown complete")


_query_queue_manager: Optional[QueryQueueManager] = None


def get_query_queue() -> QueryQueueManager:
    """Get the singleton QueryQueueManager instance."""
    global _query_queue_manager
    if _query_queue_manager is None:
        _query_queue_manager = QueryQueueManager()
    return _query_queue_manager


def enqueue_query(
    question: str,
    file_type: Optional[str] = None,
    query_language: Optional[str] = None,
    user_token: Optional[str] = None,
    session_id: Optional[str] = None,
    doc_id: Optional[str] = None,
    language: str = "en",
    priority: int = 5,
) -> str:
    """
    Convenience function to enqueue a query.
    
    Returns job_id for polling.
    """
    queue = get_query_queue()
    return queue.enqueue(
        question=question,
        file_type=file_type,
        query_language=query_language,
        user_token=user_token,
        session_id=session_id,
        doc_id=doc_id,
        language=language,
        priority=priority,
    )


def get_query_result(job_id: str, timeout: float = 30.0) -> Optional[Any]:
    """
    Convenience function to get query result with polling.
    
    Args:
        job_id: Job ID from enqueue_query
        timeout: Max seconds to wait
        
    Returns:
        Query result or None
    """
    queue = get_query_queue()
    return queue.get_job_result(job_id, timeout=timeout)
