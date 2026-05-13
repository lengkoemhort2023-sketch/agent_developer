from dataclasses import dataclass
from typing import Optional

from document.models import Document, DocumentIndexingJob


@dataclass
class EnqueuedDocumentIndexing:
    document: Document
    job: DocumentIndexingJob


def create_document_indexing_job(document: Document) -> DocumentIndexingJob:
    return DocumentIndexingJob.objects.create(
        document=document,
        status=DocumentIndexingJob.STATUS_QUEUED,
        stage=DocumentIndexingJob.STAGE_QUEUED,
        source_path=document.full_file_path or "",
    )


def enqueue_document_indexing(document: Document) -> EnqueuedDocumentIndexing:
    from document.tasks import extract_document_indexing_job

    job = create_document_indexing_job(document)
    extract_document_indexing_job.delay(str(job.id))
    return EnqueuedDocumentIndexing(document=document, job=job)


def get_latest_indexing_job(document: Document) -> Optional[DocumentIndexingJob]:
    return document.indexing_jobs.order_by("-created_at").first()
