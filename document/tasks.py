import os

from celery import shared_task
from django.conf import settings
from django.utils import timezone
from langfuse.decorators import observe

from document.models import Document
from document.utils import (
    activate_document_from_rag,
    deactivate_document_from_rag,
    delete_file_from_rag,
    upload_pdf_to_rag,
)
import logging
logger = logging.getLogger(__name__)


def _mark_job_failed(job, message: str) -> None:
    from document.models import DocumentIndexingJob

    job.status = DocumentIndexingJob.STATUS_FAILED
    job.stage = DocumentIndexingJob.STAGE_FAILED
    job.error_message = message
    job.finished_at = timezone.now()
    job.save(
        update_fields=["status", "stage", "error_message", "finished_at", "updated_at"]
    )


def _is_latest_running_job(job) -> bool:
    from document.models import DocumentIndexingJob

    newer_active_exists = DocumentIndexingJob.objects.filter(
        document=job.document,
        created_at__gt=job.created_at,
        status__in=[
            DocumentIndexingJob.STATUS_QUEUED,
            DocumentIndexingJob.STATUS_RUNNING,
        ],
    ).exists()
    return not newer_active_exists


def _run_vector_processing(full_path: str, document_id: str, document_name: str, doc_type_name: str, doc_type_id: str):
    logger.info(f"Processing upload to vector DB for document ID: {document_id}, Name: {document_name}")
    print(f"Processing upload to vector DB for document ID: {document_id}, Name: {document_name}")

    if not Document.objects.filter(id=document_id).exists():
        logger.error(f"Document {document_id} not found before upload attempt.")
        print(f"Document {document_id} not found before upload attempt.")
        raise Exception(f"Document {document_id} not found before upload attempt.")

    rag_result = upload_pdf_to_rag(
        file_path=full_path,
        file_id=document_id,
        file_name=document_name,
        file_type_name=doc_type_name,
        file_type_id=doc_type_id
    )
    logger.info(f"Upload to vector DB completed for document ID: {document_id}, Result: {rag_result}")
    print(f"Upload to vector DB completed for document ID: {document_id}, Result: {rag_result}")

    if not rag_result.get('success', True):
        logger.error(f"Upload failed for document {document_id}, not updating is_vector_processed.")
        print(f"Upload failed for document {document_id}, not updating is_vector_processed.")
        raise Exception(f"Upload failed: {rag_result.get('error', 'Unknown error')}")

    doc = Document.objects.get(id=document_id)
    doc.is_vector_processed = True
    doc.save(update_fields=['is_vector_processed'])
    logger.info(f"Document {document_id} marked as vector processed.")
    print(f"Document {document_id} marked as vector processed.")
    return rag_result


def _deactivate_superseded_family_vectors(document: Document) -> None:
    """Deactivate searchable chunks for older active versions before indexing the latest."""
    if not document.is_latest_version:
        return

    previous_versions = (
        Document.objects.filter(
            id__in=document.get_family_ids(),
            is_active=True,
            is_vector_processed=True,
        )
        .exclude(id=document.id)
        .select_related("type")
    )

    for previous_document in previous_versions:
        doc_type_name = (
            str(previous_document.type.name)
            if previous_document.type
            else str(document.type.name) if document.type else "unknown"
        )
        logger.info(
            "Deactivating vector chunks for superseded document version %s before indexing %s",
            previous_document.id,
            document.id,
        )
        rag_result = deactivate_document_from_rag(
            file_id=str(previous_document.id),
            file_type=doc_type_name,
        )
        if not rag_result.get("success", True):
            raise Exception(
                f"Failed to deactivate previous version {previous_document.id}: "
                f"{rag_result.get('error', 'Unknown error')}"
            )

@shared_task
@observe(name="celery.process_upload_to_vector_db")
def process_upload_to_vector_db(full_path: str, document_id: str, document_name: str, doc_type_name: str, doc_type_id: str):
    try:
        _run_vector_processing(full_path, document_id, document_name, doc_type_name, doc_type_id)
    except Exception as e:
        logger.error(f"Error during upload to vector DB for document ID: {document_id}: {str(e)}")
        print(f"Error during upload to vector DB for document ID: {document_id}: {str(e)}")
        raise


@shared_task
def process_document_indexing_job(job_id: str):
    """Backward-compatible entrypoint that starts the staged indexing flow."""
    extract_document_indexing_job.delay(job_id)


@shared_task
@observe(name="celery.extract_document_indexing_job")
def extract_document_indexing_job(job_id: str):
    from document.models import DocumentIndexingJob

    job = DocumentIndexingJob.objects.select_related("document", "document__type").get(id=job_id)
    document = job.document

    job.status = DocumentIndexingJob.STATUS_RUNNING
    job.stage = DocumentIndexingJob.STAGE_EXTRACTING
    job.started_at = timezone.now()
    job.attempts += 1
    job.error_message = ""
    job.source_path = document.full_file_path or job.source_path
    job.save(
        update_fields=[
            "status",
            "stage",
            "started_at",
            "attempts",
            "error_message",
            "source_path",
            "updated_at",
        ]
    )

    try:
        if not document.full_file_path:
            raise Exception(f"Document {document.id} has no file path to index.")
        index_document_indexing_job.delay(job_id)
    except Exception as exc:
        job.status = DocumentIndexingJob.STATUS_FAILED
        job.stage = DocumentIndexingJob.STAGE_FAILED
        job.error_message = str(exc)
        job.finished_at = timezone.now()
        job.save(
            update_fields=["status", "stage", "error_message", "finished_at", "updated_at"]
        )
        raise


@shared_task
@observe(name="celery.index_document_indexing_job")
def index_document_indexing_job(job_id: str):
    from document.models import DocumentIndexingJob

    job = DocumentIndexingJob.objects.select_related("document", "document__type").get(id=job_id)
    document = job.document

    if not _is_latest_running_job(job):
        _mark_job_failed(
            job,
            "Skipped stale indexing job because a newer job exists for this document.",
        )
        logger.info(
            "Skipping stale indexing job %s for document %s; newer job is already queued/running.",
            job.id,
            document.id,
        )
        return

    job.status = DocumentIndexingJob.STATUS_RUNNING
    job.stage = DocumentIndexingJob.STAGE_INDEXING
    job.error_message = ""
    job.save(update_fields=["status", "stage", "error_message", "updated_at"])

    try:
        job.stage = DocumentIndexingJob.STAGE_DEACTIVATING_PREVIOUS_VERSIONS
        job.save(update_fields=["stage", "updated_at"])
        _deactivate_superseded_family_vectors(document)
        job.stage = DocumentIndexingJob.STAGE_INDEXING
        job.save(update_fields=["stage", "updated_at"])
        _run_vector_processing(
            full_path=document.full_file_path,
            document_id=str(document.id),
            document_name=str(document.original_filename),
            doc_type_name=str(document.type.name) if document.type else "unknown",
            doc_type_id=str(document.type.id) if document.type else "",
        )
        finalize_document_indexing_job.delay(job_id)
    except Exception as exc:
        _mark_job_failed(job, str(exc))
        raise


@shared_task
@observe(name="celery.finalize_document_indexing_job")
def finalize_document_indexing_job(job_id: str):
    from document.models import DocumentIndexingJob

    job = DocumentIndexingJob.objects.select_related("document").get(id=job_id)
    job.status = DocumentIndexingJob.STATUS_RUNNING
    job.stage = DocumentIndexingJob.STAGE_FINALIZING
    job.save(update_fields=["status", "stage", "updated_at"])

    job.status = DocumentIndexingJob.STATUS_COMPLETED
    job.stage = DocumentIndexingJob.STAGE_COMPLETED
    job.finished_at = timezone.now()
    job.save(update_fields=["status", "stage", "finished_at", "updated_at"])

@shared_task
@observe(name="celery.process_unprocessed_documents")
def process_unprocessed_documents():
    from base.application.document_indexing import create_document_indexing_job
    from document.models import DocumentIndexingJob

    logger.info("Starting batch process for unprocessed documents.")
    print("Starting batch process for unprocessed documents.")
    unprocessed_docs = Document.objects.filter(is_vector_processed=False)
    for doc in unprocessed_docs:
        try:
            has_active_job = DocumentIndexingJob.objects.filter(
                document=doc,
                status__in=[
                    DocumentIndexingJob.STATUS_QUEUED,
                    DocumentIndexingJob.STATUS_RUNNING,
                ],
            ).exists()
            if has_active_job:
                logger.info(
                    "Skipping document %s in periodic indexing; queued/running job already exists.",
                    doc.id,
                )
                continue
            job = create_document_indexing_job(doc)
            extract_document_indexing_job.delay(str(job.id))
            logger.info(f"Triggered processing for document ID: {doc.id}")
            print(f"Triggered processing for document ID: {doc.id}")
        except Exception as e:
            logger.error(f"Error triggering processing for document ID: {doc.id}: {str(e)}")
            print(f"Error triggering processing for document ID: {doc.id}: {str(e)}")
            raise
    logger.info("Batch process for unprocessed documents finished.")
    print("Batch process for unprocessed documents finished.")

@shared_task
@observe(name="celery.deactivate_document")
def deactivate_document(document_id: str, doc_type_name: str = "unknown"):
    """
    Deactivate document a document by deactivating it in the RAG system.

    Args:
        document_id (str): Unique identifier for the document
        doc_type_name (str): Type of the document
    """
    logger.info(f"Processing soft delete for document ID: {document_id}, Type: {doc_type_name}")
    try:
        rag_result = deactivate_document_from_rag(
            file_id=document_id,
            file_type=doc_type_name
        )
        logger.info(f"Soft delete completed for document ID: {document_id}, Result: {rag_result}")
        if not rag_result.get('success', True):
            raise Exception(f"Soft delete failed: {rag_result.get('error', 'Unknown error')}")
        return rag_result
    except Exception as e:
        logger.error(f"Error during soft delete for document ID: {document_id}, Error: {str(e)}")
        raise

@shared_task
@observe(name="celery.activate_document")
def activate_document(document_id: str):
    """
    Activate document by activating chunk in RAG system.

    Args:
        document_id (str) : Unique ID for document
    """

    logger.info(f"Processing activate document for document ID : {document_id} ")
    try:
        # Validate document_id before proceeding
        if not document_id or document_id is None:
            logger.error(f"Invalid document_id provided: {document_id}")
            raise Exception(f"Invalid document_id: {document_id}")

        rag_result = activate_document_from_rag(
            file_id=document_id
        )
        logger.info(f"Activate document completed, Result: {rag_result}")
        
        # Handle missing chunks gracefully - might already be cleaned up
        if not rag_result.get('success', True):
            error_msg = rag_result.get('error', 'Unknown error')
            if 'No chunks found' in error_msg or 'Failed to activate document' in error_msg:
                logger.warning(f"Document activation skipped - chunks not found in inactive collection (already cleaned up): {document_id}")
                # Return success since document is already in consistent state
                return {'success': True, 'message': 'Document already activated or chunks cleaned up'}
            else:
                raise Exception(f"Activate document failed: {error_msg}")
        return rag_result
    except Exception as e:
        error_str = str(e)
        # Don't treat missing chunks as critical error
        if 'No chunks found' in error_str or 'already activated' in error_str:
            logger.warning(f"Document activation completed with warnings: {error_str}")
            return {'success': True, 'message': 'Activation completed with warnings'}
        logger.error(f"Error during activate document, Error: {error_str}")
        raise


@shared_task
@observe(name="celery.cleanup_deleted_document_artifacts")
def cleanup_deleted_document_artifacts(document_id: str, file_path: str = "", doc_type_name: str = "unknown"):
    """Remove document artifacts after the DB row has already been deleted."""
    logger.info(
        "Cleaning up deleted document artifacts for document ID: %s, file_path: %s, type: %s",
        document_id,
        file_path,
        doc_type_name,
    )

    try:
        if file_path:
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)
            if os.path.exists(full_path):
                os.remove(full_path)
                logger.info("Deleted document file from disk: %s", full_path)
    except Exception as exc:
        logger.warning("Failed to delete file for document %s: %s", document_id, exc)

    try:
        rag_result = delete_file_from_rag(document_id, doc_type_name)
        logger.info("Deleted RAG artifacts for document %s: %s", document_id, rag_result)
        return rag_result
    except Exception as exc:
        logger.error("Failed to delete RAG artifacts for document %s: %s", document_id, exc)
        raise
