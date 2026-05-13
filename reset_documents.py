#!/usr/bin/env python3
"""
Reset all documents from the Django database and Qdrant collections.

When launched from a host Python environment that cannot connect to the
containerized services, this script automatically retries inside the Docker
`web` container.
"""

from __future__ import annotations

import argparse
import os
import traceback

from app.core.env import load_environment
from reset_support import run_script_in_docker, should_retry_in_docker


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Reset documents and vectors.")
    parser.add_argument(
        "--django-only",
        action="store_true",
        help="clear only Django document records",
    )
    parser.add_argument(
        "--qdrant-only",
        action="store_true",
        help="clear only Qdrant collections",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation prompt",
    )
    parser.add_argument(
        "--in-container",
        action="store_true",
        help="internal flag used by Docker fallback",
    )
    parser.add_argument(
        "--no-docker-fallback",
        action="store_true",
        help="do not retry inside the Docker web container on host-side failures",
    )
    return parser.parse_args()


def maybe_retry_in_docker(args, error, extra_args=None):
    """Retry the current operation inside Docker when the host environment cannot."""

    if args.in_container or args.no_docker_fallback or not should_retry_in_docker(error):
        return None
    return run_script_in_docker("reset_documents.py", extra_args=extra_args)


def reset_django_documents(args) -> bool:
    """Reset documents using the Django ORM."""

    try:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.core.settings")

        import django

        django.setup()

        from document.models import Document

        count = Document.objects.count()

        if count == 0:
            print("No documents found in Django database.")
            return True

        print(f"Found {count} documents in Django database.")

        Document.objects.all().delete()

        remaining = Document.objects.count()
        if remaining == 0:
            print(f"Deleted {count} documents from Django database.")
            return True

        print(f"{remaining} documents still remain.")
        return False

    except ImportError as error:
        print(f"Django not available: {error}")
        print("Make sure you are running this from the agent_developer directory with Django installed.")
        docker_result = maybe_retry_in_docker(args, error, extra_args=["--django-only"])
        return docker_result if docker_result is not None else False
    except Exception as error:
        print(f"Error deleting documents: {error}")
        docker_result = maybe_retry_in_docker(args, error, extra_args=["--django-only"])
        if docker_result is not None:
            return docker_result
        traceback.print_exc()
        return False


def build_qdrant_urls(args):
    """Build candidate Qdrant URLs for host-side and container-side execution."""

    load_environment()

    host = os.environ.get("QDRANT_HOST", "")
    port = os.environ.get("QDRANT_PORT", "")
    candidates = [f"http://{host}:{port}"]

    if not args.in_container and host not in {"localhost", "127.0.0.1"}:
        candidates.append(f"http://localhost:{port}")
        candidates.append(f"http://127.0.0.1:{port}")

    deduped = []
    for url in candidates:
        if url not in deduped:
            deduped.append(url)
    return deduped


def get_qdrant_client(args) -> QdrantClient:
    """Return a connected Qdrant client using the first reachable URL."""

    from qdrant_client import QdrantClient

    last_error = None
    for url in build_qdrant_urls(args):
        try:
            client = QdrantClient(url=url)
            client.get_collections()
            print(f"Connected to Qdrant at {url}")
            return client
        except Exception as error:
            last_error = error

    if last_error is None:
        raise RuntimeError("No Qdrant URL candidates were generated.")
    raise last_error


def reset_qdrant_collections(args) -> bool:
    """Reset all Qdrant collections."""

    try:
        client = get_qdrant_client(args)

        collections = client.get_collections()
        collection_names = [collection.name for collection in collections.collections]

        if not collection_names:
            print("No collections found in Qdrant.")
            return True

        print(f"Found {len(collection_names)} collections in Qdrant:")
        for name in collection_names:
            print(f"  - {name}")

        deleted = 0
        for name in collection_names:
            try:
                client.delete_collection(name)
                print(f"Deleted collection: {name}")
                deleted += 1
            except Exception as error:
                print(f"Error deleting {name}: {error}")

        print(f"Deleted {deleted}/{len(collection_names)} Qdrant collections.")
        return deleted == len(collection_names)

    except ImportError as error:
        print(f"qdrant-client is not available: {error}")
        docker_result = maybe_retry_in_docker(args, error, extra_args=["--qdrant-only"])
        return docker_result if docker_result is not None else False
    except Exception as error:
        print(f"Error connecting to Qdrant: {error}")
        docker_result = maybe_retry_in_docker(args, error, extra_args=["--qdrant-only"])
        if docker_result is not None:
            return docker_result
        traceback.print_exc()
        return False


def main():
    """Run the document reset flow."""

    args = parse_args()

    print("=" * 60)
    print("DOCUMENT SYSTEM RESET TOOL")
    print("=" * 60)
    print()
    print("WARNING: This will delete indexed documents and vector data.")
    print()

    response = "yes" if args.yes else input("Are you sure you want to continue? (yes/no): ").strip().lower()
    if response != "yes":
        print("Operation cancelled.")
        return

    if args.django_only:
        print("Mode: Django documents only")
        success = reset_django_documents(args)
    elif args.qdrant_only:
        print("Mode: Qdrant collections only")
        success = reset_qdrant_collections(args)
    else:
        print("Mode: Full reset (Django + Qdrant)")
        print()
        print("-" * 60)
        print("Step 1: Clearing Django documents...")
        print("-" * 60)
        django_success = reset_django_documents(args)

        print()
        print("-" * 60)
        print("Step 2: Clearing Qdrant collections...")
        print("-" * 60)
        qdrant_success = reset_qdrant_collections(args)
        success = django_success and qdrant_success

    print()
    print("=" * 60)
    if success:
        print("Reset complete.")
    else:
        print("Reset completed with warnings.")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"1. Refresh your frontend ({os.environ.get('FRONTEND_URL', 'http://localhost:3000')}/d)")
    print("2. Upload documents again if needed")
    print("3. Re-run indexing jobs for newly uploaded documents")


if __name__ == "__main__":
    main()
