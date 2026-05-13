from django.core.management.base import BaseCommand
from document.models import Document
from base.services.agent.rag.providers import RagProviders
from qdrant_client import QdrantClient
from qdrant_client.http import models
from decouple import config
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Cleans up the vector store by removing references to non-existent documents. Use --reset to drop all Qdrant data.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--reset',
            action='store_true',
            help='Drop ALL Qdrant data (equivalent to flush for vector DB)',
        )
        parser.add_argument(
            '--confirm',
            action='store_true',
            help='Confirm destructive operations',
        )

    def handle(self, *args, **options):
        if options['reset']:
            return self.handle_reset(options)

        self.stdout.write(self.style.SUCCESS('Starting vector store cleanup...'))
        rag_provider = RagProviders()

        # Get all document IDs from the main database
        db_document_ids = set(str(doc_id) for doc_id in Document.objects.values_list('id', flat=True))
        self.stdout.write(f"Found {len(db_document_ids)} documents in the main database.")

        # Get all collections from Qdrant
        qdrant_collections = rag_provider.vector_store.client.get_collections()
        self.stdout.write(f"Found {len(qdrant_collections.collections)} collections in Qdrant.")

        cleaned_up_count = 0

        for collection_info in qdrant_collections.collections:
            collection_name = collection_info.name
            self.stdout.write(f"Processing collection: {collection_name}")

            # Skip history collections as they are handled separately or not directly linked to documents
            if collection_name.startswith("history"):
                self.stdout.write(f"Skipping history collection: {collection_name}")
                continue

            # Fetch all items (chunks) from the current collection
            # This might be memory intensive for very large collections
            try:
                # Scroll through all points in the collection to get payloads
                scroll_result = rag_provider.vector_store.client.scroll(
                    collection_name=collection_name,
                    limit=10000  # Adjust if needed for larger collections
                )
                points = scroll_result[0]

                # Extract unique document IDs from the chunks in this collection
                qdrant_doc_ids_in_collection = set()
                for point in points:
                    # Assuming 'source' in payload stores the original document ID
                    source_id = point.payload.get('source')
                    if source_id:
                        qdrant_doc_ids_in_collection.add(source_id)

                self.stdout.write(f"Collection {collection_name} contains references to {len(qdrant_doc_ids_in_collection)} unique documents.")

                # Identify document IDs in Qdrant that are not in the main database
                orphaned_qdrant_doc_ids = qdrant_doc_ids_in_collection - db_document_ids

                if orphaned_qdrant_doc_ids:
                    self.stdout.write(self.style.WARNING(f"Found {len(orphaned_qdrant_doc_ids)} orphaned document references in collection {collection_name}."))
                    for orphaned_doc_id in orphaned_qdrant_doc_ids:
                        self.stdout.write(f"Attempting to delete orphaned document ID: {orphaned_doc_id} from collection {collection_name}...")
                        # Delete all chunks associated with this orphaned_doc_id from the collection
                        try:
                            rag_provider.vector_store.client.delete(
                                collection_name=collection_name,
                                points_selector=models.Filter(
                                    must=[
                                        models.FieldCondition(
                                            key="source",
                                            match=models.MatchValue(value=orphaned_doc_id)
                                        )
                                    ]
                                )
                            )
                            self.stdout.write(self.style.SUCCESS(f"Successfully deleted orphaned document ID: {orphaned_doc_id} from collection {collection_name}."))
                            cleaned_up_count += 1
                        except Exception as delete_e:
                            self.stdout.write(self.style.ERROR(f"Failed to delete orphaned document ID: {orphaned_doc_id} from collection {collection_name}: {delete_e}"))
                else:
                    self.stdout.write(f"No orphaned document references found in collection {collection_name}.")

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error processing collection {collection_name}: {e}"))
                logger.error(f"Error processing collection {collection_name}: {e}")

        self.stdout.write(self.style.SUCCESS(f'Vector store cleanup finished. Cleaned up {cleaned_up_count} orphaned document references.'))

    def handle_reset(self, options):
        """Handle the --reset option to drop all Qdrant data"""
        if not options.get('confirm'):
            self.stdout.write(
                self.style.ERROR(
                    'DANGER: This will delete ALL vector data from Qdrant!\n'
                    'This is equivalent to "python manage.py flush" for the vector database.\n'
                    'Use --confirm to proceed with the reset.'
                )
            )
            return

        try:
            # Get Qdrant connection settings
            qdrant_host = config("QDRANT_HOST", default="")
            qdrant_port = config("QDRANT_PORT", default="")

            self.stdout.write(self.style.WARNING(f'Connecting to Qdrant at {qdrant_host}:{qdrant_port}...'))

            # Create Qdrant client with reset permission
            client = QdrantClient(
                host=qdrant_host,
                port=int(qdrant_port)
            )

            # Test connection
            collections = client.get_collections()
            self.stdout.write(f'Connected to Qdrant (found {len(collections.collections)} collections)')

            # List collections before reset
            collections_before = client.get_collections()
            self.stdout.write(f'Found {len(collections_before.collections)} collections before reset')

            # Perform the reset
            self.stdout.write(self.style.WARNING('Resetting Qdrant database...'))
            # Note: QdrantClient may not have a direct reset method; using delete_collection for all
            for col in collections_before.collections:
                client.delete_collection(col.name)
            self.stdout.write('Deleted all collections')

            # Verify reset
            collections_after = client.get_collections()
            self.stdout.write(f'Collections after reset: {len(collections_after.collections)}')

            self.stdout.write(
                self.style.SUCCESS(
                    'Qdrant reset completed successfully!\n'
                    'All vector data has been permanently deleted.\n'
                    'You will need to re-upload and re-process all documents.'
                )
            )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error during Qdrant reset: {str(e)}')
            )
            logger.error(f'Qdrant reset error: {str(e)}', exc_info=True)







