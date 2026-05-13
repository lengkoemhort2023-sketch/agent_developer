from django.core.management.base import BaseCommand
from document.models import Document
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Permanently delete all archived (inactive) documents from the database, file system, and vector store.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--confirm',
            action='store_true',
            help='Confirm the destructive operation',
        )

    def handle(self, *args, **options):
        if not options['confirm']:
            self.stdout.write(
                self.style.ERROR(
                    'DANGER: This will permanently delete ALL archived documents!\n'
                    'This includes all files, database records, and vector store data.\n'
                    'Use --confirm to proceed with the deletion.'
                )
            )
            return

        # Get all inactive documents
        inactive_documents = Document.objects.filter(is_active=False)
        count = inactive_documents.count()

        if count == 0:
            self.stdout.write(self.style.SUCCESS('No archived documents found.'))
            return

        self.stdout.write(self.style.WARNING(f'Found {count} archived documents to delete.'))

        deleted_count = 0
        for doc in inactive_documents:
            try:
                doc.delete()  # This handles file deletion, RAG cleanup, and DB removal
                deleted_count += 1
                if deleted_count % 10 == 0:
                    self.stdout.write(f'Deleted {deleted_count}/{count} documents...')
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error deleting document {doc.id}: {e}')
                )
                logger.error(f'Error deleting document {doc.id}: {e}')

        self.stdout.write(
            self.style.SUCCESS(f'Successfully deleted {deleted_count} archived documents.')
        )






