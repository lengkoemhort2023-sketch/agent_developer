"""
Management command to pre-convert all DOCX documents to PDF for faster viewing.

Usage:
    python manage.py preconvert_pdfs
    python manage.py preconvert_pdfs --skip-existing  # Skip if PDF already exists
"""

from django.core.management.base import BaseCommand
from django.db.models import Q
from document.models import Document
from document.utils import convert_to_pdf, get_pdf_version_path
import os
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Pre-convert all DOCX documents to PDF for faster viewing'

    def add_arguments(self, parser):
        parser.add_argument(
            '--skip-existing',
            action='store_true',
            help='Skip documents that already have a cached PDF',
        )

    def handle(self, *args, **options):
        office_formats = ['docx', 'xlsx', 'pptx', 'odt']
        skip_existing = options.get('skip_existing', False)

        # Get all Office format documents
        documents = Document.objects.filter(
            file_format__in=office_formats,
            is_active=True
        ).order_by('-created_at')

        if not documents.exists():
            self.stdout.write(self.style.WARNING('No office documents found to convert'))
            return

        total = documents.count()
        converted = 0
        skipped = 0
        failed = 0

        self.stdout.write(f'\n{"="*70}')
        self.stdout.write(f'Pre-converting {total} documents to PDF')
        self.stdout.write(f'{"="*70}\n')

        for idx, document in enumerate(documents, 1):
            try:
                pdf_path = get_pdf_version_path(document)

                if not pdf_path:
                    self.stdout.write(self.style.ERROR(
                        f'[{idx}/{total}] ✗ {document.original_filename} - Cannot determine PDF path'
                    ))
                    failed += 1
                    continue

                # Check if already exists
                if os.path.exists(pdf_path):
                    if skip_existing:
                        self.stdout.write(self.style.SUCCESS(
                            f'[{idx}/{total}] ⊘ {document.original_filename} - Already cached'
                        ))
                        skipped += 1
                        continue
                    else:
                        # Re-convert even if exists
                        self.stdout.write(self.style.WARNING(
                            f'[{idx}/{total}] ◐ {document.original_filename} - Re-converting...'
                        ))
                else:
                    self.stdout.write(f'[{idx}/{total}] ◐ {document.original_filename} - Converting...')

                # Perform conversion
                convert_to_pdf(document.full_file_path, pdf_path)

                # Verify PDF was created
                if os.path.exists(pdf_path):
                    file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Convert to MB
                    self.stdout.write(self.style.SUCCESS(
                        f'[{idx}/{total}] ✓ {document.original_filename} ({file_size:.1f}MB)'
                    ))
                    converted += 1
                else:
                    self.stdout.write(self.style.ERROR(
                        f'[{idx}/{total}] ✗ {document.original_filename} - PDF not created'
                    ))
                    failed += 1

            except Exception as e:
                self.stdout.write(self.style.ERROR(
                    f'[{idx}/{total}] ✗ {document.original_filename} - Error: {str(e)[:60]}'
                ))
                logger.error(f'Failed to convert {document.original_filename}: {str(e)}')
                failed += 1

        # Summary
        self.stdout.write(f'\n{"="*70}')
        self.stdout.write(self.style.SUCCESS(f'Converted: {converted}'))
        if skipped:
            self.stdout.write(self.style.WARNING(f'Skipped: {skipped}'))
        if failed:
            self.stdout.write(self.style.ERROR(f'Failed: {failed}'))
        self.stdout.write(f'{"="*70}\n')
