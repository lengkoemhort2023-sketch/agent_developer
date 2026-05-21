import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.core import serializers
from django.apps import apps


class Command(BaseCommand):
    help = 'Create database and media files backup'

    def add_arguments(self, parser):
        parser.add_argument(
            '--output-dir',
            type=str,
            help='Output directory for backup files',
            default='backups'
        )
        parser.add_argument(
            '--exclude-media',
            action='store_true',
            help='Exclude media files from backup'
        )
        parser.add_argument(
            '--compress',
            action='store_true',
            help='Compress the backup'
        )

    def handle(self, *args, **options):
        output_dir = Path(options['output_dir'])
        exclude_media = options['exclude_media']
        compress = options['compress']

        # Create timestamp for backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = output_dir / f"backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        self.stdout.write(f"Starting backup to: {backup_dir}")

        try:
            # Backup database
            self.backup_database(backup_dir)

            # Backup media files
            if not exclude_media:
                self.backup_media(backup_dir)

            # Create backup metadata
            self.create_metadata(backup_dir, timestamp)

            # Compress if requested
            if compress:
                self.compress_backup(backup_dir)

            self.stdout.write(
                self.style.SUCCESS(f"Backup completed successfully: {backup_dir}")
            )

        except Exception as e:
            # Cleanup on failure
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            raise CommandError(f"Backup failed: {e}")

    def backup_database(self, backup_dir):
        """Backup database data to JSON files."""
        self.stdout.write("Backing up database...")

        db_backup_dir = backup_dir / "database"
        db_backup_dir.mkdir(exist_ok=True)

        # Get all models
        models_to_backup = [
            'user.User',
            'chat.ChatSession',
            'chat.ChatMessage',
            'chat.ChatInput',
            'document.Document',
            'department.Department',
            'docs_type.DocumentType',
        ]

        for model_path in models_to_backup:
            try:
                app_label, model_name = model_path.split('.')
                model = apps.get_model(app_label, model_name)
                queryset = model.objects.all()

                if queryset.exists():
                    filename = f"{model_path.replace('.', '_')}.json"
                    filepath = db_backup_dir / filename

                    with open(filepath, 'w', encoding='utf-8') as f:
                        serializers.serialize('json', queryset, stream=f, indent=2)

                    self.stdout.write(f"  Backed up {queryset.count()} {model_path} records")

            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(f"  Failed to backup {model_path}: {e}")
                )

    def backup_media(self, backup_dir):
        """Backup media files."""
        self.stdout.write("Backing up media files...")

        media_backup_dir = backup_dir / "media"
        media_root = Path(settings.MEDIA_ROOT)

        if media_root.exists():
            # Copy media directory
            shutil.copytree(media_root, media_backup_dir, dirs_exist_ok=True)
            self.stdout.write(f"  Media files backed up to {media_backup_dir}")
        else:
            self.stdout.write("  No media directory found")

    def create_metadata(self, backup_dir, timestamp):
        """Create backup metadata file."""
        metadata = {
            'timestamp': timestamp,
            'created_at': datetime.now().isoformat(),
            'django_version': '5.2.7',
            'settings': {
                'debug': settings.DEBUG,
                'database_engine': settings.DATABASES['default']['ENGINE'],
                'media_root': str(settings.MEDIA_ROOT),
                'static_root': str(settings.STATIC_ROOT),
            },
            'apps_backed_up': [
                'user', 'chat', 'document', 'department', 'docs_type'
            ]
        }

        metadata_file = backup_dir / "backup_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def compress_backup(self, backup_dir):
        """Compress the backup directory."""
        import tarfile

        self.stdout.write("Compressing backup...")

        tar_path = backup_dir.parent / f"{backup_dir.name}.tar.gz"

        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(backup_dir, arcname=backup_dir.name)

        # Remove uncompressed directory
        shutil.rmtree(backup_dir)

        self.stdout.write(f"Backup compressed: {tar_path}")
