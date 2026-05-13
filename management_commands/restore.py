import os
import json
import shutil
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.core import serializers
from django.apps import apps
from django.core.management import call_command


class Command(BaseCommand):
    help = 'Restore database and media files from backup'

    def add_arguments(self, parser):
        parser.add_argument(
            'backup_path',
            type=str,
            help='Path to backup directory or compressed file'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be restored without actually doing it'
        )
        parser.add_argument(
            '--skip-media',
            action='store_true',
            help='Skip media files restoration'
        )

    def handle(self, *args, **options):
        backup_path = Path(options['backup_path'])
        dry_run = options['dry_run']
        skip_media = options['skip_media']

        if not backup_path.exists():
            raise CommandError(f"Backup path does not exist: {backup_path}")

        # Handle compressed backups
        if backup_path.suffix == '.gz' and backup_path.suffixes[-2:] == ['.tar', '.gz']:
            backup_path = self.extract_backup(backup_path)

        if not backup_path.is_dir():
            raise CommandError(f"Backup path is not a directory: {backup_path}")

        # Validate backup
        self.validate_backup(backup_path)

        if dry_run:
            self.stdout.write("DRY RUN - No changes will be made")
            self.show_backup_contents(backup_path)
            return

        # Confirm restoration
        if not self.confirm_restoration(backup_path):
            self.stdout.write("Restoration cancelled")
            return

        try:
            # Restore database
            self.restore_database(backup_path)

            # Restore media files
            if not skip_media:
                self.restore_media(backup_path)

            self.stdout.write(
                self.style.SUCCESS("Restoration completed successfully")
            )

        except Exception as e:
            raise CommandError(f"Restoration failed: {e}")

    def extract_backup(self, compressed_path):
        """Extract compressed backup."""
        import tarfile

        self.stdout.write(f"Extracting backup: {compressed_path}")

        extract_dir = compressed_path.parent / compressed_path.stem
        extract_dir.mkdir(exist_ok=True)

        with tarfile.open(compressed_path, "r:gz") as tar:
            tar.extractall(extract_dir)

        # Find the actual backup directory
        backup_dirs = list(extract_dir.glob("backup_*"))
        if not backup_dirs:
            raise CommandError("No backup directory found in extracted archive")

        return backup_dirs[0]

    def validate_backup(self, backup_path):
        """Validate backup structure and metadata."""
        metadata_file = backup_path / "backup_metadata.json"

        if not metadata_file.exists():
            raise CommandError("Backup metadata file not found")

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            required_keys = ['timestamp', 'created_at', 'apps_backed_up']
            for key in required_keys:
                if key not in metadata:
                    raise CommandError(f"Invalid backup metadata: missing {key}")

            self.stdout.write(f"Backup created: {metadata['created_at']}")
            self.stdout.write(f"Apps backed up: {', '.join(metadata['apps_backed_up'])}")

        except json.JSONDecodeError:
            raise CommandError("Invalid backup metadata format")

    def show_backup_contents(self, backup_path):
        """Show what would be restored."""
        self.stdout.write("Backup contents:")

        # Database files
        db_dir = backup_path / "database"
        if db_dir.exists():
            json_files = list(db_dir.glob("*.json"))
            self.stdout.write(f"  Database: {len(json_files)} model files")
            for f in json_files[:3]:  # Show first 3
                self.stdout.write(f"    - {f.name}")

        # Media files
        media_dir = backup_path / "media"
        if media_dir.exists():
            total_files = sum(1 for _ in media_dir.rglob("*") if _.is_file())
            self.stdout.write(f"  Media: {total_files} files")
        else:
            self.stdout.write("  Media: No media files in backup")

    def confirm_restoration(self, backup_path):
        """Ask for confirmation before restoration."""
        self.stdout.write(
            self.style.WARNING("WARNING: This will overwrite existing data!")
        )
        self.stdout.write(f"Restoring from: {backup_path}")

        response = input("Are you sure you want to continue? (yes/no): ")
        return response.lower() in ['yes', 'y']

    def restore_database(self, backup_path):
        """Restore database from backup files."""
        self.stdout.write("Restoring database...")

        db_dir = backup_path / "database"
        if not db_dir.exists():
            self.stdout.write("No database backup found")
            return

        # Clear existing data (optional - be careful!)
        # self.clear_existing_data()

        # Restore each model
        json_files = list(db_dir.glob("*.json"))
        for json_file in json_files:
            try:
                model_name = json_file.stem.replace('_', '.')
                app_label, model_short_name = model_name.rsplit('.', 1)

                self.stdout.write(f"  Restoring {model_name}...")

                with open(json_file, 'r', encoding='utf-8') as f:
                    objects = serializers.deserialize('json', f)
                    for obj in objects:
                        obj.save()

                self.stdout.write(f"    Restored {model_name}")

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"    Failed to restore {json_file.name}: {e}")
                )

    def restore_media(self, backup_path):
        """Restore media files."""
        self.stdout.write("Restoring media files...")

        media_dir = backup_path / "media"
        if not media_dir.exists():
            self.stdout.write("No media files in backup")
            return

        media_root = Path(settings.MEDIA_ROOT)

        # Clear existing media (optional)
        if media_root.exists():
            shutil.rmtree(media_root)

        # Copy media files
        shutil.copytree(media_dir, media_root)
        self.stdout.write(f"Media files restored to {media_root}")

    def clear_existing_data(self):
        """Clear existing data before restoration."""
        self.stdout.write("Clearing existing data...")

        models_to_clear = [
            'chat.ChatInput',
            'chat.ChatMessage',
            'chat.ChatSession',
            'document.Document',
            'user.User',
            'department.Department',
            'docs_type.DocumentType',
        ]

        for model_path in models_to_clear:
            try:
                app_label, model_name = model_path.split('.')
                model = apps.get_model(app_label, model_name)
                count = model.objects.count()
                model.objects.all().delete()
                self.stdout.write(f"  Cleared {count} {model_path} records")
            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(f"  Failed to clear {model_path}: {e}")
                )
