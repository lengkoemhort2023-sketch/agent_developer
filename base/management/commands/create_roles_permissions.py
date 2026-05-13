from django.core.management.base import BaseCommand

from base.roles import ensure_default_groups

class Command(BaseCommand):
    help = 'Create user roles/groups and assign permissions.'

    def handle(self, *args, **options):
        ensure_default_groups()
        self.stdout.write(self.style.SUCCESS('Roles and permissions setup complete.'))







