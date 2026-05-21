from django.core.management.base import BaseCommand
import secrets

class Command(BaseCommand):
    help = "Generate a Django secret key"

    def handle(self, *args, **options):
        # Generate a secure secret key
        secret_key = secrets.token_urlsafe(50)
        self.stdout.write("Generated Django Secret Key:")
        self.stdout.write(secret_key)
        self.stdout.write("\nAdd this to your .env file (.env.dev or .env.prod) as:")
        self.stdout.write(f"SECRET_KEY={secret_key}")







