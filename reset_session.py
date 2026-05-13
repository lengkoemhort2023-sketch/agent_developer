#!/usr/bin/env python3
"""
Reset all chat sessions from the Django database.

When launched from a host Python environment that cannot connect to the
containerized PostgreSQL service, this script automatically retries inside the
Docker `web` container.
"""

from __future__ import annotations

import argparse
import os
import traceback

from reset_support import run_script_in_docker, should_retry_in_docker


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Delete all chat sessions.")
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


def reset_chat_sessions(args) -> bool:
    """Reset all chat sessions using the Django ORM."""

    try:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.core.settings")

        import django

        django.setup()

        from chat.models import ChatInput, ChatMessage, ChatSession

        session_count = ChatSession.objects.count()
        message_count = ChatMessage.objects.count()
        input_count = ChatInput.objects.count()

        if session_count == 0:
            print("No chat sessions found in database.")
            return True

        print("Found in Django database:")
        print(f"  - {session_count} chat sessions")
        print(f"  - {message_count} chat messages")
        print(f"  - {input_count} chat inputs")
        print()

        print("Deleting all chat sessions...")
        ChatSession.objects.all().delete()

        print("Deleting any orphaned chat inputs...")
        ChatInput.objects.all().delete()

        remaining_sessions = ChatSession.objects.count()
        remaining_messages = ChatMessage.objects.count()
        remaining_inputs = ChatInput.objects.count()

        if remaining_sessions == 0 and remaining_messages == 0 and remaining_inputs == 0:
            print()
            print("Successfully deleted:")
            print(f"  - {session_count} chat sessions")
            print(f"  - {message_count} chat messages")
            print(f"  - {input_count} chat inputs")
            return True

        print()
        print("Some records still remain:")
        print(f"  - {remaining_sessions} sessions")
        print(f"  - {remaining_messages} messages")
        print(f"  - {remaining_inputs} inputs")
        return False

    except ImportError as error:
        print(f"Django not available: {error}")
        print("Make sure you are running this from the agent_developer directory with Django installed.")
        if not args.in_container and not args.no_docker_fallback and should_retry_in_docker(error):
            return run_script_in_docker("reset_session.py")
        return False
    except Exception as error:
        print(f"Error deleting chat sessions: {error}")
        if not args.in_container and not args.no_docker_fallback and should_retry_in_docker(error):
            return run_script_in_docker("reset_session.py")
        traceback.print_exc()
        return False


def main():
    """Run the reset flow."""

    args = parse_args()

    print("=" * 60)
    print("CHAT SESSION RESET TOOL")
    print("=" * 60)
    print()
    print("WARNING: This will delete ALL chat sessions for ALL users.")
    print()

    response = "yes" if args.yes else input("Are you sure you want to continue? (yes/no): ").strip().lower()
    if response != "yes":
        print("Operation cancelled.")
        return

    print()
    print("-" * 60)
    print("Clearing chat sessions...")
    print("-" * 60)
    print()

    success = reset_chat_sessions(args)

    print()
    print("=" * 60)
    if success:
        print("Reset complete.")
    else:
        print("Reset completed with warnings.")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Refresh your frontend (http://localhost:3000/c)")
    print("2. Start new conversations with the chatbot")
    print("3. All users will see an empty chat history")


if __name__ == "__main__":
    main()
