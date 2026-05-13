#!/usr/bin/env python
"""
Simple script to query the Agentic RAG system directly.
"""
import os
import sys
import django

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.core.settings')

# Setup Django
django.setup()

from base.services.agent.rag.providers import RagProviders

def query(question: str, file_type: str = None):
    """Query the RAG system."""
    rag = RagProviders()
    
    # Call the response method which uses Agentic RAG
    result = rag.generative_service.response(
        question=question,
        file_type=file_type,
    )
    
    return result

if __name__ == "__main__":
    # Get question from command line or use default
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "show me all the Table of Content in this document system administration"
    print(f"Querying: {question}")
    print("-" * 50)
    
    try:
        result = query(question)
        print("\nResult:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
