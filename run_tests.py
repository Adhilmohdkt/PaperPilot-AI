#!/usr/bin/env python
"""Run the existing pytest tests."""
import os
import sys

# Set environment variables
os.environ['GROQ_API_KEY'] = 'REDACTED_GROQ_KEY'
os.environ['GEMINI_API_KEY'] = 'AIzaSyCyXIg5jnowM6HMLcff400feIwUc5gCzIs'

# Run pytest
sys.exit(os.system(f"python -m pytest backend\\test_workflow.py -v 2>&1 > test_output.txt"))