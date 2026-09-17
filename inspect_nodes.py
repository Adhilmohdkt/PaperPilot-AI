#!/usr/bin/env python3
import os
os.environ['GEMINI_API_KEY'] = 'AIzaSyCyXIg5jnowM6HMLcff400feIwUc5gCzIs'
os.environ['GROQ_API_KEY'] = 'REDACTED_GROQ_KEY'

import sys
sys.path.insert(0, r'C:\Users\ASUS\Desktop\PaperPilot AI\backend')
import generation.nodes as nodes_mod
# Get all public functions/classes
public = [(name, getattr(nodes_mod, name)) for name in dir(nodes_mod) if not name.startswith('_')]
for name, obj in public:
    print(f"{name}: {type(obj).__name__}")