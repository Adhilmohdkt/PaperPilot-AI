#!/usr/bin/env python3
import os
os.environ['GEMINI_API_KEY'] = 'AIzaSyCyXIg5jnowM6HMLcff400feIwUc5gCzIs'
os.environ['GROQ_API_KEY'] = 'REDACTED_GROQ_KEY'

import sys
sys.path.insert(0, r'C:\Users\ASUS\Desktop\PaperPilot AI\backend')

from generation.state import AgentState
print([x for x in dir() if not x.startswith('_')])