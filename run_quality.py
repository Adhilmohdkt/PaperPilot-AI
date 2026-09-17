#!/usr/bin/env python
import os
os.environ['GROQ_API_KEY'] = 'REDACTED_GROQ_KEY'
os.environ['GEMINI_API_KEY'] = 'AIzaSyCyXIg5jnowM6HMLcff400feIwUc5gCzIs'

import sys
sys.path.insert(0, r'C:\Users\ASUS\Desktop\PaperPilot AI\backend')

sys.exit(os.system("python -m pytest backend\tests\quality\test_quality.py -v 2>&1 > quality_output.txt"))