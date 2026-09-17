#!/usr/bin/env python3
import os
os.environ['GEMINI_API_KEY'] = 'AIzaSyCyXIg5jnowM6HMLcff400feIwUc5gCzIs'
os.environ['GROQ_API_KEY'] = 'REDACTED_GROQ_KEY'

import sys
sys.path.insert(0, r'C:\Users\ASUS\Desktop\PaperPilot AI\backend')

from generation.workflow import workflow

# Check the graph structure
graph = workflow.graph
print('Graph type:', type(graph))
print()

# Print all edges
print('Edges:')
for source, targets in graph._graph.items():
    print(f'  {source} -> {list(targets)}')
print()

# Print all nodes
print('Nodes:', list(graph.nodes))