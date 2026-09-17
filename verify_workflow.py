import os
os.environ['GROQ_API_KEY'] = 'REDACTED_GROQ_KEY'
os.environ['GEMINI_API_KEY'] = 'AIzaSyCyXIg5jnowM6HMLcff400feIwUc5gCzIs'

import sys
sys.path.insert(0, r'C:\Users\ASUS\Desktop\PaperPilot AI\backend')

# Test the actual function call path - import nodes first (which uses lazy imports)
from generation.nodes import academic_search_node
print('nodes: OK - lazy imports work')

# Now test the academic_search_node function execution
import asyncio

async def test():
    state = {
        'conversation_id': 'test',
        'user_query': 'Find recent papers about RAG evaluation.',
        'messages': [],
        'intent': 'general_answer',
        'response': '',
    }
    
    # Call academic_search_node
    after_ac = await academic_search_node({**state, 'intent': 'academic_research'})
    print(f'academic_papers: {len(after_ac.get("academic_papers", []))}')
    print(f'ranked_papers: {len(after_ac.get("ranked_papers", []))}')
    
    # Show some paper details
    papers = after_ac.get('academic_papers', [])
    for p in papers[:3]:
        print(f'  Paper: title={p.title[:50] if p.title else "N/A"}, provider={p.provider}, arxiv_id={p.arxiv_id}')
    
    ranked = after_ac.get('ranked_papers', [])
    for p in ranked[:3]:
        print(f'  Ranked: title={p.title[:50] if p.title else "N/A"}')

asyncio.run(test())