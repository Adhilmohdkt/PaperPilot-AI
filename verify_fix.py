import os
os.environ['GROQ_API_KEY'] = 'REDACTED_GROQ_KEY'
os.environ['GEMINI_API_KEY'] = 'AIzaSyCyXIg5jnowM6HMLcff400feIwUc5gCzIs'

import sys
sys.path.insert(0, r'C:\Users\ASUS\Desktop\PaperPilot AI\backend')

from generation.workflow import workflow
import asyncio

async def test():
    initial_state = {
        'conversation_id': 'test',
        'user_query': 'Find recent papers about RAG evaluation.',
        'messages': [],
        'intent': 'general_answer',
        'response': '',
    }
    
    chunks = []
    async for chunk in workflow.astream(initial_state, config={'configurable': {'thread_id': 'test'}}):
        chunks.append(chunk)
    
    result = await workflow.ainvoke(initial_state, config={'configurable': {'thread_id': 'test'}})
    
    intent = result.get('intent')
    answer = str(result.get('answer', ''))[:200]
    academic_papers = result.get('academic_papers', [])
    ranked_papers = result.get('ranked_papers', [])
    
    print('=== Live Academic Search Results ===')
    print('Query: Find recent papers about RAG evaluation.')
    print(f'Intent: {intent}')
    print(f'Answer: {answer}...')
    print()
    print(f'Academic papers found: {len(academic_papers)}')
    print(f'Ranked papers: {len(ranked_papers)}')
    print()
    
    # Show detailed info about ranked papers
    for i, paper in enumerate(ranked_papers[:5]):
        print(f'--- Ranked Paper {i+1} ---')
        title = paper.title if paper.title else 'N/A'
        print(f'Title: {title}')
        provider = 'arXiv' if paper.provider == 'arxiv' else 'OpenAlex'
        print(f'Provider: {provider}')
        arxiv_id = paper.arxiv_id if paper.arxiv_id else 'N/A'
        print(f'ArXiv ID: {arxiv_id}')
        oalex_id = paper.openalex_id if paper.openalex_id else 'N/A'
        print(f'OpenAlex ID: {oalex_id}')
        pub_date = paper.publication_date if paper.publication_date else 'N/A'
        print(f'Publication date: {pub_date}')
        paper_url = paper.paper_url if paper.paper_url else 'N/A'
        print(f'Paper URL: {paper_url}')
        pdf_url = paper.pdf_url if paper.pdf_url else 'N/A'
        print(f'PDF URL: {pdf_url}')
        abstract = paper.abstract if paper.abstract else 'N/A'
        print(f'Abstract: {abstract[:200]}...')
        print()

asyncio.run(test())