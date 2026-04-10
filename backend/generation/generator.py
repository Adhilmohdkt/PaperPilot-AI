from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY"))

def generate_answer_stream(query, chunks, history=None):
    if history is None:
        history = []
        
    context_str = ""
    for idx, c in enumerate(chunks):
        # Fallback to string if a bare string is passed during tests
        text = c["text"] if isinstance(c, dict) else c
        source = c["source"] if isinstance(c, dict) else "Unknown"
        context_str += f"--- Source {idx+1}: {source} ---\n{text}\n\n"

    system_prompt = f"""
You are an expert Research Assistant answering a user's query based on the following documents.

Rules for your response:
1. Synthesize the context into a clear, direct, and comprehensive answer.
2. DO NOT use robotic introductory phrases like "Based on the provided context, the answer is...". Just answer directly.
3. You MUST cite your claims inline using the provided source filenames (e.g., "This approach improves retrieval accuracy (source.pdf).").
4. If the context does not contain enough information to answer the question, clearly state "I could not find the answer in the retrieved documents."
5. You have access to the conversation memory. Use it to answer conversational follow-up questions intelligently.

Context:
{context_str}
"""

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Inject conversational memory into the LLM context
    for msg in history:
        messages.append(msg)
        
    # Append the actual new query
    messages.append({"role": "user", "content": f"Question:\n{query}"})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=messages,
        stream=True
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content