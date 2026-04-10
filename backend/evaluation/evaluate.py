import os
import json
import warnings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    faithfulness,
    context_recall
)
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.retriever import retrieve
from generation.generator import generate_answer_stream

# Ignore LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

def run_evaluation():
    print("Starting End-to-End RAGAS Evaluation...")
    
    groq_api_key = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
    
    try:
        from langchain_groq import ChatGroq
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        print("Missing Evaluation Dependencies! Run: pip install langchain-groq langchain-community")
        return

    eval_llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=groq_api_key)
    eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 1. Real Test Cases based on your uploaded PDFs
    test_queries = [
        {
            "query": "What is RAG?",
            "ground_truth": "RAG stands for Retrieval-Augmented Generation."
        }
    ]

    # 2. Run them through your actual pipeline to get Answers and Contexts
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    print("Running queries through the Retriever & Generator...")
    for idx, test in enumerate(test_queries):
        query = test["query"]
        print(f"   [{idx+1}/3] Answering: {query}")
        
        chunks = retrieve(query)
        # Extract just the text from the chunk dictionaries for RAGAS
        context_texts = [c["text"] if isinstance(c, dict) else c for c in chunks]
        
        # Drain the new Server-Sent Event stream into a full string for Ragas
        generator_obj = generate_answer_stream(query, chunks, history=[])
        answer = "".join([token for token in generator_obj])
        
        questions.append(query)
        answers.append(answer)
        contexts_list.append(context_texts)
        ground_truths.append(test["ground_truth"])

    # 3. Format strictly for RAGAS expected layout
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    metrics = [faithfulness]

    try:
        print("\nScoring your RAG pipeline using LLaMA and Sentence-Transformers...")
        results = evaluate(
            dataset=dataset, 
            metrics=metrics,
            llm=eval_llm,
            embeddings=eval_embeddings
        )
        
        print("\nEvaluation Complete! Here are your Quality Scores:")
        
        # output metric averages across all questions
        scores = results.to_pandas().mean(numeric_only=True).to_dict()
        print(json.dumps(scores, indent=4))
        
    except Exception as e:
        print(f"\nEvaluation Failed: {e}")

if __name__ == "__main__":
    run_evaluation()
