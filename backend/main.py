from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "PaperPilot AI Backend Running 🚀"}

@app.get("/query")
def query(q: str):
    return {"query": q, "response": "This is a placeholder"}