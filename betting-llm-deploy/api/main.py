from fastapi import FastAPI
from api.schemas import Query
from src.inference import generate_answer

app = FastAPI(title="Betting Q&A LLM")

@app.post("/ask")
def ask(query: Query):
    answer = generate_answer(query.question)
    return {"answer": answer}
