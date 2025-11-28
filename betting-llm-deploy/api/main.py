from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import generate_answer

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QuestionRequest):
    answer = generate_answer(req.question)
    return {"answer": answer}
