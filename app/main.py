from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from model import generate_response

app = FastAPI()

class Message(BaseModel):
    message: str

@app.post("/chat")
def chat(msg: Message):
    response = generate_response(msg.message)
    return {"response": response}

@app.get("/")
def login_page():
    return FileResponse("static/login.html")

@app.get("/chat")
def chat_page():
    return FileResponse("static/index.html")

app.mount("/", StaticFiles(directory="static", html=True), name="static")