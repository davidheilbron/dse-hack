import pickle
import pandas as pd
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="Titanic Survival CLassification",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

model = pickle.load(
    open('model.pkl', 'rb')
)


@app.get("/")
def read_root(text: str = ""):
    if not text:
        return f"Try to append ?text=something in the URL!"
    else:
        return text


class Features(BaseModel):
    Pclass: int
    Sex: str
    SibSp: int
    Parch: int

@app.post("/predict/")
def predict(persons: List[Features]) -> List[str]:
    X = pd.DataFrame([dict(person) for person in persons])
    y_pred = model.predict(X)
    return list(y_pred)
