from typing import Dict
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from keyword_category_predictor.models.bert_base_multilingual_cased import Model, get_model

app = FastAPI()


class KeywordRequest(BaseModel):
    text: str


class KeywordResponse(BaseModel):
    probabilities: Dict[str, float]


@app.post("/predict", response_model=KeywordResponse)
def predict(request: KeywordRequest, model: Model = Depends(get_model)):
    _, probabilities = model.predict(request.text)
    return KeywordResponse(probabilities=probabilities)
