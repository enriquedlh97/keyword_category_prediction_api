from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List
from keyword_category_predictor.models.bert_base_multilingual_cased import Model, get_model


app = FastAPI()


class BatchRequest(BaseModel):
    batch: list


class Label(BaseModel):
    label: str
    score: float


class Keyword(BaseModel):
    keyword: str
    labels: List[Label] = []


class BatchResponse(BaseModel):
    classifications: List[Keyword] = []


@app.post("/predict", response_model=BatchResponse)
def predict(request: BatchRequest, model: Model = Depends(get_model)):

    classifications = []
    for keyword in request.batch:
        _, probabilities = model.predict(keyword)
        keyword_output = {"keyword": keyword}
        category_scores = []
        for category in probabilities:
            category_scores.append({"label": category, "score": probabilities[category]})

        keyword_output["labels"] = category_scores
        classifications.append(keyword_output)

    return BatchResponse(classifications=classifications)
