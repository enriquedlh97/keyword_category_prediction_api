from typing import Dict
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from keyword_category_predictor.models.bert_base_multilingual_cased import Model, get_model

app = FastAPI()


class KeywordRequest(BaseModel):
    text: list
    # text: str


class KeywordResponse(BaseModel):
    classifications: Dict


@app.post("/predict", response_model=KeywordResponse)
def predict(request: KeywordRequest, model: Model = Depends(get_model)):
    classifications = []
    for batch in request.text:
        _, probabilities = model.predict(request.text)
        batch_output = {"keyword": batch}
        category_scores = []
        for category, score in probabilities:
            category_scores.append({"label": category, "score": score})

        batch_output["labels"] = category_scores

    classifications.append(batch_output)

    return KeywordResponse(classifications=classifications)


# {
#   “classifications”: [
#    {
#     “keyword”: “fried chicken”,
#     “labels”: [
#      {
#       “label”: “Food & Groceries”,
#       “score”: 0.85
#      },
#      {
#       “label”: “Dining & Nightlife”,
#       “score”: 0.65
#      }
#     ]
#    },
#    {
#     “keyword”: “hotels”,
#     “labels”: [
#      {
#       “label”: “Travel & Tourism”,
#       “score”: 0.9
#      }
#     ]
#    }
#   ]
# }
