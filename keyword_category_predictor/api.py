from typing import Dict
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from keyword_category_predictor.models.model import Model, get_model

app = FastAPI()


class KeywordRequest(BaseModel):
    text: str


class KeywordResponse(BaseModel):
    # top_category: str
    confidence: float
    probabilities: Dict[str, float]


@app.post("/predict", response_model=KeywordResponse)
def predict(request: KeywordRequest, model: Model = Depends(get_model)):
    confidence, probabilities = model.predict(request.text)
    return KeywordResponse(
        # top_category="Travel & Tourism",
        confidence=confidence,
        probabilities=probabilities

        # probabilities = dict(
        #     health=0.0987362191081047,
        #     vehicles=0.038122113794088364,
        #     hobbies_and_leisure=0.03911390155553818,
        #     food_and_groceries=0.0387321300804615,
        #     retailers_and_general_merchandise=0.03501097485423088,
        #     arts_and_entertainment=0.02139897830784321,
        #     jobs_and_education=0.04255586117506027,
        #     law_and_government=0.04836435243487358,
        #     home_and_garden=0.07447872310876846,
        #     finance=0.03807186335325241,
        #     computers_and_consumer_electronics=0.019762972369790077,
        #     internet_and_telecom=0.01948094740509987,
        #     sports_and_fitness=0.03428463265299797,
        #     dining_and_nightlife=0.02335537038743496,
        #     business_and_industrial=0.23443226516246796,
        #     occasions_and_gifts=0.021712874993681908,
        #     travel_and_tourism=0.14958643913269043,
        #     news_media_and_publications=0.04497528821229935,
        #     apparel=0.020958907902240753,
        #     beauty_and_personal_care=0.02543235570192337,
        #     family_and_community=0.06581879407167435,
        #     real_estate=0.07752881944179535
        # )
    )
