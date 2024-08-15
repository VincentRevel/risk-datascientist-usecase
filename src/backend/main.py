# Import Libraries
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle

# Initialization
with open('../../models/baseline_best_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('../../models/optimal_binning_process.pkl', 'rb') as file:
    binning_process = pickle.load(file)
app = FastAPI()


class Item(BaseModel):
    data: dict


@app.post("/score")
async def get_score(item: Item):
    try:
        input_data = item.data
        df = pd.DataFrame([input_data])
        df_binned = binning_process.transform(df)
        df_binned = df_binned[model.feature_names_in_]
        df_binned = df_binned.astype(np.float32)
        input_vector = df_binned.to_numpy()
        prediction = model.predict_proba(input_vector)[:, 1][0]
        prediction = float(prediction)
        return {"score": prediction}
    except Exception as error:
        return {"error": str(error)}