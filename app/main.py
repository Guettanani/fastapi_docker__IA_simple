import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Optional

from predictor import CommandPredictor

# Modèles Pydantic pour la validation des données
class CommandInput(BaseModel):
    command: Union[List[str], str] = Field(..., description="Commande à prédire")

class PredictionResponse(BaseModel):
    command: Union[List[str], str]
    predicted_class: str
    confidence: Optional[float] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

# Initialisation de l'application FastAPI et du prédicteur
app = FastAPI(
    title="Command Classifier API",
    description="API pour la prédiction de commandes vocales",
    version="1.0.0"
)

# Initialiser le prédicteur de commandes
predictor = CommandPredictor()

@app.post("/predict", response_model=PredictionResponse)
async def predict_command(command_input: CommandInput):
    """
    Prédit la classe d'une commande unique
    
    - **command**: Commande à prédire (peut être une liste de mots ou une chaîne)
    - Retourne la classe prédite et la confiance
    """
    try:
        prediction, confidence = predictor.predict(command_input.command)
        
        return PredictionResponse(
            command=command_input.command,
            predicted_class=prediction,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_commands(commands: List[CommandInput]):
    """
    Prédit les classes pour un lot de commandes
    
    - **commands**: Liste de commandes à prédire
    - Retourne une liste de prédictions avec leurs confidences
    """
    try:
        batch_predictions = []
        
        for cmd_input in commands:
            prediction, confidence = predictor.predict(cmd_input.command)
            batch_predictions.append(
                PredictionResponse(
                    command=cmd_input.command,
                    predicted_class=prediction,
                    confidence=confidence
                )
            )
        
        return BatchPredictionResponse(predictions=batch_predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """
    Retourne les informations du modèle
    """
    return {
        "classes": list(predictor.classes),
        "model_params": predictor.best_params,
        "vectorizer_params": predictor.vectorizer.get_params()
    }