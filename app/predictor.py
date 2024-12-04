import pickle
import numpy as np

class CommandPredictor:
    def __init__(self, model_path='command_classifier_overfitted_svm.pkl'):
        """
        Initialise le prédicteur en chargeant le modèle sauvegardé
        
        Args:
            model_path (str): Chemin vers le fichier du modèle sauvegardé
        """
        with open(model_path, 'rb') as f:
            self.components = pickle.load(f)
        
        self.vectorizer = self.components['vectorizer']
        self.model = self.components['model']
        self.classes = self.components['classes']
        self.best_params = self.components['best_params']
    
    def predict(self, command):
        """
        Prédit la classe correspondant à une commande
        
        Args:
            command (list or str): Commande à prédire
        
        Returns:
            str: Classe prédite
            float: Probabilité de prédiction
        """
        # Convertir la commande en chaîne si c'est une liste
        if isinstance(command, list):
            command = " ".join(command)
        
        # Vectorisation de la commande
        X_input = self.vectorizer.transform([command])
        
        # Prédiction
        prediction = self.model.predict(X_input)[0]
        
        # Probabilités (si disponibles)
        try:
            proba = self.model.predict_proba(X_input)[0]
            confidence = max(proba)
        except AttributeError:
            confidence = None
        
        return prediction, confidence