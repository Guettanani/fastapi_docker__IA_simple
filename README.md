# Command Classifier API

## Prérequis
- Docker
- Docker Compose

## Lancement
```bash
docker-compose up --build


project_root/
│
├── app/
│   ├── main.py         # Code FastAPI
│   ├── predictor.py    # Classe de prédiction
│   └── command_classifier_overfitted_svm.pkl  # Modèle sauvegardé
│
├── requirements.txt    # Dépendances Python
├── Dockerfile          # Instructions de build Docker
├── docker-compose.yml  # Configuration Docker Compose
└── README.md           # Documentation