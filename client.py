import requests
import time

# Base URL of the FastAPI server
BASE_URL = "http://192.168.1.136:8002"  # Remplacez par l'adresse de votre serveur si différente

# Endpoints
PREDICT_ENDPOINT = f"{BASE_URL}/predict"
PREDICT_BATCH_ENDPOINT = f"{BASE_URL}/predict/batch"
MODEL_INFO_ENDPOINT = f"{BASE_URL}/model-info"

def send_predict_request(data):
    try:
        # Envoi de la requête POST à l'endpoint de prédiction
        response = requests.post(PREDICT_ENDPOINT, json=data)
        
        if response.status_code == 200:
            print("Prediction Response:")
            print(response.json())
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def send_batch_predict_request(commands):
    try:
        # Préparer les données pour la requête par lot
        batch_data = [{"command": cmd} for cmd in commands]
        
        # Envoi de la requête POST à l'endpoint de prédiction par lot
        response = requests.post(PREDICT_BATCH_ENDPOINT, json=batch_data)
        
        if response.status_code == 200:
            print("Batch Prediction Response:")
            print(response.json())
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Batch request failed: {e}")

def get_model_info():
    try:
        # Requête GET pour obtenir les informations du modèle
        response = requests.get(MODEL_INFO_ENDPOINT)
        
        if response.status_code == 200:
            print("Model Information:")
            print(response.json())
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Model info request failed: {e}")

if __name__ == "__main__":
    # Liste de commandes pour test
    # commands = [
    #     "what is there to drink",
    #     "available drinks", 
    #     "what drinks do you have",
    #     "drink options",
    #     "beverage list please"
    # ]
    # commands = [
    #     "turn down the sound",
    #     "reduce the volume",
    #     "lower the volume",
    #     "decrease the sound",
    #     "read playlist Disco",
    #     "launch playlist Disco",
    #     "play playlist Disco",
    #     "start playlist Disco",
    #     "open playlist Disco",
    #     "read playlist Classical",
    #     "launch playlist Classical",
    #     "play playlist Classical",
    #     "start playlist Classical",
    #     "open playlist Classical",
    #     "read playlist Electro",
    #     "launch playlist Electro",
    #     "play playlist Electro",
    #     "start playlist Electro",
    #     "open playlist Electro",
    #     "read playlist Hip-Hop",
    #     "launch playlist Hip-Hop",
    #     "play playlist Hip-Hop",
    #     "start playlist Hip-Hop",
    #     "open playlist Hip-Hop",
    #     "read playlist Jazz",
    #     "launch playlist Jazz",
    #     "play playlist Jazz",
    #     "start playlist Jazz",
    #     "open playlist Jazz",
    #     "read playlist Rock",
    #     "launch playlist Rock",
    #     "play playlist Rock",
    #     "start playlist Rock",
    #     "open playlist Rock",
    #     "launch the turbulence scenario",
    #     "launch the turbulence mode",
    #     "put yourself in turbulence mode",
    #     "activate turbulence mode",
    #     "start the turbulence scenario",
    #     "launch the meeting",
    #     "start the meeting",
    #     "start the video",
    #     "launch the video conference",
    #     "begin the meeting",
    #     "connect to videoconferencing",
    #     "join the video call",
    #     "connect to video call",
    #     "join the conference",
    #     "start video conferencing",
    #     "disconnect from the meeting",
    #     "leave the meeting",
    #     "exit the meeting",
    #     "end the call",
    #     "hang up the conference",
    #     "launch the evacuation scenario",
    #     "launch evacuation mode",
    #     "set evacuation mode",
    #     "activate evacuation mode",
    #     "initiate the evacuation process",
    #     "launch sky guesser",
    #     "start sky guesser",
    #     "activate sky guesser",
    #     "play sky guesser",
    #     "begin sky guesser",
    #     "launch the game",
    #     "start the game",
    #     "activate the game",
    #     "play the game",
    #     "begin the game",
    #     "answer the call",
    #     "pick up the phone",
    #     "take the call",
    #     "accept the call",
    #     "respond to the call",
    #     "launch podcast La french touch",
    #     "open podcast La french touch",
    #     "play podcast La french touch",
    #     "start podcast La french touch",
    #     "begin podcast La french touch",
    #     "put the next podcast",
    #     "next podcast",
    #     "play the next podcast",
    #     "switch to the next podcast",
    #     "move to the next podcast",
    #     "put on the previous podcast",
    #     "previous podcast",
    #     "play the previous podcast",
    #     "switch to the previous podcast",
    #     "move to the previous podcast",
    #     "put on the last podcast",
    #     "last podcast",
    #     "play the last podcast",
    #     "switch to the last podcast",
    #     "move to the last podcast",
    #     "put on the next song",
    #     "next song",
    #     "play the next song",
    #     "switch to the next song",
    #     "move to the next track",
    #     "put on the previous song",
    #     "previous song",
    #     "play the previous song",
    #     "switch to the previous song",
    #     "move to the previous track",
    #     "put on the last song",
    #     "last song",
    #     "play the last song",
    #     "switch to the last song",
    #     "move to the last track",
    #     "show me the lyrics",
    #     "display the lyrics",
    #     "give me the lyrics",
    #     "what are the lyrics",
    #     "lyrics please",
    #     "launch the karaoke",
    #     "start karaoke",
    #     "activate karaoke mode",
    #     "begin karaoke",
    #     "open karaoke",
    #     "start a blind test",
    #     "launch a blind test",
    #     "begin blind test",
    #     "play a blind test",
    #     "activate blind test mode",
    #     "what's the title of the track",
    #     "tell me the song title",
    #     "what song is this",
    #     "name of the song",
    #     "track title please",
    #     "launch intro Odyssey mode",
    #     "put yourself in intro Odyssey mode",
    #     "activate intro Odyssey scenario",
    #     "start intro Odyssey mode",
    #     "initiate intro Odyssey scenario",
    #     "launch take off mode",
    #     "put yourself in take off mode",
    #     "activate take off scenario",
    #     "start take off mode",
    #     "initiate take off scenario",
    #     "launch landing mode",
    #     "put yourself in landing mode",
    #     "activate landing scenario",
    #     "start landing mode",
    #     "initiate landing scenario",
    #     "launch the Interstellar film",
    #     "set the Interstellar film",
    #     "play movie Interstellar",
    #     "show Interstellar movie",
    #     "display Interstellar film",
    #     "launch The Dark Knight film",
    #     "set the The Dark Knight film",
    #     "play movie The Dark Knight",
    #     "show The Dark Knight movie",
    #     "display The Dark Knight film",
    #     "launch the Inception film",
    #     "set the Inception film",
    #     "play movie Inception",
    #     "show Inception movie",
    #     "display Inception film",
    #     "launch the Inglorious Bastards film",
    #     "set the Inglorious Bastards film",
    #     "play movie Inglorious Bastards",
    #     "show Inglorious Bastards movie",
    #     "display Inglorious Bastards film",
    #     "launch the Léon film",
    #     "set the Léon film",
    #     "play movie Léon",
    #     "show Léon movie",
    #     "display Léon film",
    #     "launch the Ready Player One film",
    #     "set the Ready Player One film",
    #     "play movie Ready Player One",
    #     "show Ready Player One movie",
    #     "display Ready Player One film",
    #     "launch the Spider Man film",
    #     "set the Spider Man film",
    #     "play movie Spider Man",
    #     "show Spider Man movie",
    #     "display Spider Man film",
    #     "launch the Star Wars film",
    #     "set the Star Wars film",
    #     "play movie Star Wars",
    #     "show Star Wars movie",
    #     "display Star Wars film",
    #     "I would like a glass of champagne",
    #     "can I have champagne",
    #     "please serve champagne",
    #     "bring me champagne",
    #     "champagne please",
    #     "when do we eat",
    #     "how soon is dinner",
    #     "how much longer until we eat",
    #     "what time is the meal",
    #     "when is mealtime",
    #     "what time will the meal be served?",
    #     "when does the meal start?",
    #     "what are we eating today",
    #     "today's meal",
    #     "what's on the menu",
    #     "what food is available today",
    #     "today's dining options",
    #     "what is there to drink",
    #     "available drinks",
    #     "what drinks do you have",
    #     "drink options",
    #     "beverage list please"
    # ]
    commands = [
        "turn off the sound",
        "reduce the audio",
        "lower the noise level",
        "decrease the volume",
        "play Disco playlist",
        "launch Disco playlist",
        "start Disco playlist",
        "begin Disco playlist",
        "access Disco playlist",
        "play Classical playlist",
        "launch Classical playlist",
        "start Classical playlist",
        "begin Classical playlist",
        "access Classical playlist",
        "play Electro playlist",
        "launch Electro playlist",
        "start Electro playlist",
        "begin Electro playlist",
        "access Electro playlist",
        "play Hip-Hop playlist",
        "launch Hip-Hop playlist",
        "start Hip-Hop playlist",
        "begin Hip-Hop playlist",
        "access Hip-Hop playlist",
        "play Jazz playlist",
        "launch Jazz playlist",
        "start Jazz playlist",
        "begin Jazz playlist",
        "access Jazz playlist",
        "play Rock playlist",
        "launch Rock playlist",
        "start Rock playlist",
        "begin Rock playlist",
        "access Rock playlist",
        "initiate turbulence mode",
        "activate turbulence setting",
        "enable turbulence mode",
        "switch to turbulence scenario",
        "begin turbulence mode",
        "start the meeting session",
        "initiate the meeting",
        "begin the video session",
        "launch the video call",
        "start video meeting",
        "connect to the video conference",
        "join the virtual meeting",
        "access the video call",
        "enter the video conference",
        "begin video connection",
        "end the video session",
        "leave the call",
        "disconnect from the conference",
        "terminate the video call",
        "hang up the meeting",
        "initiate evacuation process",
        "activate evacuation procedure",
        "enable evacuation scenario",
        "start evacuation mode",
        "trigger evacuation process",
        "play Sky Guesser",
        "start Sky Guesser",
        "begin Sky Guesser game",
        "launch Sky Guesser game",
        "enable Sky Guesser mode",
        "start the game session",
        "initiate the gameplay",
        "play the session",
        "begin the gaming experience",
        "activate game mode",
        "accept the phone call",
        "answer the phone",
        "pick up the call",
        "respond to the phone call",
        "take the incoming call",
        "start the podcast La French Touch",
        "play podcast La French Touch",
        "listen to podcast La French Touch",
        "begin podcast La French Touch",
        "activate podcast La French Touch",
        "move to the upcoming podcast",
        "play the next podcast episode",
        "switch to the next audio show",
        "load the next podcast",
        "queue up the next podcast",
        "play the prior podcast",
        "switch to the previous episode",
        "access the earlier podcast",
        "load the previous audio show",
        "listen to the prior podcast",
        "access the final podcast episode",
        "play the last podcast entry",
        "load the concluding podcast",
        "switch to the ending podcast",
        "listen to the ultimate podcast",
        "play the next song in the list",
        "switch to the following track",
        "listen to the upcoming song",
        "queue the next audio",
        "play the succeeding song",
        "return to the earlier track",
        "play the previous music piece",
        "access the preceding song",
        "listen to the former track",
        "queue up the earlier song",
        "play the concluding track",
        "switch to the ultimate music piece",
        "listen to the final song",
        "queue the last music file",
        "play the ending audio",
        "show the song lyrics",
        "display the text of the song",
        "provide the song lyrics",
        "what are the words of the song",
        "can I see the lyrics",
        "activate karaoke feature",
        "begin karaoke session",
        "turn on karaoke mode",
        "start singing karaoke",
        "open the karaoke system",
        "begin a music guessing game",
        "launch a blind music test",
        "play a guessing game",
        "start the blind test challenge",
        "activate a blind test round",
        "what is the name of this song",
        "tell me the track title",
        "identify the song being played",
        "what’s the name of this tune",
        "what track is this",
        "switch to intro Odyssey mode",
        "start the intro Odyssey experience",
        "enable the intro Odyssey scenario",
        "initiate the intro Odyssey setting",
        "activate intro Odyssey mode",
        "begin takeoff mode",
        "enable the takeoff scenario",
        "switch to takeoff experience",
        "start the takeoff simulation",
        "activate the takeoff process",
        "start landing scenario",
        "enable the landing mode",
        "initiate the landing experience",
        "activate the landing sequence",
        "switch to landing mode",
        "play Interstellar movie",
        "start Interstellar film",
        "show Interstellar video",
        "watch the Interstellar movie",
        "display the film Interstellar",
        "start The Dark Knight video",
        "play The Dark Knight film",
        "show the movie The Dark Knight",
        "display the video The Dark Knight",
        "watch The Dark Knight movie",
        "start the film Inception",
        "play Inception movie",
        "watch the video Inception",
        "show the movie Inception",
        "display Inception video",
        "start Inglorious Bastards video",
        "play Inglorious Bastards movie",
        "show the film Inglorious Bastards",
        "display Inglorious Bastards video",
        "watch Inglorious Bastards movie",
        "play Léon movie",
        "start Léon film",
        "watch Léon video",
        "display the movie Léon",
        "show the video Léon",
        "play Ready Player One film",
        "start the movie Ready Player One",
        "watch the video Ready Player One",
        "display the film Ready Player One",
        "show Ready Player One video",
        "play Spider-Man movie",
        "start Spider-Man film",
        "watch Spider-Man video",
        "display the movie Spider-Man",
        "show the film Spider-Man",
        "play Star Wars movie",
        "start the film Star Wars",
        "watch Star Wars video",
        "display the movie Star Wars",
        "show the video Star Wars",
        "I would like a glass of sparkling wine",
        "can I get champagne, please",
        "serve some champagne for me",
        "bring over a glass of bubbly",
        "champagne would be great",
        "how long until dinner",
        "when will food be served",
        "is it almost time to eat",
        "what is the meal schedule",
        "when will the food arrive",
        "what’s being served today",
        "what meals are available",
        "what’s the menu today",
        "what food options do we have",
        "what’s being cooked today",
        "what beverages are on the menu",
        "what drinks are available",
        "what kind of drinks can I order",
        "can you show me the drink options",
        "please provide the drink selection"
        ]

    # Démonstration des différents endpoints
    
    # 1. Test de prédiction individuelle
    print("--- Individual Prediction Test ---")
    for command in commands:
        request_data = {
            "command": command,
            # "model_name": "logistic_regression_model.pkl"
        }
        send_predict_request(request_data)
        time.sleep(2)  # Délai entre les requêtes
    
    # 2. Test de prédiction par lot
    print("\n--- Batch Prediction Test ---")
    # send_batch_predict_request(commands)
    
    # 3. Récupération des informations du modèle
    print("\n--- Model Information Test ---")
    get_model_info()