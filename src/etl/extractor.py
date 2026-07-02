import os
import requests
import pandas as pd
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("STEAM_API_KEY")
MON_STEAM_ID = "76561198305752421"

# --- FONCTIONS API ---
def get_friends(steam_id):
    url = "http://api.steampowered.com/ISteamUser/GetFriendList/v0001/"
    params = {
        "key": API_KEY, 
        "steamid": steam_id, 
        "relationship": "friend"
        }
    response = requests.get(url, params=params)
        
    if response.status_code == 200:
        data = response.json()
        return [friend["steamid"] for friend in data.get("friendslist", {}).get("friends", [])]
    return []

def get_owned_games(steam_id):
    url = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    params = {
        "key": API_KEY, 
        "steamid": steam_id, 
        "format": "json",
        "include_played_free_games": "1"
    } 
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data.get("response", {}).get("games", [])
    return []

# --- PIPELINE D'EXTRACTION ---
def run_extraction(start_id, max_users=100):
    print("Démarrage de l'extraction...")
    utilisateurs_visites = set()
    file_attente = [start_id]
    dataset = []

    while file_attente and len(utilisateurs_visites) < max_users:
        user_id = file_attente.pop(0)
        
        if user_id in utilisateurs_visites:
            continue
            
        utilisateurs_visites.add(user_id)
        jeux = get_owned_games(user_id)
        
        if jeux:
            print(f"[{len(utilisateurs_visites)}/{max_users}] Profil public trouvé ({len(jeux)} jeux) : {user_id}")
            for jeu in jeux:
                dataset.append({
                    "user_id": user_id,
                    "game_id": jeu.get("appid"),
                    "playtime_minutes": jeu.get("playtime_forever", 0)
                })
            nouveaux_amis = get_friends(user_id)
            for ami in nouveaux_amis:
                if ami not in utilisateurs_visites and ami not in file_attente:
                    file_attente.append(ami)
        else:
            print(f"[{len(utilisateurs_visites)}/{max_users}] Profil privé ou vide ignoré.")

    # Sauvegarde des données
    df = pd.DataFrame(dataset)
    if not df.empty:
        # Creation du dossier 'data' s'il n'existe pas déjà
        os.makedirs("data", exist_ok=True)
        df = df[df["playtime_minutes"] > 0]
        df.to_csv("data/steam_raw_data.csv", index=False)
        print(f"\nExtraction terminée ! {len(df)} interactions sauvegardées dans data/steam_raw_data.csv")
    else:
        print("\nAucune donnée récupérée.")

# --- POINT D'ENTREE DU SCRIPT ---
if __name__ == "__main__":
    run_extraction(MON_STEAM_ID, max_users=100)