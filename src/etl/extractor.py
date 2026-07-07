import os
import requests
import pandas as pd
from src.config import API_KEY, MON_STEAM_ID, MAX_USERS


# --- FONCTIONS API ---
def get_friends(steam_id):
    """
    Interroge l'API Steam pour récupérer le réseau d'amis d'un joueur.

    Args:
        steam_id (str): L'identifiant Steam64 du joueur cible.

    Returns:
        list: Une liste de SteamIDs (str), ou une liste vide si le profil est privé/inaccessible.
    """
    url = "http://api.steampowered.com/ISteamUser/GetFriendList/v0001/"
    params = {"key": API_KEY, "steamid": steam_id, "relationship": "friend"}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        return [
            friend["steamid"]
            for friend in data.get("friendslist", {}).get("friends", [])
        ]
    return []


def get_owned_games(steam_id):
    """
    Récupère la bibliothèque de jeux d'un utilisateur, y compris les jeux partagés et gratuits.

    Args:
        steam_id (str): L'identifiant Steam64 du joueur cible.

    Returns:
        list: Une liste de dictionnaires contenant les données des jeux (appid, playtime_forever),
              ou une liste vide en cas d'erreur.
    """
    url = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    params = {
        "key": API_KEY,
        "steamid": steam_id,
        "format": "json",
        "include_played_free_games": "1",  # inclut les Free-to-play et les jeux issus du partage familial
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data.get("response", {}).get("games", [])
    return []


# --- PIPELINE D'EXTRACTION ---
def run_extraction(start_id, MAX_USERS):
    """
    Exécute un parcours de graphe (BFS) pour aspirer les données des joueurs et de leurs amis.
    Transforme les données brutes et les sauvegarde au format CSV.

    Args:
        start_id (str): L'identifiant Steam64 du point de départ.
        max_users (int): Limite de sécurité pour éviter le rate-limiting de l'API.
    """
    print("Démarrage de l'extraction...")

    # Utilisation d'un set (ensemble) pour garantir l'unicité et optimiser la vitesse de recherche
    utilisateurs_visites = set()

    # File d'attente pour le parcours en largeur (Breadth-First Search)
    file_attente = [start_id]
    dataset = []

    # Condition d'arrêt : file vide ou limite de sécurité atteinte
    while file_attente and len(utilisateurs_visites) < MAX_USERS:
        user_id = file_attente.pop(0)

        # Prévention des boucles infinies (si A est ami avec B, et B ami avec A)
        if user_id in utilisateurs_visites:
            continue

        utilisateurs_visites.add(user_id)
        jeux = get_owned_games(user_id)

        # Filtre sur les profils publics (qui renvoient des jeux)
        if jeux:
            print(
                f"[{len(utilisateurs_visites)}/{MAX_USERS}] Profil public trouvé ({len(jeux)} jeux) : {user_id}"
            )

            # Etape de transformation : on aplatit le JSON complexxe en une structure tabulaire simple
            for jeu in jeux:
                dataset.append(
                    {
                        "user_id": user_id,
                        "game_id": jeu.get("appid"),
                        "playtime_minutes": jeu.get("playtime_forever", 0),
                    }
                )

            # Elargissement du réseau : on ajoute les amis à la file d'attente
            nouveaux_amis = get_friends(user_id)
            for ami in nouveaux_amis:
                if ami not in utilisateurs_visites and ami not in file_attente:
                    file_attente.append(ami)
        else:
            print(
                f"[{len(utilisateurs_visites)}/{MAX_USERS}] Profil privé ou vide ignoré."
            )

    # Etape de chargement (load) : sauvegarde dans un fichier plat
    df = pd.DataFrame(dataset)
    if not df.empty:
        # Creation du dossier de destination s'il n'existe pas
        os.makedirs("data", exist_ok=True)

        # Nettoyage final : on retire les jeux installés mais jamais lancés (bruit statistique)
        df = df[df["playtime_minutes"] > 0]
        df.to_csv("data/steam_raw_data.csv", index=False)
        print(
            f"\nExtraction terminée ! {len(df)} interactions sauvegardées dans data/steam_raw_data.csv"
        )
    else:
        print("\nAucune donnée récupérée.")


# --- POINT D'ENTREE DU SCRIPT ---
# Ce bloc garantit que l'extraction ne se lance pas automatiquement si le fichier est importé ailleurs
if __name__ == "__main__":
    run_extraction(MON_STEAM_ID, MAX_USERS)
