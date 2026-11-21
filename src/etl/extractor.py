import os
import time

import pandas as pd
import requests

from src.config import MAX_USERS, MON_STEAM_ID, STEAM_API_KEY

REQUEST_DELAY = 1.0


# --- FONCTIONS API ---
class SteamAPIError(RuntimeError):
    """Échec technique : ne doit pas être assimilé à une bibliothèque vide."""


class SteamProfileUnavailable(SteamAPIError):
    """Bibliothèque inaccessible ou réponse ne permettant pas de la déterminer."""


def _call_steam_api(url, params, max_retries=3):
    if not params.get("key"):
        raise SteamAPIError("Renseigne STEAM_API_KEY dans le fichier .env.")
    if max_retries < 1:
        raise ValueError("max_retries doit être positif")
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=(5, 30))
        except requests.RequestException:
            if attempt + 1 == max_retries:
                raise SteamAPIError(
                    "Steam est injoignable après plusieurs tentatives."
                ) from None
        else:
            if response.status_code == 200:
                return response
            if response.status_code == 401:
                raise SteamAPIError("Clé Steam refusée : vérifier STEAM_API_KEY.")
            if response.status_code == 403:
                raise SteamProfileUnavailable(
                    "Accès refusé : vérifier la clé et la visibilité du profil."
                )
            if response.status_code != 429 and not 500 <= response.status_code < 600:
                raise SteamAPIError(f"Erreur Steam HTTP {response.status_code}.")
            if attempt + 1 == max_retries:
                raise SteamAPIError(
                    f"Steam indisponible : HTTP {response.status_code}."
                )
        time.sleep(REQUEST_DELAY * 2**attempt)
    raise SteamAPIError("Échec de l'appel Steam.")


def _read_json(response):
    try:
        data = response.json()
    except ValueError:
        raise SteamAPIError("Réponse JSON Steam invalide.") from None
    if not isinstance(data, dict):
        raise SteamAPIError("Format de réponse Steam invalide.")
    return data


def get_friends(steam_id):
    """
    Interroge l'API Steam pour récupérer le réseau d'amis d'un joueur.

    Args:
        steam_id (str): L'identifiant Steam64 du joueur cible.

    Returns:
        list: Une liste de SteamIDs (str), ou une liste vide si le profil est privé/inaccessible.
    """
    url = "https://api.steampowered.com/ISteamUser/GetFriendList/v0001/"
    params = {"key": STEAM_API_KEY, "steamid": steam_id, "relationship": "friend"}
    response = _call_steam_api(url, params)

    if response.status_code == 200:
        data = _read_json(response)
        return [
            friend["steamid"]
            for friend in data.get("friendslist", {}).get("friends", [])
        ]
    return []


def get_owned_games(steam_id):
    """
    Récupère la bibliothèque de jeux d'un utilisateur, avec les jeux gratuits joués.

    Args:
        steam_id (str): L'identifiant Steam64 du joueur cible.

    Returns:
        list: Une liste de dictionnaires contenant les données des jeux (appid, playtime_forever),
              ou une liste vide pour une bibliothèque vide. Les erreurs lèvent SteamAPIError.
    """
    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    params = {
        "key": STEAM_API_KEY,
        "steamid": steam_id,
        "format": "json",
        "include_played_free_games": "1",  # inclut les jeux gratuits joués
    }
    response = _call_steam_api(url, params)

    if response.status_code == 200:
        data = _read_json(response)
        payload = data.get("response", {})
        if not isinstance(payload, dict):
            raise SteamAPIError("Format de bibliothèque invalide.")
        if "games" not in payload and payload.get("game_count") != 0:
            raise SteamProfileUnavailable(
                "Bibliothèque inaccessible : vérifier sa visibilité."
            )
        games = payload.get("games", [])
        if not isinstance(games, list) or any(
            not isinstance(game, dict) or "appid" not in game for game in games
        ):
            raise SteamAPIError("Liste de jeux Steam invalide.")
        return games
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
    file_attente = [str(start_id)]
    dataset = []

    # Condition d'arrêt : file vide ou limite de sécurité atteinte
    while file_attente and len(utilisateurs_visites) < MAX_USERS:
        user_id = file_attente.pop(0)

        # Prévention des boucles infinies (si A est ami avec B, et B ami avec A)
        if user_id in utilisateurs_visites:
            continue

        utilisateurs_visites.add(user_id)
        try:
            jeux = get_owned_games(user_id)
        except SteamProfileUnavailable as exc:
            print(f"Profil inaccessible : {exc}")
            jeux = []

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
            try:
                nouveaux_amis = get_friends(user_id)
            except SteamProfileUnavailable:
                nouveaux_amis = []
            for ami in nouveaux_amis:
                if ami not in utilisateurs_visites and ami not in file_attente:
                    file_attente.append(ami)
        else:
            print(
                f"[{len(utilisateurs_visites)}/{MAX_USERS}] Profil privé ou vide ignoré."
            )

        time.sleep(REQUEST_DELAY)

    # Etape de chargement (load) : sauvegarde dans un fichier plat
    df = pd.DataFrame(dataset)
    if not df.empty:
        # Creation du dossier de destination s'il n'existe pas
        os.makedirs("data", exist_ok=True)

        # Conserver aussi les jeux possédés jamais lancés pour les exclure des suggestions.
        df = df.drop_duplicates(["user_id", "game_id"], keep="last")
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
