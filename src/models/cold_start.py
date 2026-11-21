# --> nouveau fichier, à placer dans src/models/cold_start.py
"""
Gestion du cold start : recommandations pour un SteamID absent de la
matrice d'interaction utilisée à l'entraînement.

Principe : la matrice de similarité Item-Item (sim_items) ne dépend que des
jeux, pas des utilisateurs. On peut donc calculer des recommandations pour
un nouveau joueur SANS recalculer sim_items, simplement en construisant à
la volée son vecteur de notes (via l'API Steam) et en le comparant au
catalogue déjà connu.
"""

import pandas as pd

from src.etl.extractor import get_owned_games
from src.models.data_utils import convert_playtime_to_rating


def build_new_user_vector(steam_id, catalogue_jeux):
    """
    Interroge l'API Steam pour un joueur inconnu du dataset et construit
    son vecteur de notes, aligné sur le catalogue de jeux déjà connu.

    Args:
        steam_id (str): SteamID64 du nouveau joueur.
        catalogue_jeux (pd.Index): Les game_id présents dans la matrice d'entraînement.

    Returns:
        pd.Series | None: Vecteur indexé par game_id (0 = non possédé ou
                           hors catalogue), ou None si profil privé/sans jeux.
    """
    jeux = get_owned_games(steam_id)
    if not jeux:
        return None

    notes = {}
    for jeu in jeux:
        game_id = jeu.get("appid")
        heures = jeu.get("playtime_forever", 0) / 60
        notes[game_id] = convert_playtime_to_rating(heures)

    vecteur = pd.Series(0.0, index=catalogue_jeux)
    for game_id, note in notes.items():
        # On ignore les jeux hors catalogue : impossibles à comparer car
        # absents de la matrice de similarité (jamais vus à l'entraînement).
        if game_id in vecteur.index:
            vecteur[game_id] = note

    return vecteur


def get_cold_start_recommendations(steam_id, matrice_interaction, df_sim_items, top_n):
    """Utilise le même moteur Item-Based, avec une bibliothèque récupérée en direct."""
    from src.models.item_based import get_item_recommendations

    games = get_owned_games(steam_id)
    owned = {game["appid"] for game in games}
    vector = pd.Series(0.0, index=matrice_interaction.columns)
    for game in games:
        if game["appid"] in vector.index:
            vector[game["appid"]] = convert_playtime_to_rating(
                game.get("playtime_forever", 0) / 60
            )
    if not (vector > 0).any():
        print("Aucun historique joué exploitable dans le catalogue.")
        return []
    return get_item_recommendations(
        steam_id,
        pd.DataFrame([vector], index=[steam_id]),
        df_sim_items,
        top_n,
        owned_games=owned,
    )
