import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DATA_PATH, MON_STEAM_ID, SIMILARITY_METRIC, TOP_N
from src.models.data_utils import prepare_data


# --- FONCTIONS DU MODELE ---
def build_item_similarity(matrice_interaction, metric=SIMILARITY_METRIC, min_periods=5):
    """
    Calcule la matrice de similarité cosinus entre les jeux (Item-Based).

    Args:
        matrice_interaction (pd.DataFrame): La matrice d'interaction Utilisateur-Jeu.
        metric (str): "cosine" ou "pearson".
        min_periods (int): Nombre minimum d'observation communes pour Pearson.

    Returns:
        df_sim_items (pd.DataFrame): Une matrice carrée de similarité (lignes = jeux, colonnes = jeux).
    """
    if metric == "cosine":
        # 1. Transposition : les jeux deviennent les lignes (le coeur de l'Item-Based)
        matrice_items = matrice_interaction.T

        # 2. Calcul vectoriel ultra-rapide avec Scikit-Learn
        matrice_sim = cosine_similarity(matrice_items)

        # 3. Reformatage en DataFrame pour conserver les identifiants (game_id)
        df_sim_items = pd.DataFrame(
            matrice_sim, index=matrice_items.index, columns=matrice_items.index
        )
        return df_sim_items
    elif metric == "pearson":
        # On masque les 0 en NaN : la corrélation ne portera que sur les
        # joueurs ayant réellement noté les deux jeux comparés
        matrice_masquee = matrice_interaction.replace(0, np.nan)
        df_sim_items = matrice_masquee.corr(method="pearson", min_periods=min_periods)
        return df_sim_items.fillna(0.0)
    else:
        raise ValueError(f"Métrique non supportée : {metric}")


def get_item_recommendations(
    user_id, matrice_interaction, df_sim_items, top_n, owned_games=None
):
    """
    Génère des recommandations pour un utilisateur basé sur ses jeux joués
    et la similarité mathématique entre les items.

    Args:
        user_id (int): L'identifiant du joueur cible.
        matrice_interaction (pd.DataFrame): La matrice des temps de jeu.
        df_sim_items (pd.DataFrame): La matrice de similarité Item-Item.
        top_n (int): Le nombre de recommandations à retourner.

    Returns:
        list: Une liste de tuples (game_id, score_recommandation).
    """

    if user_id not in matrice_interaction.index:
            return []

    own = matrice_interaction.loc[user_id]
    played = own[own > 0]

    if played.empty:
        return []

    excluded = set(() if owned_games is None else owned_games)

    candidate_ids = df_sim_items.index
    candidate_ids = candidate_ids[
        (own.reindex(candidate_ids) == 0)
        & ~candidate_ids.isin(excluded)
    ]

    if candidate_ids.empty:
        return []

    similarities = df_sim_items.loc[
        candidate_ids, played.index
    ].to_numpy(dtype=float)

    scores = np.zeros(len(candidate_ids), dtype=float)

    # Conserver les similarités négatives de Pearson
    # et l'ordre historique des additions.
    for column, engagement in enumerate(played.to_numpy()):
        scores += similarities[:, column] * engagement

    order = np.argsort(-scores, kind="stable")[:top_n]

    return [
        (candidate_ids[i], float(scores[i]))
        for i in order
    ]


# --- POINT D'ENTREE (TEST LOCAL) ---
# Ce bloc permet de tester ce fichier indépendamment sans gêner les futurs imports
if __name__ == "__main__":
    print("Chargement des données...")
    matrice_inter = prepare_data(DATA_PATH)

    print(f"Calcul de la similarité Item-Item pour {matrice_inter.shape[1]} jeux...")
    matrice_sim = build_item_similarity(matrice_inter)
    print("Matrice générée avec succès !")

    print("\nRecherche de jeux similaires à tes jeux préférés...")
    recommandations = get_item_recommendations(
        MON_STEAM_ID, matrice_inter, matrice_sim, TOP_N
    )

    print(
        f"\nTon Top {len(recommandations)} des recommandations (Approche Item-Based) :"
    )
    for game_id, score in recommandations:
        print(f"- AppID {game_id} | Score de pertinence : {round(score, 2)}")
        print(f"  Lien : https://store.steampowered.com/app/{game_id}\n")
