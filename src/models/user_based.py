import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DATA_PATH, MON_STEAM_ID, SIMILARITY_METRIC, TOP_N
from src.models.data_utils import prepare_data


# --- FONCTIONS DU MODELE ---
def build_user_similarity(matrice_interaction, metric=SIMILARITY_METRIC, min_periods=5):
    """
    Calcule la matrice de similarité cosinus entre les joueurs (User-Based).

    Args:
        matrice_interaction (pd.DataFrame): La matrice d'interaction Utilisateur-Jeu.
        metric (str): "cosine" ou "pearson".
        min_periods (int): Nombre minimum d'observations communes pour Pearson.

    Returns:
        pd.DataFrame: Une matrice carrée de similarité (lignes = joueurs, colonnes = joueurs).
    """
    if metric == "cosine":
        # Calcul direct et optimisé sur les lignes (les joueurs)
        matrice_sim = cosine_similarity(matrice_interaction)

        # Reformatage en DataFrame pour conserver les identifiants (user_id)
        df_sim_users = pd.DataFrame(
            matrice_sim,
            index=matrice_interaction.index,
            columns=matrice_interaction.index,
        )
        return df_sim_users
    elif metric == "pearson":
        # .corr() calcule la corrélation entre les colonnes.
        # On transpose (.T) pour que les joueurs deviennent les colonnes.
        # Les 0 en NaN pour ne comparer que sur les jeux réellement communs.
        matrice_masquee = matrice_interaction.T.replace(0, np.nan)
        df_sim_users = matrice_masquee.corr(method="pearson", min_periods=min_periods)

        return df_sim_users.fillna(0.0)
    else:
        raise ValueError(f"Métrique non supportée : {metric}")


def get_user_recommendations(
    user_id, matrice_interaction, df_sim_users, top_n, owned_games=None
):
    """
    Génère des recommandations en se basant sur les bibliothèques des joueurs similaires.

    Args:
        user_id (int): L'identifiant du joueur cible.
        matrice_interaction (pd.DataFrame): La matrice des temps de jeu.
        df_sim_users (pd.DataFrame): La matrice de similarité Joueur-Joueur.
        top_n (int): Le nombre de recommandations à retourner.

    Returns:
        list: Une liste de tuples (game_id, score_recommandation).
    """
    if user_id not in matrice_interaction.index:
            return []

    neighbors = (
        df_sim_users[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
    )
    neighbors = neighbors[neighbors > 0]

    if neighbors.empty:
        return []

    own = matrice_interaction.loc[user_id]
    excluded = set(() if owned_games is None else owned_games)

    candidates = matrice_interaction.columns[
        (own <= 0) & ~own.index.isin(excluded)
    ]

    if candidates.empty:
        return []

    values = matrice_interaction.loc[
        neighbors.index, candidates
    ].to_numpy(dtype=float)

    positive = values > 0
    present = positive.any(axis=0)

    # Préserver l'ordre de première rencontre en cas d'égalité.
    order = np.argsort(positive.argmax(axis=0), kind="stable")
    order = order[present[order]]

    scores = np.zeros(len(candidates), dtype=float)

    # Conserver l'ordre des additions de la version précédente.
    for row, weight in zip(
        values, neighbors.to_numpy(), strict=True
    ):
        scores += np.where(row > 0, row, 0.0) * weight

    ranked = sorted(
        ((candidates[i], float(scores[i])) for i in order),
        key=lambda pair: pair[1],
        reverse=True,
    )

    return ranked[:top_n]


# --- POINT D'ENTREE (TEST LOCAL) ---
if __name__ == "__main__":
    print("Chargement des données via l'utilitaire partagé...")
    matrice_inter = prepare_data(DATA_PATH)

    print(
        f"Calcul de la similarité Joueur-Joueur pour {matrice_inter.shape[0]} profils..."
    )
    matrice_sim = build_user_similarity(matrice_inter)
    print("Matrice générée avec succès !")

    print("\nAnalyse des bibliothèques de tes voisins...")
    recommandations = get_user_recommendations(
        MON_STEAM_ID, matrice_inter, matrice_sim, TOP_N
    )

    print(
        f"\nTon Top {len(recommandations)} des recommandations (Approche User-Based) :"
    )
    for game_id, score in recommandations:
        print(f"- AppID {game_id} | Score de pertinence : {round(score, 2)}")
        print(f"  Lien : https://store.steampowered.com/app/{game_id}\n")
