import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.config import DATA_PATH, MON_STEAM_ID, TOP_N
from src.models.data_utils import prepare_data

# --- FONCTIONS DU MODELE ---
def build_user_similarity(matrice_interaction):
    """
    Calcule la matrice de similarité cosinus entre les joueurs (User-Based).

    Args:
        matrice_interaction (pd.DataFrame): La matrice d'interaction Utilisateur-Jeu.

    Returns:
        pd.DataFrame: Une matrice carrée de similarité (lignes = joueurs, colonnes = joueurs).
    """
    # Calcul direct et optimisé sur les lignes (les joueurs)
    matrice_sim = cosine_similarity(matrice_interaction)

    # Reformatage en DataFrame pour conserver les identifiants (user_id)
    df_sim_users = pd.DataFrame(
        matrice_sim, index=matrice_interaction.index, columns=matrice_interaction.index
    )
    return df_sim_users


def get_user_recommendations(user_id, matrice_interaction, df_sim_users, TOP_N):
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
        print("Utilisateur introuvable dans la base de données.")
        return []

    # 1. Isoler les voisins (en s'excluant soi-même)
    proches = df_sim_users[user_id].drop(user_id).sort_values(ascending=False)
    voisins_utiles = proches[proches > 0]

    # 2. Identifier les jeux déjà possédés pour ne pas les recommander
    mes_jeux = matrice_interaction.loc[user_id]
    jeux_deja_joues = set(mes_jeux[mes_jeux > 0].index)

    scores_recommandation = {}

    # 3. Calculer les scores basés sur l'influence des voisins
    for voisin_id, score_similarite in voisins_utiles.items():
        jeux_du_voisin = matrice_interaction.loc[voisin_id]

        for game_id, playtime in jeux_du_voisin.items():
            if playtime > 0 and game_id not in jeux_deja_joues:
                if game_id not in scores_recommandation:
                    scores_recommandation[game_id] = 0

                scores_recommandation[game_id] += playtime * score_similarite

    top_jeux = sorted(scores_recommandation.items(), key=lambda x: x[1], reverse=True)[
        :TOP_N
    ]
    return top_jeux


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
