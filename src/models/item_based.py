import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.config import DATA_PATH, MON_STEAM_ID, TOP_N
from src.models.data_utils import prepare_data


# --- FONCTIONS DU MODELE ---
def build_item_similarity(matrice_interaction):
    """
    Calcule la matrice de similarité cosinus entre les jeux (Item-Based).

    Args:
        matrice_interaction (pd.DataFrame): La matrice d'interaction Utilisateur-Jeu.

    Returns:
        df_sim_items (pd.DataFrame): Une matrice carrée de similarité (lignes = jeux, colonnes = jeux).
    """
    # 1. Transposition : les jeux deviennent les lignes (le coeur de l'Item-Based)
    matrice_items = matrice_interaction.T

    # 2. Calcul vectoriel ultra-rapide avec Scikit-Learn
    matrice_sim = cosine_similarity(matrice_items)

    # 3. Reformatage en DataFrame pour conserver les identifiants (game_id)
    df_sim_items = pd.DataFrame(
        matrice_sim, index=matrice_items.index, columns=matrice_items.index
    )
    return df_sim_items


def get_item_recommendations(user_id, matrice_interaction, df_sim_items, TOP_N):
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
    # Vérification de sécurité : le joueur existe-t-il dans notre dataset ?
    # if user_id not in matrice_interaction.index:
    # print("Utilisateur introuvable dans la base de données.")
    # return []

    mes_jeux = matrice_interaction.loc[user_id]
    mes_jeux_joues = mes_jeux[mes_jeux > 0]

    scores_recommandation = {}

    for game_id, playtime in mes_jeux_joues.items():
        # Extraction de la ligne de similarité pour le jeu en cours
        jeux_similaires = df_sim_items[game_id]

        for sim_game_id, score_sim in jeux_similaires.items():
            # Exclusion des jeux déjà joués et de la similarité du jeu avec lui-même
            if sim_game_id != game_id and mes_jeux[sim_game_id] == 0:
                if sim_game_id not in scores_recommandation:
                    scores_recommandation[sim_game_id] = 0

                # Pondération du score de similarité par le temps passé sur le jeu source
                scores_recommandation[sim_game_id] += score_sim * playtime

    # Tri décroissant selon le score final et sélection du Top N
    top_jeux = sorted(scores_recommandation.items(), key=lambda x: x[1], reverse=True)[
        :TOP_N
    ]
    return top_jeux


# --- POINT D'ENTREE (TEST LOCAL) ---
# Ce bloc permet de tester ce fichier indépendamment sans gêner les futurs imports
if __name__ == "__main__":
    print("Chargement des données...")
    matrice_inter = prepare_data(DATA_PATH)

    print(f"Calcul de la similarité Item-Item pour {matrice_inter.shape[1]} jeux...")
    matrice_sim = build_item_similarity(matrice_inter)
    print("Matrice générée avec succès !")

    print("\nRecherche de jeux similaires à tes jeux préférés...")
    recommandations = get_item_recommendations(MON_STEAM_ID, matrice_inter, matrice_sim)

    print(
        f"\nTon Top {len(recommandations)} des recommandations (Approche Item-Based) :"
    )
    for game_id, score in recommandations:
        print(f"- AppID {game_id} | Score de pertinence : {round(score, 2)}")
        print(f"  Lien : https://store.steampowered.com/app/{game_id}\n")
