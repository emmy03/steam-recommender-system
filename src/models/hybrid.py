from src.config import DATA_PATH, MON_STEAM_ID, TOP_N
from src.models.data_utils import prepare_data
from src.models.item_based import build_item_similarity, get_item_recommendations
from src.models.user_based import build_user_similarity, get_user_recommendations


def normalize_scores(reco_list):
    """
    Applique une normalisation Min-Max pour ramener les scores entre 0.0 et1.0.
    Cela permet de comparer équitablement deux algorithmes d'échelles différentes.
    """
    if not reco_list:
        return {}

    scores = [score for _, score in reco_list]
    max_score = max(scores)
    min_score = min(scores)

    # Sécurité anti-division par zéro si tous les scores sont identiques
    if max_score == min_score:
        return {game_id: 1.0 for game_id, _ in reco_list}

    return {
        game_id: (score - min_score) / (max_score - min_score)
        for game_id, score in reco_list
    }


def get_hybrid_recommendations(
    user_id, matrice_inter, sim_users, sim_items, weight_user, top_n, owned_games=None
):
    """
    Combine les algorithmes User-Based et Item-Based via une pondération normalisée.

    Args:
        user_id (int): L'ID du joueur.
        matrice_inter (pd.DataFrame): Matrice des temps de jeu.
        sim_users (pd.DataFrame): Matrice de similarité Joueur-Joueur.
        sim_items (pd.DataFrame): Matrice de similarité Jeu-Jeu.
        weight_user (float): Poids accordé au modèle User (entre 0.0 et 1.0).
        TOP_N (int): Nombre de jeux à recommander au final.
    """
    # 1. On récupère une liste large (Top 100) de chaque modèle pour maximiser les croisement
    recos_user = get_user_recommendations(
        user_id, matrice_inter, sim_users, top_n=100, owned_games=owned_games
    )
    recos_item = get_item_recommendations(
        user_id, matrice_inter, sim_items, top_n=100, owned_games=owned_games
    )

    # 2. On normalise les scores (de 0 à 1)
    norm_user = normalize_scores(recos_user)
    norm_item = normalize_scores(recos_item)

    hybrid_scores = {}

    # 3. On fusionne les deux listes de jeux (l'union des deux ensembles)
    all_games = set(norm_user.keys()).union(set(norm_item.keys()))

    # 4. Calcul du score pondéré
    for game_id in all_games:
        score_u = norm_user.get(game_id, 0.0)
        score_i = norm_item.get(game_id, 0.0)

        # La formule mathématique d'hybridation
        hybrid_scores[game_id] = (weight_user * score_u) + (
            (1.0 - weight_user) * score_i
        )

    # 5. Tri final et sélection du Top N
    top_jeux = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_jeux


# --- POINT D'ENTREE (TEST LOCAL) ---
if __name__ == "__main__":
    print("Chargement des données...")
    matrice_inter = prepare_data(DATA_PATH)

    print("Calcul des matrices de similarité...")
    sim_users = build_user_similarity(matrice_inter)
    sim_items = build_item_similarity(matrice_inter)

    print("\nLancement du Moteur Hybride (50% User / 50% Item)...")
    recos = get_hybrid_recommendations(
        MON_STEAM_ID, matrice_inter, sim_users, sim_items, weight_user=0.5, top_n=TOP_N
    )

    print("\nTON TOP 5 DES RECOMMANDATIONS HYBRIDES :")
    for game_id, score in recos:
        print(f"- AppID {game_id} | Score pondéré : {round(score, 3)}")
        print(f"  Lien : https://store.steampowered.com/app/{game_id}\n")
