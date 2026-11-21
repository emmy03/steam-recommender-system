from src.config import DATA_PATH, MON_STEAM_ID, TOP_N
from src.etl.extractor import SteamAPIError, get_owned_games
from src.models.cold_start import get_cold_start_recommendations
from src.models.data_utils import prepare_data
from src.models.hybrid import get_hybrid_recommendations
from src.models.item_based import build_item_similarity, get_item_recommendations
from src.models.svd_based import get_svd_recommendations
from src.models.user_based import build_user_similarity, get_user_recommendations
from src.utils.cache_utils import get_or_compute


def _demander_metrique():
    """Demande à l'utilisateur la métrique à utiliser et la retourne."""
    print("\nQuelle formule mathématique souhaitez-vous utiliser pour cet algorithme ?")
    print("1. Cosinus (Classique, idéal pour les bibliothèques creuses)")
    print("2. Pearson (Corrélation fine, centrée sur la moyenne)")
    choix_metric = input("Entrez votre choix (1 ou 2) : ")
    metric = "pearson" if choix_metric == "2" else "cosine"
    print(f"\nCalcul/Chargement des matrices avec la méthode '{metric}'...")
    return metric


def main():
    print("===================================================")
    print(" BIENVENUE DANS LE STEAM RECOMMENDER SYSTEM ")
    print("===================================================\n")

    print("Initialisation du moteur et chargement des données...")
    matrice_inter = prepare_data(DATA_PATH)
    print(f"\nProfil cible sélectionné : {MON_STEAM_ID}")

    # --- Gestion du cold start ---
    if MON_STEAM_ID not in matrice_inter.index:
        print("\nCe profil n'est pas dans le dataset d'entraînement.")
        print("Seule l'approche Item-Based est disponible pour les nouveaux profils.")
        metric = _demander_metrique()
        sim_items = get_or_compute(
            "sim_items",
            DATA_PATH,
            metric,
            build_item_similarity,
            matrice_inter,
            metric=metric,
        )
        print("Récupération de la bibliothèque du joueur et calcul à la volée...")
        recos = get_cold_start_recommendations(
            MON_STEAM_ID, matrice_inter, sim_items, TOP_N
        )
        _afficher_resultats(recos)
        return

    print("\nQuel algorithme veux-tu tester aujourd'hui ?")
    print("1. Moteur User-Based (Découvertes inattendues via des joueurs similaires)")
    print("2. Moteur Item-Based (Précision mathématique basée sur les jeux)")
    print("3. Moteur Hybride (Mix User/Item)")
    print("4. Moteur SVD (Factorisation par SVD tronquée)")

    choix = input("\nEntre ton choix (1, 2, 3 ou 4) : ")

    if choix not in {"1", "2", "3", "4"}:
        print("Choix invalide.")
        return
    # La bibliothèque en direct garantit l'exclusion des jeux possédés,
    # y compris avec l'ancien CSV qui ne conservait pas les jeux à zéro minute.
    owned_games = {game["appid"] for game in get_owned_games(MON_STEAM_ID)}
    if choix in {"1", "2", "3"}:
        metric = _demander_metrique()
    if choix in {"1", "3"}:
        sim_users = get_or_compute(
            "sim_users",
            DATA_PATH,
            metric,
            build_user_similarity,
            matrice_inter,
            metric=metric,
        )
    if choix in {"2", "3"}:
        sim_items = get_or_compute(
            "sim_items",
            DATA_PATH,
            metric,
            build_item_similarity,
            matrice_inter,
            metric=metric,
        )

    if choix == "1":
        print("\nLancement du moteur User-Based...")
        recos = get_user_recommendations(
            MON_STEAM_ID, matrice_inter, sim_users, TOP_N, owned_games=owned_games
        )

    elif choix == "2":
        print("\nLancement du moteur Item-Based...")
        recos = get_item_recommendations(
            MON_STEAM_ID, matrice_inter, sim_items, TOP_N, owned_games=owned_games
        )

    elif choix == "3":
        print("Lancement du moteur hybrid...")
        recos = get_hybrid_recommendations(
            MON_STEAM_ID,
            matrice_inter,
            sim_users,
            sim_items,
            weight_user=0.5,
            top_n=TOP_N,
            owned_games=owned_games,
        )

    elif choix == "4":
        print("\nLancement du moteur SVD (Réduction de dimensionnalité)...")
        recos = get_svd_recommendations(
            MON_STEAM_ID,
            matrice_inter,
            n_factors=20,
            top_n=TOP_N,
            owned_games=owned_games,
        )

    else:
        print("Choix invalide. Relance le script.")
        return

    _afficher_resultats(recos)


def _afficher_resultats(recos):
    """Affichage commun des résultats, factorisé pour être réutilisé par le cold start."""
    if recos:
        print(f"\nTON TOP {len(recos)} DES RECOMMANDATIONS :")
        for game_id, score in recos:
            lien = f"https://store.steampowered.com/app/{game_id}"
            print(f"- AppID {game_id} | Score : {round(score, 2)} | Lien : {lien}")
    else:
        print("\nAucune recommandation n'a pu être générée pour ce profil.")


if __name__ == "__main__":
    try:
        main()
    except SteamAPIError as exc:
        print(f"Impossible de recommander : {exc}")
