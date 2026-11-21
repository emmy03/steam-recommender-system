import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.config import DATA_PATH, MON_STEAM_ID, TOP_N
from src.models.data_utils import prepare_data


def get_svd_recommendations(
    user_id, matrice_interaction, n_factors=20, top_n=5, owned_games=None
):
    """
    Recommande des jeux en utilisant la factorisation de matrice (SVD).
    """
    if user_id not in matrice_interaction.index:
        return []

    # 1. On applique la SVD pour réduire la dimensionnalité (trouver les concepts cachés)
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    matrice_compressee = svd.fit_transform(matrice_interaction)

    # 2. On reconstruit une matrice "idéale" sans les trous (prédiction des notes manquantes)
    matrice_reconstruite = svd.inverse_transform(matrice_compressee)
    df_predictions = pd.DataFrame(
        matrice_reconstruite,
        index=matrice_interaction.index,
        columns=matrice_interaction.columns,
    )

    # 3. On isole les prédictions pour notre joueur
    mes_predictions = df_predictions.loc[user_id]
    mes_jeux_actuels = matrice_interaction.loc[user_id]

    # 4. On filtre les jeux déjà possédés et on trie
    excluded = set(owned_games or ())
    recos = {}
    for game_id, score_predit in mes_predictions.items():
        if (
            game_id not in excluded and mes_jeux_actuels[game_id] == 0
        ):  # Si je n'y ai pas joué
            recos[game_id] = score_predit

    top_jeux = sorted(recos.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_jeux


if __name__ == "__main__":
    matrice_inter = prepare_data(DATA_PATH)
    print("Calcul de la SVD en cours...")
    recos = get_svd_recommendations(
        MON_STEAM_ID, matrice_inter, n_factors=10, top_n=TOP_N
    )

    print("\nTON TOP 5 SVD (Découverte via Facteurs Latents) :")
    for game_id, score in recos:
        print(f"- AppID {game_id} | Score : {round(score, 2)}")
