import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.config import DATA_PATH, MON_STEAM_ID, TOP_N
from src.models.data_utils import prepare_data


def fit_svd(matrice_interaction, n_factors=20, random_state=42):
    """Entraîne une fois et renvoie les scores de tous les utilisateurs."""
    if n_factors < 1:
        raise ValueError("n_factors doit être positif")
    if min(matrice_interaction.shape) == 0:
        raise ValueError("La matrice est vide")
    if matrice_interaction.shape[1] == 1:
        return matrice_interaction.astype(float).copy()
    svd = TruncatedSVD(
        n_components=min(n_factors, *matrice_interaction.shape),
        random_state=random_state,
    )
    compressed = svd.fit_transform(matrice_interaction)
    return pd.DataFrame(
        svd.inverse_transform(compressed),
        index=matrice_interaction.index,
        columns=matrice_interaction.columns,
    )


def get_svd_recommendations(
    user_id,
    matrice_interaction,
    n_factors=20,
    top_n=5,
    owned_games=None,
    predictions=None,
):
    """Réutilise les prédictions préentraînées lorsqu'elles sont fournies."""
    if user_id not in matrice_interaction.index:
        return []
    df_predictions = (
        fit_svd(matrice_interaction, n_factors) if predictions is None else predictions
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
