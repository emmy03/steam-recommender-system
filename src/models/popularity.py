"""Référence : nombre de joueurs ayant joué à chaque jeu dans l'entraînement."""


def build_popularity(matrice_train):
    return (matrice_train > 0).sum(axis=0).sort_values(ascending=False, kind="stable")


def get_popularity_recommendations(
    user_id, matrice_train, popularity, top_n=10, owned_games=None
):
    excluded = set(owned_games or ())
    if user_id in matrice_train.index:
        row = matrice_train.loc[user_id]
        excluded.update(row[row > 0].index)
    return [
        (game, float(score))
        for game, score in popularity.items()
        if score > 0 and game not in excluded
    ][:top_n]
