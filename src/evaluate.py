"""
Évaluation quantitative des 4 moteurs de recommandation.

Méthode : leave-k-out par utilisateur.
Pour chaque utilisateur ayant joué à au moins MIN_JEUX jeux, on masque
(TEST_SIZE %) de ses jeux joués (on les remet à 0 dans une copie "train" de
la matrice) et on vérifie si le modèle parvient à les retrouver dans son
Top-K de recommandations calculées sur la matrice train.

Métriques :
- Precision@K : parmi les K jeux recommandés, quelle proportion étaient
  effectivement dans les jeux masqués (donc "pertinents") ?
- Recall@K    : parmi les jeux masqués, quelle proportion a été retrouvée
  dans le Top-K ?
- Coverage    : quelle proportion du catalogue total est recommandée au
  moins une fois, tous utilisateurs confondus (mesure de diversité —
  un modèle qui recommande toujours les 10 mêmes jeux "populaires" aura
  une precision correcte mais une coverage très faible).

Lancer avec : uv run -m src.evaluate
"""

import random
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.config import DATA_PATH
from src.models.data_utils import prepare_data
from src.models.user_based import build_user_similarity, get_user_recommendations
from src.models.item_based import build_item_similarity, get_item_recommendations
from src.models.hybrid import get_hybrid_recommendations

RANDOM_STATE = 42
TEST_SIZE = 0.2  # proportion des jeux masqués par utilisateur éligible
MIN_JEUX = 5  # utilisateurs avec moins de jeux exclus (pas assez de signal)
K = 10  # le K de Precision@K / Recall@K / coverage


# --- SPLIT TRAIN/TEST ---
def train_test_split_matrix(
    matrice_interaction,
    test_size=TEST_SIZE,
    min_jeux=MIN_JEUX,
    random_state=RANDOM_STATE,
):
    """
    Masque aléatoirement une partie des jeux joués de chaque utilisateur éligible.

    Returns:
        matrice_train (pd.DataFrame): Matrice avec les jeux masqués remis à 0.
        jeux_masques (dict): {user_id: set(game_id masqués)} -> la "vérité terrain".
    """
    random.seed(random_state)
    matrice_train = matrice_interaction.copy()
    jeux_masques = {}

    for user_id in matrice_interaction.index:
        jeux_joues = matrice_interaction.loc[user_id]
        jeux_joues = list(jeux_joues[jeux_joues > 0].index)

        if len(jeux_joues) < min_jeux:
            continue

        n_a_masquer = max(1, int(len(jeux_joues) * test_size))
        a_masquer = set(random.sample(jeux_joues, n_a_masquer))

        matrice_train.loc[user_id, list(a_masquer)] = 0
        jeux_masques[user_id] = a_masquer

    return matrice_train, jeux_masques


# --- METRIQUES ---
def precision_recall_at_k(recos, jeux_masques_user, k=K):
    """Precision@K et Recall@K pour un seul utilisateur."""
    top_k_ids = {game_id for game_id, _ in recos[:k]}
    vrais_positifs = top_k_ids & jeux_masques_user

    precision = len(vrais_positifs) / k if k > 0 else 0.0
    recall = len(vrais_positifs) / len(jeux_masques_user) if jeux_masques_user else 0.0
    return precision, recall


def evaluer_recos_precalculees(
    nom_moteur, recos_par_user, jeux_masques, catalogue_total, k=K
):
    """Calcule les métriques agrégées à partir de recommandations déjà calculées pour chaque utilisateur."""
    precisions, recalls = [], []
    jeux_recommandes_global = set()

    for user_id, masques in jeux_masques.items():
        recos = recos_par_user.get(user_id, [])
        if not recos:
            continue

        p, r = precision_recall_at_k(recos, masques, k=k)
        precisions.append(p)
        recalls.append(r)
        jeux_recommandes_global.update(game_id for game_id, _ in recos[:k])

    coverage = (
        len(jeux_recommandes_global) / len(catalogue_total)
        if len(catalogue_total)
        else 0.0
    )

    return {
        "moteur": nom_moteur,
        f"precision@{k}": round(sum(precisions) / len(precisions), 4)
        if precisions
        else 0.0,
        f"recall@{k}": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "coverage": round(coverage, 4),
        "n_utilisateurs_evalues": len(precisions),
    }


# --- POINT D'ENTREE ---
def main():
    print("Chargement des données...")
    matrice_inter = prepare_data(DATA_PATH)

    print(
        f"Split train/test (masquage de {int(TEST_SIZE * 100)}% des jeux, min {MIN_JEUX} jeux/utilisateur)..."
    )
    matrice_train, jeux_masques = train_test_split_matrix(matrice_inter)
    print(f"{len(jeux_masques)} utilisateurs éligibles pour l'évaluation.\n")

    print("Calcul des matrices de similarité sur le jeu d'ENTRAINEMENT uniquement...")
    sim_users = build_user_similarity(matrice_train)
    sim_items = build_item_similarity(matrice_train)

    # --- User-Based / Item-Based / Hybride : un appel par utilisateur ---
    recos_user, recos_item, recos_hybride = {}, {}, {}
    for user_id in jeux_masques:
        recos_user[user_id] = get_user_recommendations(
            user_id, matrice_train, sim_users, top_n=K
        )
        recos_item[user_id] = get_item_recommendations(
            user_id, matrice_train, sim_items, top_n=K
        )
        recos_hybride[user_id] = get_hybrid_recommendations(
            user_id, matrice_train, sim_users, sim_items, weight_user=0.5, top_n=K
        )

    # --- SVD : on fit UNE SEULE FOIS sur toute la matrice train (pas un fit par utilisateur) ---
    svd = TruncatedSVD(n_components=20, random_state=RANDOM_STATE)
    matrice_compressee = svd.fit_transform(matrice_train)
    matrice_reconstruite = svd.inverse_transform(matrice_compressee)
    df_predictions = pd.DataFrame(
        matrice_reconstruite, index=matrice_train.index, columns=matrice_train.columns
    )

    recos_svd = {}
    for user_id in jeux_masques:
        mes_jeux_actuels = matrice_train.loc[user_id]
        predictions_user = df_predictions.loc[user_id]
        recos_svd[user_id] = sorted(
            (
                (game_id, score)
                for game_id, score in predictions_user.items()
                if mes_jeux_actuels[game_id] == 0
            ),
            key=lambda x: x[1],
            reverse=True,
        )[:K]

    # --- Agrégation des résultats ---
    resultats = [
        evaluer_recos_precalculees(
            "User-Based", recos_user, jeux_masques, matrice_inter.columns
        ),
        evaluer_recos_precalculees(
            "Item-Based", recos_item, jeux_masques, matrice_inter.columns
        ),
        evaluer_recos_precalculees(
            "Hybride", recos_hybride, jeux_masques, matrice_inter.columns
        ),
        evaluer_recos_precalculees(
            "SVD", recos_svd, jeux_masques, matrice_inter.columns
        ),
    ]

    df_resultats = pd.DataFrame(resultats)
    print("\n=== COMPARATIF DES MOTEURS ===")
    print(df_resultats.to_string(index=False))

    df_resultats.to_csv("data/evaluation_resultats.csv", index=False)
    print("\nRésultats sauvegardés dans data/evaluation_resultats.csv")


if __name__ == "__main__":
    main()
