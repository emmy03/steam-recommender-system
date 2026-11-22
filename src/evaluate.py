"""Comparaison hors ligne sur masquages aléatoires communs aux cinq moteurs.

Tous les jeux joués masqués sont pertinents (pertinence binaire).
Les jeux absents de l'entraînement restent dans le dénominateur du rappel,
mais ne sont proposés par aucun moteur. Aucun appel à Steam ni cache disque.
"""

import argparse
import hashlib
import json
import math
import platform
import random
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, SIMILARITY_METRIC
from src.models.data_utils import prepare_data
from src.models.hybrid import get_hybrid_recommendations
from src.models.item_based import build_item_similarity, get_item_recommendations
from src.models.popularity import build_popularity, get_popularity_recommendations
from src.models.svd_based import fit_svd, get_svd_recommendations
from src.models.user_based import build_user_similarity, get_user_recommendations

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_JEUX = 5
K = 10


def train_test_split_matrix(
    matrice_interaction,
    test_size=TEST_SIZE,
    min_jeux=MIN_JEUX,
    random_state=RANDOM_STATE,
):
    if not 0 < test_size < 1:
        raise ValueError("test_size doit être entre 0 et 1 exclus")
    if min_jeux < 2:
        raise ValueError("min_jeux doit être au moins 2")
    rng = random.Random(random_state)
    train = matrice_interaction.copy()
    hidden = {}
    for user in matrice_interaction.index:
        row = matrice_interaction.loc[user]
        games = list(row[row > 0].index)
        if len(games) < min_jeux:
            continue
        count = min(len(games) - 1, max(1, int(len(games) * test_size)))
        hidden[user] = set(rng.sample(games, count))
        train.loc[user, list(hidden[user])] = 0
    return train, hidden


def _top_ids(recos, k):
    if k < 1:
        raise ValueError("k doit être positif")
    ids = [game for game, _ in recos[:k]]
    if len(ids) != len(set(ids)):
        raise ValueError("Une recommandation contient des jeux dupliqués")
    return ids


def precision_recall_at_k(recos, jeux_masques_user, k=K):
    hits = len(set(_top_ids(recos, k)) & jeux_masques_user)
    return hits / k, hits / len(jeux_masques_user) if jeux_masques_user else 0.0


def ndcg_at_k(recos, relevant, k=K):
    ids = _top_ids(recos, k)
    dcg = sum(
        1 / math.log2(rank + 2) for rank, game in enumerate(ids) if game in relevant
    )
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def evaluer_recos_precalculees(
    nom_moteur, recos_par_user, jeux_masques, catalogue_total, k=K
):
    if k < 1:
        raise ValueError("k doit être positif")
    n = len(jeux_masques)
    totals = [0.0, 0.0, 0.0]
    recommended = set()
    served = 0
    catalog = set(catalogue_total)
    for user, relevant in jeux_masques.items():
        recos = recos_par_user.get(user, [])
        ids = _top_ids(recos, k)
        if not set(ids) <= catalog:
            raise ValueError("Jeu recommandé hors catalogue")
        precision, recall = precision_recall_at_k(recos, relevant, k)
        totals[0] += precision
        totals[1] += recall
        totals[2] += ndcg_at_k(recos, relevant, k)
        served += bool(ids)
        recommended.update(ids)
    return {
        "moteur": nom_moteur,
        f"precision@{k}": totals[0] / n if n else 0.0,
        f"recall@{k}": totals[1] / n if n else 0.0,
        f"ndcg@{k}": totals[2] / n if n else 0.0,
        "coverage": len(recommended) / len(catalog) if catalog else 0.0,
        "taux_utilisateurs_servis": served / n if n else 0.0,
        "n_utilisateurs_evalues": n,
    }


def evaluate_split(
    matrix,
    ownership,
    seed=42,
    k=10,
    test_size=0.2,
    min_games=5,
    metric="cosine",
    factors=20,
    max_users=None,
):
    train, hidden = train_test_split_matrix(matrix, test_size, min_games, seed)
    if max_users is not None:
        if max_users < 1:
            raise ValueError("max_users doit être positif")
        chosen = random.Random(seed).sample(list(hidden), min(max_users, len(hidden)))
        hidden = {user: hidden[user] for user in chosen}
    if not hidden:
        raise ValueError("Aucun utilisateur éligible")
    known = set(train.columns[(train > 0).any(axis=0)])
    unavailable = set(train.columns) - known
    users = build_user_similarity(train, metric=metric)
    items = build_item_similarity(train, metric=metric)
    popularity = build_popularity(train)
    predictions = fit_svd(train, factors, random_state=seed)
    recos = {
        name: {}
        for name in ("Popularité", "User-Based", "Item-Based", "Hybride", "SVD")
    }
    for index, (user, relevant) in enumerate(hidden.items(), 1):
        # Les achats masqués ne doivent pas fuiter dans le filtre d'exclusion.
        excluded = (ownership.get(user, set()) - relevant) | unavailable
        recos["Popularité"][user] = get_popularity_recommendations(
            user, train, popularity, k, excluded
        )
        recos["User-Based"][user] = get_user_recommendations(
            user, train, users, k, excluded
        )
        recos["Item-Based"][user] = get_item_recommendations(
            user, train, items, k, excluded
        )
        recos["Hybride"][user] = get_hybrid_recommendations(
            user, train, users, items, 0.5, k, excluded
        )
        recos["SVD"][user] = get_svd_recommendations(
            user, train, top_n=k, owned_games=excluded, predictions=predictions
        )
        if index % 25 == 0:
            print(f"  {index}/{len(hidden)} utilisateurs", flush=True)
    rows = []
    for name, recommendations in recos.items():
        result = evaluer_recos_precalculees(
            name, recommendations, hidden, matrix.columns, k
        )
        result.update(
            seed=seed,
            n_jeux_absents_train=len(unavailable),
            n_interactions_test=sum(map(len, hidden.values())),
            n_interactions_test_sans_signal=sum(
                len(games & unavailable) for games in hidden.values()
            ),
        )
        rows.append(result)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--output-dir", default="reports/evaluation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--min-games", type=int, default=MIN_JEUX)
    parser.add_argument(
        "--metric", choices=["cosine", "pearson"], default=SIMILARITY_METRIC
    )
    parser.add_argument("--factors", type=int, default=20)
    parser.add_argument(
        "--max-users",
        type=int,
        help="Échantillon d'utilisateurs évalués pour une vérification rapide",
    )
    args = parser.parse_args(argv)
    if args.k < 1 or args.factors < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error(
            "k et factors doivent être positifs ; les seeds doivent être distinctes"
        )
    matrix = prepare_data(args.data)
    raw = pd.read_csv(args.data)
    ownership = raw.groupby("user_id")["game_id"].agg(set).to_dict()
    rows = []
    for seed in args.seeds:
        print(f"Découpage seed={seed}", flush=True)
        rows.extend(
            evaluate_split(
                matrix,
                ownership,
                seed,
                args.k,
                args.test_size,
                args.min_games,
                args.metric,
                args.factors,
                args.max_users,
            )
        )
    detail = pd.DataFrame(rows)
    metrics = [
        f"precision@{args.k}",
        f"recall@{args.k}",
        f"ndcg@{args.k}",
        "coverage",
        "taux_utilisateurs_servis",
    ]
    summary = detail.groupby("moteur", sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary = summary.reset_index()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "resultats_par_decoupage.csv", index=False)
    summary.to_csv(output / "synthese.csv", index=False)
    metadata = vars(args) | {
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "versions": {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn")
        },
        "n_users": len(matrix),
        "n_games": len(matrix.columns),
        "protocol": "random interaction holdout; binary relevance; unknown-train items count as misses; owned unplayed items excluded when present in CSV; sample std across seeds; not a temporal or purchase prediction",
    }
    (output / "parametres.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Résultats enregistrés dans {output}")


if __name__ == "__main__":
    main()
