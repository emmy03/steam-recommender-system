import contextlib
import io
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.evaluate import (
    evaluate_split,
    evaluer_recos_precalculees,
    main,
    ndcg_at_k,
    precision_recall_at_k,
    train_test_split_matrix,
)
from src.models.popularity import build_popularity, get_popularity_recommendations
from src.models.svd_based import fit_svd, get_svd_recommendations


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.m = pd.DataFrame(
            [[5.0, 2.0, 0.0, 4.0], [3.0, 0.0, 2.0, 5.0], [0.0, 4.0, 3.0, 1.0]],
            index=[1, 2, 3],
            columns=[10, 20, 30, 40],
        )

    def test_metrics_known_example(self):
        recos = [(10, 3), (20, 2), (30, 1)]
        self.assertEqual(precision_recall_at_k(recos, {10, 30}, 3), (2 / 3, 1.0))
        self.assertAlmostEqual(
            ndcg_at_k(recos, {10, 30}, 3), 1.5 / (1 + 1 / math.log2(3))
        )
        self.assertEqual(ndcg_at_k([(30, 2), (10, 1)], {10, 30}, 3), 1.0)
        self.assertGreater(
            ndcg_at_k([(10, 2), (20, 1)], {10}, 2),
            ndcg_at_k([(20, 2), (10, 1)], {10}, 2),
        )

    def test_empty_predictions_count(self):
        result = evaluer_recos_precalculees(
            "test", {1: [(10, 1)]}, {1: {10}, 2: {20}}, [10, 20], 1
        )
        for key in (
            "precision@1",
            "recall@1",
            "ndcg@1",
            "taux_utilisateurs_servis",
            "coverage",
        ):
            self.assertEqual(result[key], 0.5)
        self.assertEqual(result["n_utilisateurs_evalues"], 2)
        self.assertEqual(precision_recall_at_k([], {10}, 2), (0.0, 0.0))
        self.assertEqual(precision_recall_at_k([(10, 1)], {10}, 10), (0.1, 1.0))

    def test_invalid_rankings_rejected(self):
        with self.assertRaises(ValueError):
            ndcg_at_k([(10, 2), (10, 1)], {10}, 2)
        with self.assertRaises(ValueError):
            precision_recall_at_k([], set(), 0)

    def test_split_reproducible_and_no_global_rng_side_effect(self):
        before = self.m.copy()
        state = random.getstate()
        train, hidden = train_test_split_matrix(self.m, 0.9, 2, 42)
        train2, hidden2 = train_test_split_matrix(self.m, 0.9, 2, 42)
        pd.testing.assert_frame_equal(train, train2)
        pd.testing.assert_frame_equal(self.m, before)
        self.assertEqual(hidden, hidden2)
        self.assertEqual(random.getstate(), state)
        for user, games in hidden.items():
            self.assertTrue((train.loc[user, list(games)] == 0).all())
            self.assertGreater((train.loc[user] > 0).sum(), 0)
            self.assertEqual(
                set(train.columns[train.loc[user] > 0]) | games,
                set(self.m.columns[self.m.loc[user] > 0]),
            )

    def test_popularity_train_only_and_exclusion(self):
        train = pd.DataFrame([[1, 0, 0], [1, 1, 0]], columns=[10, 20, 30])
        pop = build_popularity(train)
        self.assertEqual(pop.to_dict(), {10: 2, 20: 1, 30: 0})
        self.assertEqual(get_popularity_recommendations(0, train, pop, 10), [(20, 1.0)])
        self.assertEqual(get_popularity_recommendations(0, train, pop, 10, {20}), [])

    def test_svd_shared_pretrained_predictions(self):
        scores = fit_svd(self.m, 2)
        direct = get_svd_recommendations(1, self.m, 2, 3)
        with patch(
            "src.models.svd_based.fit_svd", side_effect=AssertionError("unexpected fit")
        ):
            cached = get_svd_recommendations(1, self.m, 2, 3, predictions=scores)
        self.assertEqual(direct, cached)

    def test_pipeline_excludes_owned_unplayed_but_not_hidden(self):
        ownership = {user: set(self.m.columns) for user in self.m.index}
        _, hidden = train_test_split_matrix(self.m, 0.2, 2, 42)
        # All columns have training signal in this fixture.
        with patch(
            "src.evaluate.get_user_recommendations", return_value=[]
        ) as recommend:
            results = evaluate_split(self.m, ownership, min_games=2, factors=2, k=2)
        for call in recommend.call_args_list:
            user, train, _, _, excluded = call.args
            self.assertEqual(excluded, ownership[user] - hidden[user])
            self.assertTrue((train.loc[user, list(hidden[user])] == 0).all())
        self.assertEqual(len(results), 5)
        self.assertTrue(all(row["n_utilisateurs_evalues"] == 3 for row in results))
        user_result = next(row for row in results if row["moteur"] == "User-Based")
        self.assertEqual(user_result["precision@2"], 0.0)

    def test_absent_training_games_remain_in_recall_denominator(self):
        matrix = pd.DataFrame([[1.0, 1.0, 1.0]], index=[1], columns=[10, 20, 30])
        with patch(
            "src.evaluate.get_user_recommendations", return_value=[]
        ) as recommend:
            results = evaluate_split(
                matrix, {1: {10, 20, 30}}, min_games=2, factors=1, k=2
            )
        self.assertEqual(set(recommend.call_args.args[-1]), {10, 20, 30})
        for row in results:
            self.assertEqual(row["n_interactions_test_sans_signal"], 1)
            self.assertEqual(row["recall@2"], 0.0)
            self.assertEqual(row["n_utilisateurs_evalues"], 1)

    def test_cli_exports_three_files_and_common_seeds(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(tmp)
            csv = root / "input.csv"
            rows = [
                (u, g, float(v) * 60)
                for u, row in self.m.iterrows()
                for g, v in row.items()
            ]
            pd.DataFrame(
                rows, columns=["user_id", "game_id", "playtime_minutes"]
            ).to_csv(csv, index=False)
            main(
                [
                    "--data",
                    str(csv),
                    "--output-dir",
                    str(root / "out"),
                    "--seeds",
                    "42",
                    "43",
                    "--min-games",
                    "2",
                    "--factors",
                    "2",
                    "--k",
                    "2",
                ]
            )
            detail = pd.read_csv(root / "out/resultats_par_decoupage.csv")
            self.assertEqual(len(detail), 10)
            self.assertTrue((detail.groupby("seed").size() == 5).all())
            self.assertTrue((root / "out/synthese.csv").is_file())
            self.assertTrue((root / "out/parametres.json").is_file())


if __name__ == "__main__":
    unittest.main()
