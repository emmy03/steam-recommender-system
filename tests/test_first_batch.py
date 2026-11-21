import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests

import main
from src.etl import extractor as api
from src.models.cold_start import get_cold_start_recommendations
from src.models.hybrid import get_hybrid_recommendations
from src.models.item_based import build_item_similarity, get_item_recommendations
from src.models.svd_based import get_svd_recommendations
from src.models.user_based import build_user_similarity, get_user_recommendations


class FirstBatchTests(unittest.TestCase):
    def setUp(self):
        self.matrix = pd.DataFrame(
            [[5.0, 0.0, 0.0], [5.0, 4.0, 3.0], [2.0, 3.0, 5.0]],
            index=[1, 2, 3],
            columns=[10, 20, 30],
        )

    def test_all_models_exclude_owned_unplayed(self):
        m = self.matrix
        users, items = build_user_similarity(m), build_item_similarity(m)
        results = [
            get_user_recommendations(1, m, users, 10, owned_games={10, 20}),
            get_item_recommendations(1, m, items, 10, owned_games={10, 20}),
            get_hybrid_recommendations(
                1, m, users, items, 0.5, 10, owned_games={10, 20}
            ),
            get_svd_recommendations(1, m, 2, 10, owned_games={10, 20}),
        ]
        for recos in results:
            self.assertEqual([g for g, _ in recos], [30])

    def test_new_user_excludes_zero_minutes(self):
        games = [
            {"appid": 10, "playtime_forever": 100},
            {"appid": 20, "playtime_forever": 0},
        ]
        with patch("src.models.cold_start.get_owned_games", return_value=games):
            recos = get_cold_start_recommendations(
                99, self.matrix, build_item_similarity(self.matrix), 10
            )
        self.assertEqual([g for g, _ in recos], [30])

    def test_invalid_metric_raises(self):
        for builder in (build_user_similarity, build_item_similarity):
            with self.assertRaises(ValueError):
                builder(self.matrix, metric="invalid")

    def test_cli_existing_user_all_choices(self):
        m = pd.DataFrame([[float((u + g) % 5) for g in range(25)] for u in range(22)])
        for choice in ("1", "2", "3", "4"):
            with self.subTest(choice=choice), contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(main, "MON_STEAM_ID", 0))
                stack.enter_context(patch.object(main, "prepare_data", return_value=m))
                stack.enter_context(
                    patch.object(main, "get_owned_games", return_value=[{"appid": 0}])
                )
                stack.enter_context(patch("builtins.input", side_effect=[choice, "1"]))
                cache = stack.enter_context(
                    patch.object(
                        main,
                        "get_or_compute",
                        side_effect=lambda name, path, cache_metric, fn, *a, **kw: fn(
                            *a, **kw
                        ),
                    )
                )
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                main.main()
                self.assertEqual(
                    cache.call_count, {"1": 1, "2": 1, "3": 2, "4": 0}[choice]
                )

    def test_retries_and_timeout(self):
        for failure in (
            Mock(status_code=429),
            Mock(status_code=503),
            requests.Timeout(),
        ):
            with (
                patch.object(
                    api.requests, "get", side_effect=[failure, Mock(status_code=200)]
                ) as get,
                patch.object(api.time, "sleep"),
            ):
                api._call_steam_api("https://example.test", {"key": "test"})
                self.assertEqual(get.call_count, 2)
                self.assertEqual(get.call_args.kwargs["timeout"], (5, 30))

    def test_api_errors_do_not_become_empty_library(self):
        for response in (Mock(status_code=500), Mock(status_code=403)):
            with (
                patch.object(api.requests, "get", return_value=response),
                patch.object(api.time, "sleep"),
                patch.object(api, "STEAM_API_KEY", "test"),
                self.assertRaises(api.SteamAPIError),
            ):
                api.get_owned_games(1)

    def test_empty_and_unavailable_are_distinct(self):
        with patch.object(api, "_call_steam_api") as call:
            call.return_value = Mock(status_code=200)
            call.return_value.json.return_value = {"response": {"game_count": 0}}
            self.assertEqual(api.get_owned_games(1), [])
            call.return_value.json.return_value = {"response": {}}
            with self.assertRaises(api.SteamProfileUnavailable):
                api.get_owned_games(1)
            call.return_value.json.side_effect = ValueError()
            with self.assertRaises(api.SteamAPIError):
                api.get_owned_games(1)

    def test_both_endpoints_use_common_client(self):
        with patch.object(api, "_call_steam_api") as call:
            call.return_value = Mock(status_code=200)
            call.return_value.json.return_value = {
                "friendslist": {"friends": []},
                "response": {"game_count": 0},
            }
            api.get_friends(1)
            api.get_owned_games(1)
            self.assertEqual(call.call_count, 2)
            self.assertTrue(
                all(c.args[0].startswith("https://") for c in call.call_args_list)
            )

    def test_extraction_keeps_unplayed_games(self):
        games = [
            {"appid": 10, "playtime_forever": 100},
            {"appid": 20, "playtime_forever": 0},
        ]
        with (
            patch.object(api, "get_owned_games", return_value=games),
            patch.object(api, "get_friends", return_value=[]),
            patch.object(api.time, "sleep"),
            patch.object(api.os, "makedirs"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            # Capture the exported frame without writing over project data.
            frames = []
            with patch.object(
                pd.DataFrame,
                "to_csv",
                autospec=True,
                side_effect=lambda frame, *a, **kw: frames.append(frame.copy()),
            ):
                api.run_extraction(1, 1)
            self.assertEqual(frames[0]["playtime_minutes"].tolist(), [100, 0])

    def test_missing_key_fails_before_network(self):
        with patch.object(api.requests, "get") as get:
            with self.assertRaises(api.SteamAPIError):
                api._call_steam_api("https://example.test", {"key": None})
            get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
