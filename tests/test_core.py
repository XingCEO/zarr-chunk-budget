import unittest
from zarr_chunk_budget import analyze


class ChunkTests(unittest.TestCase):
    def test_edge_chunk_not_full_size(self):
        r = analyze(
            {
                "shape": [11],
                "item_bytes": 1,
                "candidates": [[10]],
                "queries": [{"start": [10], "stop": [11]}],
            }
        )
        self.assertEqual(r["candidates"][0]["weighted_fetched_bytes"], 1)

    def test_empty_slice(self):
        r = analyze(
            {
                "shape": [10],
                "item_bytes": 4,
                "candidates": [[3]],
                "queries": [{"start": [2], "stop": [2]}],
            }
        )
        self.assertEqual(r["candidates"][0]["weighted_requests"], 0)

    def test_unaligned_crossing(self):
        r = analyze(
            {
                "shape": [10],
                "item_bytes": 1,
                "candidates": [[5]],
                "queries": [{"start": [4], "stop": [6]}],
            }
        )
        self.assertEqual(r["candidates"][0]["weighted_requests"], 2)
        self.assertEqual(r["candidates"][0]["read_amplification"], 5)

    def test_oob_rejected(self):
        with self.assertRaises(ValueError):
            analyze(
                {
                    "shape": [10],
                    "item_bytes": 1,
                    "candidates": [[5]],
                    "queries": [{"start": [0], "stop": [11]}],
                }
            )

    def test_weighted_zero(self):
        self.assertIsNone(
            analyze(
                {"shape": [10], "item_bytes": 1, "candidates": [[5]], "queries": []}
            )["candidates"][0]["read_amplification"]
        )
