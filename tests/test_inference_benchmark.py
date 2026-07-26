import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.inference.common import (
    BenchmarkConfig,
    build_manifest,
    summarize_samples,
    validate_sample,
    load_humanml_test_captions,
    write_jsonl,
    read_jsonl,
)


class InferenceBenchmarkCommonTests(unittest.TestCase):
    def test_manifest_is_seeded_and_marks_warmups(self):
        captions = ["caption %02d" % i for i in range(40)]
        first = build_manifest(captions, [60, 120, 196], num_runs=20, warmups=2, seed=42)
        second = build_manifest(captions, [60, 120, 196], num_runs=20, warmups=2, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 66)
        self.assertEqual(sum(item["warmup"] for item in first), 6)
        self.assertEqual(sorted({item["frames"] for item in first}), [60, 120, 196])
        self.assertEqual(len({item["prompt_id"] for item in first}), 22)

    def test_summary_reports_mean_std_median_and_p95(self):
        samples = []
        for value in (10.0, 20.0, 30.0, 40.0):
            samples.append(
                {
                    "method": "demo",
                    "frames": 60,
                    "warmup": False,
                    "timings_ms": {
                        "text_ms": 1.0,
                        "retrieval_ms": 2.0,
                        "generation_ms": value,
                        "decode_ms": 3.0,
                        "e2e_ms": value + 6.0,
                    },
                }
            )
        summary = summarize_samples(samples)
        row = summary["groups"]["demo@60"]
        self.assertEqual(row["count"], 4)
        self.assertAlmostEqual(row["generation_ms"]["mean"], 25.0)
        self.assertAlmostEqual(row["generation_ms"]["median"], 25.0)
        self.assertGreater(row["generation_ms"]["p95"], 30.0)
        self.assertAlmostEqual(row["effective_fps"]["mean"], sum(60000.0 / value for value in (16, 26, 36, 46)) / 4.0)

    def test_validate_sample_rejects_wrong_motion_shape(self):
        sample = {
            "schema_version": 1,
            "method": "demo",
            "prompt_id": "p0",
            "frames": 60,
            "warmup": False,
            "output_shape": [59, 272],
            "timings_ms": {"e2e_ms": 10.0},
        }
        with self.assertRaisesRegex(ValueError, "output_shape"):
            validate_sample(sample)

    def test_config_round_trips_as_json(self):
        config = BenchmarkConfig(method="msa_t2m", frames=(60, 120, 196), seed=42)
        payload = json.loads(json.dumps(config.to_dict()))
        self.assertEqual(payload["method"], "msa_t2m")
        self.assertEqual(payload["frames"], [60, 120, 196])

    def test_humanml_caption_loader_and_jsonl_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "split").mkdir()
            (root / "texts").mkdir()
            (root / "split" / "test.txt").write_text("000001\n", encoding="utf-8")
            (root / "texts" / "000001.txt").write_text(
                "walk forward#0.0#1.0#0\nwalk forward#0.0#1.0#0\n", encoding="utf-8"
            )
            self.assertEqual(load_humanml_test_captions(str(root)), ["walk forward", "walk forward"])
            path = root / "rows.jsonl"
            rows = [{"x": 1}, {"x": "two"}]
            write_jsonl(str(path), rows)
            self.assertEqual(read_jsonl(str(path)), rows)


if __name__ == "__main__":
    unittest.main()
