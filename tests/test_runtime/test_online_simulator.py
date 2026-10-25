# coding=utf-8
"""CPU-only online-path simulator tests (no model, GPU, or Mooncake)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.config import Config
from specforge.simulate import (
    SimulatedFeatureStore,
    run_simulation,
)

ALGORITHM = builtin_algorithm_registry().resolve("eagle3")
KIB = 1 << 10


def make_config(tmp_path, *, max_resident_bytes=64 * KIB, high=32 * KIB, low=16 * KIB):
    tmp_path = Path(tmp_path)
    return Config.model_validate(
        {
            "model": {
                "target_model_path": "target",
                "draft_model_config": "draft",
                "vocab_mapping_path": "mapping.pt",
                "sglang_context_length": 4096,
            },
            "data": {
                "prompts_path": str(tmp_path / "prompts.jsonl"),
                "max_length": 2048,
            },
            "training": {
                "strategy": "eagle3",
                "max_steps": 1,
                "batch_size": 1,
                "accumulation_steps": 2,
            },
            "runtime": {
                "producer_lease": 2,
                "producer_concurrency": 2,
                "in_flight_high_watermark": 16,
                "in_flight_low_watermark": 8,
                "resident_high_watermark_bytes": high,
                "resident_low_watermark_bytes": low,
                "feature_store_max_resident_bytes": max_resident_bytes,
            },
            "run_id": "sim-test",
            "output_dir": str(tmp_path / "out"),
            "deployment": {
                "mode": "disaggregated",
                "trainer": {"nnodes": 1, "nproc_per_node": 1},
                "disaggregated": {
                    "control_dir": str(tmp_path / "control"),
                    "consumer_state_dir": str(tmp_path / "state"),
                    "backend": "mooncake",
                    "store_id": "sim-test",
                    "server_capture": "plugin",
                    "capture_timeout_s": 60,
                    "server_urls": ["http://127.0.0.1:30000"],
                },
            },
        }
    )


class TestOnlineSimulator(unittest.TestCase):
    def test_reports_exactly_once_and_stale_run_id_hazard(self):
        with tempfile.TemporaryDirectory() as work:
            cfg = make_config(work)
            report = run_simulation(
                cfg,
                ALGORITHM,
                work_dir=work,
                prompt_count=64,
                tokens_per_prompt=16,
                bytes_per_token=32,
                dispatch_refs=8,
                target_hidden_size=8,
                aux_layer_ids=(1,),
                timeout_s=30,
                consumer_delay_s=0.001,
            )
        self.assertTrue(report.ok, report.render())
        names = {check.name: check for check in report.checks}
        self.assertEqual(names["exactly-once"].level, "pass")
        self.assertEqual(names["store-capacity"].level, "pass")
        self.assertEqual(names["backpressure"].level, "pass")
        self.assertEqual(names["stale-run-id"].level, "pass")
        self.assertEqual(names["fresh-run-id"].level, "pass")
        self.assertGreater(report.metrics["stale_restart_duplicates"], 0)
        self.assertEqual(report.metrics["fresh_restart_duplicates"], 0)

    def test_feature_store_cap_fails_loudly(self):
        with tempfile.TemporaryDirectory() as work:
            cfg = make_config(
                work, max_resident_bytes=3 * KIB, high=2 * KIB, low=1 * KIB
            )
            report = run_simulation(
                cfg,
                ALGORITHM,
                work_dir=work,
                prompt_count=16,
                tokens_per_prompt=16,
                bytes_per_token=32,
                dispatch_refs=8,
                target_hidden_size=8,
                aux_layer_ids=(1,),
                timeout_s=30,
                check_restart=False,
            )
        self.assertFalse(report.ok)
        self.assertTrue(
            any(
                check.name == "store-capacity" and check.level == "fail"
                for check in report.checks
            ),
            report.render(),
        )

    def test_simulated_store_counts_bytes_without_allocating(self):
        store = SimulatedFeatureStore("sim", max_resident_bytes=100)
        ref = store.put(
            {},
            sample_id="a",
            metadata={"sim_bytes": 60, "task_id": "t", "strategy": "eagle3"},
        )
        self.assertEqual(ref.estimated_bytes, 60)
        self.assertEqual(store.health()["resident_bytes"], 60)
        with self.assertRaises(MemoryError):
            store.put(
                {},
                sample_id="b",
                metadata={"sim_bytes": 60, "task_id": "t2", "strategy": "eagle3"},
            )
        _features, handle = store.get(ref)
        store.release(handle)
        self.assertEqual(store.health()["resident_bytes"], 0)
