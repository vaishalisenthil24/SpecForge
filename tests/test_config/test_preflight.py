import json
from unittest import mock

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.config import Config
from specforge.preflight import (
    PromptStats,
    load_prompt_stats,
    run_preflight,
)
from specforge.training.capture_contract import ServerCaptureContract

ALGORITHM = builtin_algorithm_registry().resolve("eagle3")
CONTRACT = ServerCaptureContract(
    method="eagle3",
    aux_layer_ids=(1, 2, 3),
    target_hidden_size=5120,
    target_vocab_size=128000,
    draft_vocab_size=128000,
)
GIB = 1 << 30


def make_config(tmp_path, **overrides):
    raw = {
        "model": {
            "target_model_path": "target",
            "draft_model_config": "draft",
            "vocab_mapping_path": "mapping.pt",
            "sglang_context_length": 262151,
        },
        "data": {
            "prompts_path": str(tmp_path / "prompts.jsonl"),
            "max_length": 262144,
        },
        "training": {"strategy": "eagle3", "max_steps": 1},
        "runtime": {
            "producer_lease": 8,
            "resident_high_watermark_bytes": 16 * GIB,
            "resident_low_watermark_bytes": 8 * GIB,
            "feature_store_max_resident_bytes": 24 * GIB,
        },
        "run_id": "preflight-test",
        "output_dir": str(tmp_path / "out"),
        "deployment": {
            "mode": "disaggregated",
            "trainer": {"nnodes": 1, "nproc_per_node": 1},
            "disaggregated": {
                "control_dir": str(tmp_path / "control"),
                "consumer_state_dir": str(tmp_path / "state"),
                "backend": "mooncake",
                "store_id": "preflight-test",
                "server_capture": "plugin",
                "capture_timeout_s": 3600,
                "managed_local": {
                    "trainer_cuda_visible_devices": ["4"],
                    "mooncake": {
                        "global_segment_size_bytes": 64 * GIB,
                        "local_buffer_size_bytes": 1 * GIB,
                    },
                    "capture_servers": [
                        {
                            "port": 30000,
                            "cuda_visible_devices": ["0", "1", "2", "3"],
                            "tp_size": 4,
                            "chunked_prefill_size": 8192,
                        }
                    ],
                },
            },
        },
    }
    raw["training"]["accumulation_steps"] = 2
    for path, value in overrides.items():
        node = raw
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return Config.model_validate(raw)


def stats(maximum=150_000, p95=140_000, count=10):
    return PromptStats(
        count=count, p50=120_000, p95=p95, maximum=maximum, source="test"
    )


def test_fitting_window_passes(tmp_path):
    report = run_preflight(
        make_config(tmp_path),
        ALGORITHM,
        contract=CONTRACT,
        prompt_stats=stats(),
        swa_pool_tokens=200_000,
    )
    assert report.ok, report.render()
    assert report.metrics["window_refs"] == 2
    assert report.metrics["window_bytes"] > 0


def test_window_larger_than_watermark_fails(tmp_path):
    cfg = make_config(
        tmp_path,
        **{
            "runtime.resident_high_watermark_bytes": 1 * GIB,
            "runtime.resident_low_watermark_bytes": GIB // 2,
        },
    )
    report = run_preflight(cfg, ALGORITHM, contract=CONTRACT, prompt_stats=stats())
    assert not report.ok
    assert any(c.name == "store-watermark" and c.level == "fail" for c in report.checks)


def test_mooncake_segment_smaller_than_watermark_fails(tmp_path):
    cfg = make_config(
        tmp_path,
        **{
            "runtime.resident_high_watermark_bytes": 16 * GIB,
            "deployment.disaggregated.managed_local.mooncake.global_segment_size_bytes": 8
            * GIB,
        },
    )
    report = run_preflight(cfg, ALGORITHM, contract=CONTRACT, prompt_stats=stats())
    assert not report.ok
    assert any(
        c.name == "mooncake-capacity" and c.level == "fail" for c in report.checks
    )


def test_capture_timeout_shorter_than_lease_fails(tmp_path):
    cfg = make_config(
        tmp_path,
        **{"deployment.disaggregated.capture_timeout_s": 200},
    )
    report = run_preflight(cfg, ALGORITHM, contract=CONTRACT, prompt_stats=stats())
    assert not report.ok
    assert any(c.name == "capture-timeout" and c.level == "fail" for c in report.checks)


def test_prompt_over_data_max_length_fails(tmp_path):
    cfg = make_config(tmp_path, **{"data.max_length": 100_000})
    report = run_preflight(
        cfg, ALGORITHM, contract=CONTRACT, prompt_stats=stats(maximum=150_000)
    )
    assert not report.ok
    assert any(c.name == "dataset-length" and c.level == "fail" for c in report.checks)


def test_whole_prompt_capture_over_pool_fails(tmp_path):
    cfg = make_config(
        tmp_path,
        **{
            "deployment.disaggregated.managed_local.capture_servers": [
                {
                    "port": 30000,
                    "cuda_visible_devices": ["0", "1", "2", "3"],
                    "tp_size": 4,
                    "chunked_prefill_size": -1,
                }
            ]
        },
    )
    report = run_preflight(
        cfg,
        ALGORITHM,
        contract=CONTRACT,
        prompt_stats=stats(),
        swa_pool_tokens=70_000,
    )
    assert not report.ok
    assert any(c.name == "swa-pool" and c.level == "fail" for c in report.checks)


def test_stale_run_directory_fails(tmp_path):
    (tmp_path / "control").mkdir()
    (tmp_path / "control" / "refs.jsonl").write_text("{}\n")
    report = run_preflight(
        make_config(tmp_path),
        ALGORITHM,
        contract=CONTRACT,
        prompt_stats=stats(),
    )
    assert not report.ok
    assert any(c.name == "run-paths" and c.level == "fail" for c in report.checks)


def test_load_prompt_stats_reads_input_ids_and_counts_raw_rows(tmp_path):
    tokenized = tmp_path / "tokenized.jsonl"
    tokenized.write_text(
        "\n".join(
            json.dumps({"input_ids": list(range(length))}) for length in (10, 20, 30)
        )
        + "\n"
    )
    got = load_prompt_stats(str(tokenized))
    assert (got.count, got.p50, got.p95, got.maximum) == (3, 20, 30, 30)

    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps({"conversations": [{"role": "user", "content": "hi"}]}) + "\n"
    )
    got = load_prompt_stats(str(raw))
    assert got.count == 1 and got.maximum is None


def test_cli_preflight_exit_code_and_contract_resolution(tmp_path):
    from specforge.cli import main

    cfg = make_config(tmp_path)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(cfg.model_dump(mode="json")))
    with mock.patch(
        "specforge.preflight.resolve_server_capture_contract",
        return_value=CONTRACT,
    ):
        assert (
            main(
                [
                    "preflight",
                    "-c",
                    str(path),
                    "--max-prompt-tokens",
                    "1000",
                    "--p95-prompt-tokens",
                    "900",
                    "--prompt-count",
                    "10",
                ]
            )
            == 0
        )


def test_cli_preflight_accepts_dimension_overrides(tmp_path):
    from specforge.cli import main

    path = tmp_path / "run.json"
    path.write_text(json.dumps(make_config(tmp_path).model_dump(mode="json")))
    assert (
        main(
            [
                "preflight",
                "-c",
                str(path),
                "--max-prompt-tokens",
                "1000",
                "--prompt-count",
                "10",
                "--target-hidden-size",
                "5120",
                "--aux-layer-ids",
                "1,2,3",
            ]
        )
        == 0
    )


def test_external_store_capacity_is_checked_when_declared(tmp_path):
    cfg = make_config(
        tmp_path,
        **{
            "deployment.disaggregated.managed_local": None,
            "deployment.disaggregated.server_urls": ["http://127.0.0.1:30000"],
        },
    )
    stats = PromptStats(count=2, p50=128, p95=256, maximum=512, source="test")

    failing = run_preflight(
        cfg,
        ALGORITHM,
        contract=CONTRACT,
        prompt_stats=stats,
        external_store_bytes=1,
    )
    assert failing.ok is False
    assert any(
        check.name == "mooncake-capacity" and check.level == "fail"
        for check in failing.checks
    )

    passing = run_preflight(
        cfg,
        ALGORITHM,
        contract=CONTRACT,
        prompt_stats=stats,
        external_store_bytes=cfg.runtime.resident_high_watermark_bytes,
    )
    assert passing.ok is True
    assert any(
        check.name == "mooncake-capacity" and check.level == "pass"
        for check in passing.checks
    )
