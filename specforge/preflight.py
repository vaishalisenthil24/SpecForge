# coding=utf-8
"""CPU-only launch checks for a typed SpecForge run config.

The preflight catches failures that otherwise surface only after a GPU box is
warm: a capture request that cannot finish inside its timeout, a consumer
window that cannot fit the feature-store watermark, a prompt longer than the
engine context, or a reused run directory whose stale Mooncake keys poison
deduplication.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from specforge.algorithms.registry import AlgorithmRegistration
from specforge.config import Config
from specforge.inference.adapters.server_capture import resolve_capture_timeout_s
from specforge.runtime.contracts import consumer_dispatch_quantum
from specforge.training.capture_contract import (
    ServerCaptureContract,
    resolve_server_capture_contract,
)

GIB = 1 << 30


@dataclass(frozen=True)
class Check:
    level: str
    name: str
    message: str


@dataclass(frozen=True)
class PromptStats:
    count: int
    p50: Optional[int] = None
    p95: Optional[int] = None
    maximum: Optional[int] = None
    source: str = ""


@dataclass
class PreflightReport:
    checks: list[Check] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def add(self, level: str, name: str, message: str) -> None:
        self.checks.append(Check(level, name, message))

    def pass_(self, name: str, message: str) -> None:
        self.add("pass", name, message)

    def warn(self, name: str, message: str) -> None:
        self.add("warn", name, message)

    def fail(self, name: str, message: str) -> None:
        self.add("fail", name, message)

    @property
    def failures(self) -> int:
        return sum(check.level == "fail" for check in self.checks)

    @property
    def warnings(self) -> int:
        return sum(check.level == "warn" for check in self.checks)

    @property
    def ok(self) -> bool:
        return self.failures == 0

    def render(self) -> str:
        headline = "PASS" if self.ok else "FAIL"
        lines = [
            (
                f"SPECFORGE PREFLIGHT: {headline} "
                f"({self.failures} failures, {self.warnings} warnings)"
            )
        ]
        for check in self.checks:
            lines.append(f"[{check.level.upper()}] {check.name}: {check.message}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "pass" if self.ok else "fail",
            "failures": self.failures,
            "warnings": self.warnings,
            "checks": [
                {"level": check.level, "name": check.name, "message": check.message}
                for check in self.checks
            ],
            "metrics": self.metrics,
        }


def _nearest_rank(sorted_values: list[int], percentile: float) -> Optional[int]:
    if not sorted_values:
        return None
    rank = max(1, math.ceil(percentile * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def _stats_from_lengths(lengths: Iterable[int], source: str) -> PromptStats:
    values = sorted(int(value) for value in lengths)
    return PromptStats(
        count=len(values),
        p50=_nearest_rank(values, 0.50),
        p95=_nearest_rank(values, 0.95),
        maximum=values[-1] if values else None,
        source=source,
    )


def _stats_from_mapping(payload: Mapping[str, Any], source: str) -> PromptStats:
    lengths = (
        payload.get("token_lengths")
        or payload.get("prompt_token_lengths")
        or payload.get("lengths")
    )
    if lengths is not None:
        return _stats_from_lengths(lengths, source)
    count = payload.get("count")
    if count is None:
        raise ValueError(f"{source} must contain token_lengths or count/p50/p95/max")
    return PromptStats(
        count=int(count),
        p50=payload.get("p50", payload.get("median")),
        p95=payload.get("p95"),
        maximum=payload.get("max", payload.get("maximum")),
        source=source,
    )


def load_prompt_stats(path: str, *, max_rows: int = 0) -> PromptStats:
    """Read token lengths from a pre-tokenized JSONL or a JSON stats file.

    Raw conversation JSONL is counted but has no tokenizer-free length fields;
    callers should pass explicit lengths in that case.
    """
    source = os.path.expanduser(path)
    target = Path(source)
    if not target.exists():
        raise FileNotFoundError(f"prompt data not found: {source}")
    if target.suffix == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return _stats_from_lengths(payload, source)
        if isinstance(payload, dict):
            return _stats_from_mapping(payload, source)
        raise ValueError(f"{source} must be a JSON object or list of lengths")

    lengths: list[int] = []
    count = 0
    token_key: Optional[str] = None
    with target.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            count += 1
            if max_rows and count > max_rows:
                count -= 1
                break
            record = json.loads(line)
            if token_key is None:
                token_key = next(
                    (
                        key
                        for key in ("input_ids", "prompt_token_ids", "token_ids")
                        if isinstance(record.get(key), list)
                    ),
                    "",
                )
            if token_key:
                values = record.get(token_key)
                if not isinstance(values, list):
                    raise ValueError(f"{source} row {count} is missing {token_key!r}")
                lengths.append(len(values))
    if not token_key:
        return PromptStats(count=count, source=f"{source} (raw text; no token ids)")
    return _stats_from_lengths(lengths, source)


def merge_prompt_stats(
    base: Optional[PromptStats],
    *,
    count: Optional[int] = None,
    p50: Optional[int] = None,
    p95: Optional[int] = None,
    maximum: Optional[int] = None,
    source: str = "command line",
) -> Optional[PromptStats]:
    if (
        base is None
        and count is None
        and p50 is None
        and p95 is None
        and maximum is None
    ):
        return None
    if count is None and p50 is None and p95 is None and maximum is None:
        return base
    return PromptStats(
        count=count if count is not None else (base.count if base else 0),
        p50=p50 if p50 is not None else (base.p50 if base else None),
        p95=p95 if p95 is not None else (base.p95 if base else None),
        maximum=maximum if maximum is not None else (base.maximum if base else None),
        source=(base.source if base is not None else source),
    )


def _capture_metrics(
    cfg: Config, algorithm: AlgorithmRegistration, contract: ServerCaptureContract
) -> dict[str, Any]:
    from specforge.algorithms.common.providers import resolve_server_capture_layout

    layout = resolve_server_capture_layout(
        algorithm,
        cfg,
        modality=cfg.model.input_modality,
    )
    element_size = 4 if cfg.model.torch_dtype == "float32" else 2
    width = len(contract.aux_layer_ids) * contract.target_hidden_size
    if layout.last_hidden_feature is not None:
        width += contract.target_hidden_size
    passthrough_bytes = 16
    return {
        "bytes_per_token": width * element_size + passthrough_bytes,
        "feature_width": width,
        "element_size": element_size,
        "aux_feature": layout.aux_feature,
        "last_hidden_feature": layout.last_hidden_feature,
    }


def _nonempty_path(path: Optional[str]) -> bool:
    if not path:
        return False
    target = Path(os.path.expanduser(path))
    if not target.exists():
        return False
    if target.is_file():
        return target.stat().st_size > 0
    return any(target.iterdir())


def _window_refs(cfg: Config) -> int:
    trainer = cfg.deployment.trainer
    per_optimizer_step = (
        trainer.nnodes
        * trainer.nproc_per_node
        * cfg.training.batch_size
        * cfg.training.accumulation_steps
    )
    return consumer_dispatch_quantum(cfg.runtime.consumer_dispatch, per_optimizer_step)


def run_preflight(
    cfg: Config,
    algorithm: AlgorithmRegistration,
    *,
    contract: Optional[ServerCaptureContract] = None,
    prompt_stats: Optional[PromptStats] = None,
    capture_tps: float = 5000.0,
    swa_pool_tokens: Optional[int] = None,
    allow_existing: bool = False,
    data_path: Optional[str] = None,
    max_rows: int = 0,
    external_store_bytes: Optional[int] = None,
) -> PreflightReport:
    """Evaluate one config without importing a model or touching a GPU."""
    report = PreflightReport()
    report.metrics["mode"] = cfg.mode
    report.metrics["run_id"] = cfg.run_id

    report.pass_("config", f"validated {cfg.mode} run {cfg.run_id!r}")
    if cfg.mode != "online":
        report.pass_("topology", "offline config; online capture checks skipped")
        _check_paths(cfg, report, allow_existing=allow_existing)
        return report

    deployment = cfg.deployment.disaggregated
    if deployment is None:
        report.fail("topology", "online config has no disaggregated deployment")
        return report
    if not deployment.managed_local and not deployment.server_urls:
        report.fail(
            "servers",
            "online external capture needs deployment.disaggregated.server_urls",
        )
    else:
        report.pass_("servers", "capture server topology is configured")

    if contract is None:
        try:
            contract = resolve_server_capture_contract(cfg, algorithm=algorithm)
        except Exception as exc:
            report.fail(
                "capture-contract", f"could not resolve model dimensions: {exc}"
            )
            _check_paths(cfg, report, allow_existing=allow_existing)
            return report
    report.metrics["aux_layer_ids"] = list(contract.aux_layer_ids)
    report.metrics["target_hidden_size"] = contract.target_hidden_size
    report.pass_(
        "capture-contract",
        f"capture layers {list(contract.aux_layer_ids)} at hidden "
        f"{contract.target_hidden_size}",
    )

    if prompt_stats is None and data_path:
        try:
            prompt_stats = load_prompt_stats(data_path, max_rows=max_rows)
        except Exception as exc:
            report.fail("dataset", str(exc))
    if prompt_stats is not None:
        report.metrics["prompt_count"] = prompt_stats.count
        report.metrics["prompt_p50_tokens"] = prompt_stats.p50
        report.metrics["prompt_p95_tokens"] = prompt_stats.p95
        report.metrics["prompt_max_tokens"] = prompt_stats.maximum
        if prompt_stats.count <= 0:
            report.fail("dataset", f"no prompts found in {prompt_stats.source}")
        elif prompt_stats.maximum is None:
            report.warn(
                "dataset",
                f"{prompt_stats.count} prompts in {prompt_stats.source}; token "
                "lengths unavailable, so size and timeout checks are skipped",
            )
        else:
            report.pass_(
                "dataset",
                f"{prompt_stats.count} prompts, p50 {prompt_stats.p50}, p95 "
                f"{prompt_stats.p95}, max {prompt_stats.maximum} tokens "
                f"({prompt_stats.source})",
            )
            if prompt_stats.maximum > cfg.data.max_length:
                report.fail(
                    "dataset-length",
                    f"max prompt {prompt_stats.maximum} > data.max_length "
                    f"{cfg.data.max_length}; prompts would be truncated",
                )
            else:
                report.pass_(
                    "dataset-length",
                    f"max prompt {prompt_stats.maximum} <= data.max_length "
                    f"{cfg.data.max_length}",
                )
            if cfg.model.sglang_context_length is not None:
                context_limit = cfg.model.sglang_context_length - 7
                if prompt_stats.maximum > context_limit:
                    report.fail(
                        "context-length",
                        f"max prompt {prompt_stats.maximum} > capture context "
                        f"limit {context_limit}",
                    )
                else:
                    report.pass_(
                        "context-length",
                        f"max prompt {prompt_stats.maximum} <= capture limit "
                        f"{context_limit}",
                    )
    else:
        report.warn(
            "dataset",
            "no prompt data or length stats supplied; size and timeout checks "
            "are skipped",
        )

    metrics = _capture_metrics(cfg, algorithm, contract)
    report.metrics.update(metrics)
    refs = _window_refs(cfg)
    report.metrics["window_refs"] = refs
    report.pass_(
        "window",
        f"{refs} refs must be resident before an optimizer step "
        f"(policy {cfg.runtime.consumer_dispatch})",
    )

    window_bytes = None
    max_tokens = prompt_stats.maximum if prompt_stats else None
    if max_tokens is not None:
        capacity_tokens = prompt_stats.p95 or prompt_stats.p50 or max_tokens
        max_sample_bytes = max_tokens * int(metrics["bytes_per_token"])
        capacity_sample_bytes = capacity_tokens * int(metrics["bytes_per_token"])
        window_bytes = capacity_sample_bytes * refs
        report.metrics["max_sample_bytes"] = max_sample_bytes
        report.metrics["capacity_sample_bytes"] = capacity_sample_bytes
        report.metrics["window_bytes"] = window_bytes
        report.metrics["window_bytes_max"] = max_sample_bytes * refs
        report.pass_(
            "capture-size",
            f"{metrics['bytes_per_token']} bytes/token; p95 sample "
            f"{capacity_sample_bytes / GIB:.2f} GiB; window "
            f"{window_bytes / GIB:.2f} GiB (max-case "
            f"{max_sample_bytes * refs / GIB:.2f} GiB)",
        )
        _check_store(
            cfg,
            deployment,
            report,
            window_bytes,
            external_store_bytes=external_store_bytes,
        )

    _check_capture_timeout(
        cfg,
        deployment,
        report,
        prompt_stats=prompt_stats,
        capture_tps=capture_tps,
    )
    _check_swa(
        cfg,
        deployment,
        report,
        max_tokens=max_tokens,
        swa_pool_tokens=swa_pool_tokens,
    )
    _check_paths(cfg, report, allow_existing=allow_existing)
    return report


def _check_store(
    cfg: Config,
    deployment,
    report: PreflightReport,
    window_bytes: int,
    *,
    external_store_bytes: Optional[int] = None,
) -> None:
    high = cfg.runtime.resident_high_watermark_bytes
    if high is None:
        report.warn(
            "store-watermark",
            f"window needs {window_bytes / GIB:.2f} GiB, but "
            "runtime.resident_high_watermark_bytes is unset; capacity cannot "
            "be enforced",
        )
        return
    if high < window_bytes:
        report.fail(
            "store-watermark",
            f"window needs {window_bytes / GIB:.2f} GiB > resident high "
            f"watermark {high / GIB:.2f} GiB; the producer stalls before a "
            "full dispatch window",
        )
    else:
        report.pass_(
            "store-watermark",
            f"window {window_bytes / GIB:.2f} GiB <= resident high "
            f"{high / GIB:.2f} GiB",
        )

    managed = deployment.managed_local
    if managed is not None:
        segment = managed.mooncake.global_segment_size_bytes
        if segment < high:
            report.fail(
                "mooncake-capacity",
                f"resident high watermark {high / GIB:.2f} GiB > Mooncake "
                f"global segment {segment / GIB:.2f} GiB",
            )
        else:
            report.pass_(
                "mooncake-capacity",
                f"Mooncake global segment {segment / GIB:.2f} GiB >= resident "
                f"high {high / GIB:.2f} GiB",
            )
        report.metrics["mooncake_global_segment_bytes"] = segment
    elif deployment.backend == "mooncake":
        if external_store_bytes is None:
            report.warn(
                "mooncake-capacity",
                f"external Mooncake capacity is not visible; size the store for "
                f"at least {window_bytes / GIB:.2f} GiB per dispatch window",
            )
        elif external_store_bytes < high:
            report.fail(
                "mooncake-capacity",
                f"resident high watermark {high / GIB:.2f} GiB > declared "
                f"external Mooncake segment {external_store_bytes / GIB:.2f} GiB",
            )
        else:
            report.pass_(
                "mooncake-capacity",
                f"declared external Mooncake segment "
                f"{external_store_bytes / GIB:.2f} GiB >= resident high "
                f"{high / GIB:.2f} GiB",
            )
        report.metrics["external_store_bytes"] = external_store_bytes


def _check_capture_timeout(
    cfg: Config,
    deployment,
    report: PreflightReport,
    *,
    prompt_stats: Optional[PromptStats],
    capture_tps: float,
) -> None:
    timeout = resolve_capture_timeout_s(deployment.capture_timeout_s)
    report.metrics["capture_timeout_s"] = timeout
    max_tokens = prompt_stats.maximum if prompt_stats else None
    if max_tokens is None:
        report.warn(
            "capture-timeout",
            f"{timeout:.0f}s configured; no prompt lengths supplied, so the "
            "lease budget cannot be estimated",
        )
        return
    if capture_tps <= 0:
        report.warn("capture-timeout", "capture throughput must be positive")
        return
    lease = max(1, cfg.runtime.producer_lease)
    estimate = (lease * max_tokens) / capture_tps + lease
    report.metrics["capture_estimate_s"] = estimate
    if estimate > timeout:
        report.fail(
            "capture-timeout",
            f"one {lease}-prompt lease at {max_tokens} tokens needs about "
            f"{estimate:.0f}s at {capture_tps:.0f} tok/s > timeout "
            f"{timeout:.0f}s",
        )
    elif estimate > 0.75 * timeout:
        report.warn(
            "capture-timeout",
            f"lease estimate {estimate:.0f}s uses more than 75% of timeout "
            f"{timeout:.0f}s",
        )
    else:
        report.pass_(
            "capture-timeout",
            f"lease estimate {estimate:.0f}s <= timeout {timeout:.0f}s",
        )


def _check_swa(
    cfg: Config,
    deployment,
    report: PreflightReport,
    *,
    max_tokens: Optional[int],
    swa_pool_tokens: Optional[int],
) -> None:
    managed = deployment.managed_local
    if managed is None:
        report.warn(
            "swa-pool",
            "external capture server chunk size and SWA pool are not visible; "
            "confirm the server can admit the longest prompt",
        )
        return
    whole = [
        server
        for server in managed.capture_servers
        if server.chunked_prefill_size in (None, -1)
    ]
    if not whole:
        report.pass_("swa-pool", "capture servers use chunked prefill")
        return
    if max_tokens is None:
        report.warn(
            "swa-pool",
            "whole-prompt capture is configured; no prompt lengths supplied to "
            "check the SWA pool",
        )
        return
    if swa_pool_tokens is None:
        report.warn(
            "swa-pool",
            f"whole-prompt capture needs SWA pool >= {max_tokens} tokens; pass "
            "--swa-pool-tokens with the measured server value",
        )
        return
    if max_tokens > swa_pool_tokens:
        report.fail(
            "swa-pool",
            f"max prompt {max_tokens} > SWA pool {swa_pool_tokens}; the "
            "scheduler will never admit it",
        )
    else:
        report.pass_(
            "swa-pool",
            f"max prompt {max_tokens} <= SWA pool {swa_pool_tokens}",
        )


def _check_paths(cfg: Config, report: PreflightReport, *, allow_existing: bool) -> None:
    if cfg.training.resume_from is not None:
        report.pass_("run-paths", f"resume from {cfg.training.resume_from}")
        return
    deployment = cfg.deployment.disaggregated
    paths = [("output_dir", cfg.output_dir)]
    if deployment is not None:
        paths.extend(
            [
                ("control_dir", deployment.control_dir),
                ("consumer_state_dir", deployment.consumer_state_dir),
            ]
        )
    stale = [name for name, path in paths if _nonempty_path(path)]
    if stale and not allow_existing:
        report.fail(
            "run-paths",
            f"non-empty artifacts exist for {stale}; use a fresh run id or "
            "pass --allow-existing only when resuming intentionally",
        )
    elif stale:
        report.warn("run-paths", f"reusing non-empty artifacts for {stale}")
    else:
        report.pass_("run-paths", "run directories are fresh or absent")


__all__ = [
    "Check",
    "PreflightReport",
    "PromptStats",
    "load_prompt_stats",
    "merge_prompt_stats",
    "run_preflight",
]
