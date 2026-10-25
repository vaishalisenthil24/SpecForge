# coding=utf-8
"""CPU-only end-to-end simulation of the online capture/training hand-off.

The expensive failures in an online disaggregated run are not numerical: a
capture ref that never arrives, a producer that backpressures forever because
the dispatch window cannot fit, a feature-store cap that raises only after a
GPU box is warm, or a reused run id whose old feature keys poison consumer
deduplication. This module runs the real producer, reference channel, ledger,
and commit/dedup path on CPU with a synthetic capture source and a synthetic
feature store, so those failures surface in seconds.

No model, tokenizer, GPU, or Mooncake client is imported.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from specforge.algorithms.registry import AlgorithmRegistration
from specforge.config import Config
from specforge.launch import build_disagg_online_producer
from specforge.runtime.contracts import FeatureHandle, FeatureSpec, SampleRef
from specforge.runtime.control_plane.metadata_store import SQLiteMetadataStore
from specforge.runtime.data_plane.feature_store import FeatureStore
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel

GIB = 1 << 30


@dataclass(frozen=True)
class SimulationCheck:
    level: str
    name: str
    message: str


@dataclass
class SimulationReport:
    checks: List[SimulationCheck] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add(self, level: str, name: str, message: str) -> None:
        self.checks.append(SimulationCheck(level, name, message))

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
    def ok(self) -> bool:
        return self.failures == 0

    def render(self) -> str:
        headline = "PASS" if self.ok else "FAIL"
        lines = [f"SPECFORGE SIMULATE: {headline} ({self.failures} failures)"]
        for check in self.checks:
            lines.append(f"[{check.level.upper()}] {check.name}: {check.message}")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": "pass" if self.ok else "fail",
            "failures": self.failures,
            "checks": [
                {"level": check.level, "name": check.name, "message": check.message}
                for check in self.checks
            ],
            "metrics": self.metrics,
        }


class _ObservingChannel:
    """Delegating channel that records the producer-side in-flight peak."""

    def __init__(self, inner: StreamingRefChannel) -> None:
        self.inner = inner
        self.max_producer_in_flight = 0
        self._lock = threading.Lock()

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def in_flight_remote(self) -> int:
        value = self.inner.in_flight_remote()
        if threading.current_thread() is not threading.main_thread():
            with self._lock:
                self.max_producer_in_flight = max(self.max_producer_in_flight, value)
        return value


class SimulatedFeatureStore(FeatureStore):
    """Byte-accounted feature store that allocates no feature-sized tensors.

    ``put`` records the synthetic byte size from ``metadata["sim_bytes"]`` and
    enforces ``max_resident_bytes`` exactly like the local/Mooncake backstop, so
    capacity errors reproduce without multi-terabyte allocations.
    """

    get_returns_fresh_tensors = True

    def __init__(
        self,
        store_id: str,
        *,
        max_resident_bytes: Optional[int] = None,
        strategy: str = "eagle3",
    ) -> None:
        self.store_id = store_id
        self.max_resident_bytes = max_resident_bytes
        self.strategy = strategy
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.put_count = 0
        self.free_count = 0
        self._sizes: Dict[str, int] = {}
        self._generations: Dict[str, int] = {}
        self._active_leases: Dict[str, str] = {}
        self._generation_counter = 0
        self._lease_counter = 0
        self._lock = threading.RLock()

    def put(
        self,
        tensors: Dict[str, torch.Tensor],
        *,
        sample_id: str,
        metadata: Dict[str, Any],
    ) -> SampleRef:
        size = int(metadata.get("sim_bytes", 0))
        if size <= 0:
            raise ValueError("simulated puts must carry metadata['sim_bytes'] > 0")
        with self._lock:
            previous = self._sizes.get(sample_id, 0)
            projected = self.resident_bytes - previous + size
            if (
                self.max_resident_bytes is not None
                and projected > self.max_resident_bytes
            ):
                raise MemoryError(
                    "simulated feature-store hard cap exceeded: "
                    f"projected={projected} > limit={self.max_resident_bytes} bytes"
                )
            self._generation_counter += 1
            generation = self._generation_counter
            self._sizes[sample_id] = size
            self._generations[sample_id] = generation
            self.resident_bytes = projected
            self.peak_resident_bytes = max(self.peak_resident_bytes, projected)
            self.put_count += 1
        return SampleRef(
            sample_id=sample_id,
            run_id=str(metadata.get("run_id", self.store_id)),
            source_task_id=metadata.get("task_id"),
            feature_store_uri=f"sim://{self.store_id}/{sample_id}?generation={generation}",
            feature_keys={"hidden_state": f"{sample_id}/hidden_state"},
            feature_specs={
                "hidden_state": FeatureSpec(
                    name="hidden_state", shape=(1,), dtype="float16"
                )
            },
            strategy=str(metadata.get("strategy", self.strategy)),  # type: ignore[arg-type]
            num_tokens=int(metadata.get("num_tokens", 0)),
            estimated_bytes=size,
            metadata={"sim_bytes": size},
        )

    def get(
        self,
        sample_ref: SampleRef,
        *,
        device: torch.device | str = "cpu",
        names: Optional[List[str]] = None,
    ):
        with self._lock:
            if sample_ref.sample_id not in self._sizes:
                raise KeyError(f"feature sample not resident: {sample_ref.sample_id}")
            self._lease_counter += 1
            token = f"sim-lease-{self._lease_counter}"
            self._active_leases[token] = sample_ref.sample_id
            generation = self._generations[sample_ref.sample_id]
        handle = FeatureHandle(
            sample_id=sample_ref.sample_id,
            generation=generation,
            lease_token=token,
        )
        return (
            {"hidden_state": torch.zeros(1, dtype=torch.float16)},
            handle,
        )

    def release(self, handle: FeatureHandle, *, reason: str = "consumed") -> None:
        with self._lock:
            sample_id = self._active_leases.pop(handle.lease_token, None)
            if sample_id is None:
                return
            if self._generations.get(sample_id) != handle.generation:
                return
            if sample_id in self._active_leases.values():
                return
            self._free_locked(sample_id)

    def abort(self, sample_id: str, *, reason: str) -> None:
        with self._lock:
            self._free_locked(sample_id)

    def _free_locked(self, sample_id: str) -> None:
        size = self._sizes.pop(sample_id, None)
        self._generations.pop(sample_id, None)
        if size is not None:
            self.resident_bytes -= size
            self.free_count += 1

    def health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "resident_bytes": self.resident_bytes,
                "peak_resident_bytes": self.peak_resident_bytes,
                "put_count": self.put_count,
                "free_count": self.free_count,
                "samples": len(self._sizes),
            }


class SimulatedCaptureSource:
    """Stands in for the SGLang server capture adapter.

    Returns refs shaped exactly like ``RolloutWorker`` expects and accounts a
    realistic per-token feature size, without touching a server.
    """

    def __init__(
        self,
        store: SimulatedFeatureStore,
        *,
        bytes_per_token: int,
        default_tokens: int,
        strategy: str,
    ) -> None:
        self.store = store
        self.bytes_per_token = int(bytes_per_token)
        self.default_tokens = int(default_tokens)
        self.strategy = strategy
        self.calls = 0
        self.tasks = 0

    def produce_refs(self, tasks, *, capture=None) -> List[SampleRef]:
        refs: List[SampleRef] = []
        for task in tasks:
            tokens = int(task.metadata.get("tokens", self.default_tokens))
            refs.append(
                self.store.put(
                    {},
                    sample_id=f"{task.run_id}:{task.task_id}",
                    metadata={
                        "sim_bytes": tokens * self.bytes_per_token,
                        "num_tokens": tokens,
                        "task_id": task.task_id,
                        "run_id": task.run_id,
                        "strategy": self.strategy,
                    },
                )
            )
        self.calls += 1
        self.tasks += len(tasks)
        return refs


@dataclass
class _PassResult:
    produced: int
    consumed: int
    duplicates: int
    fresh: int
    peak_resident_bytes: int
    max_in_flight: int
    backpressure_waits: int


def _dispatch_refs(cfg: Config) -> int:
    from specforge.runtime.contracts import consumer_dispatch_quantum

    trainer = cfg.deployment.trainer
    per_optimizer_step = (
        trainer.nnodes
        * trainer.nproc_per_node
        * cfg.training.batch_size
        * cfg.training.accumulation_steps
    )
    return consumer_dispatch_quantum(cfg.runtime.consumer_dispatch, per_optimizer_step)


def _num_sources(cfg: Config) -> int:
    deployment = cfg.deployment.disaggregated
    if deployment is None:
        return 1
    if deployment.managed_local is not None:
        return max(1, len(deployment.managed_local.capture_servers))
    return max(1, len(deployment.server_urls or ()))


def _make_prompts(count: int, tokens: int, max_length: int):
    return [
        {
            "task_id": f"sim-{index:06d}",
            "payload": {"input_ids": [1, 2, 3]},
            "max_length": max_length,
            "metadata": {"tokens": tokens},
        }
        for index in range(count)
    ]


def _drive_consumer(
    channel: StreamingRefChannel,
    store: SimulatedFeatureStore,
    ledger: SQLiteMetadataStore,
    producer_done: threading.Event,
    *,
    poll_s: float = 0.005,
    consumer_delay_s: float = 0.0,
) -> _PassResult:
    seen: set[str] = set()
    consumed = 0
    fresh_count = 0
    duplicates = 0
    last_in_flight = 0
    backpressure_waits = 0
    stall_since: Optional[float] = None
    while True:
        refs = channel.poll()
        if refs:
            freshness = ledger.commit_samples(refs)
            fresh = [ref for ref, is_fresh in zip(refs, freshness) if is_fresh]
            duplicates += len(refs) - len(fresh)
            for ref in refs:
                if ref.sample_id in seen and ref in fresh:
                    raise RuntimeError(
                        f"ledger returned a duplicate sample as fresh: {ref.sample_id}"
                    )
                seen.add(ref.sample_id)
            for ref in fresh:
                _features, handle = store.get(ref)
                store.release(handle)
                if consumer_delay_s > 0:
                    time.sleep(consumer_delay_s)
            fresh_count += len(fresh)
            channel.mark_consumed(len(refs))
            consumed += len(refs)
        in_flight = channel.in_flight_remote()
        if in_flight == last_in_flight and in_flight > 0:
            stall_since = stall_since or time.monotonic()
            if time.monotonic() - stall_since >= 0.25:
                backpressure_waits += 1
                stall_since = None
        else:
            stall_since = None
        last_in_flight = in_flight
        if (
            producer_done.is_set()
            and not refs
            and channel.consumed_remote() == consumed
        ):
            if channel.consumed_remote() == channel.published:
                break
        time.sleep(poll_s)
    return _PassResult(
        produced=channel.published,
        consumed=consumed,
        duplicates=duplicates,
        fresh=fresh_count,
        peak_resident_bytes=store.peak_resident_bytes,
        max_in_flight=channel.max_producer_in_flight,
        backpressure_waits=backpressure_waits,
    )


def _run_pass(
    cfg: Config,
    algorithm: AlgorithmRegistration,
    *,
    work_dir: Path,
    ledger: SQLiteMetadataStore,
    run_id: str,
    prompts,
    dispatch_refs: int,
    bytes_per_token: int,
    tokens_per_prompt: int,
    target_hidden_size: int,
    aux_layer_ids: tuple[int, ...],
    timeout_s: float,
    consumer_delay_s: float,
) -> _PassResult:
    control = work_dir / run_id
    channel = _ObservingChannel(StreamingRefChannel(str(control / "refs.jsonl")))
    store = SimulatedFeatureStore(
        run_id,
        max_resident_bytes=cfg.runtime.feature_store_max_resident_bytes,
        strategy=algorithm.name,
    )
    sources = [
        SimulatedCaptureSource(
            store,
            bytes_per_token=bytes_per_token,
            default_tokens=tokens_per_prompt,
            strategy=algorithm.name,
        )
        for _ in range(_num_sources(cfg))
    ]
    workers, drive = build_disagg_online_producer(
        algorithm=algorithm,
        prompts=prompts,
        feature_store=store,
        channel=channel,
        run_id=run_id,
        target_hidden_size=target_hidden_size,
        aux_hidden_state_layer_ids=aux_layer_ids,
        num_rollout_workers=len(sources),
        feature_source=sources,
        lease=cfg.runtime.producer_lease,
        producer_concurrency=cfg.runtime.producer_concurrency,
        in_flight_high_watermark=cfg.runtime.in_flight_high_watermark,
        in_flight_low_watermark=cfg.runtime.in_flight_low_watermark,
        resident_high_watermark_bytes=cfg.runtime.resident_high_watermark_bytes,
        resident_low_watermark_bytes=cfg.runtime.resident_low_watermark_bytes,
        feature_store_max_resident_bytes=cfg.runtime.feature_store_max_resident_bytes,
        backpressure_poll_s=0.005,
        peer_wait_timeout_s=timeout_s,
        sleep=lambda seconds: time.sleep(min(seconds, 0.05)),
    )
    for worker in workers:
        worker.start()
    channel.publish_consumer_quantum(dispatch_refs)
    done = threading.Event()
    error: list[BaseException] = []

    def run_producer() -> None:
        try:
            drive()
        except BaseException as exc:  # noqa: BLE001 - surfaced by the caller
            error.append(exc)
            channel.fail(f"{type(exc).__name__}: {exc}")
        finally:
            done.set()

    thread = threading.Thread(target=run_producer, name="sim-producer", daemon=True)
    thread.start()
    try:
        result = _drive_consumer(
            channel,
            store,
            ledger,
            done,
            consumer_delay_s=consumer_delay_s,
        )
    finally:
        thread.join(timeout=timeout_s)
        if thread.is_alive():
            raise TimeoutError(f"simulated producer did not stop in {timeout_s:.0f}s")
    if error:
        raise error[0]
    failure = channel.failure()
    if failure:
        raise RuntimeError(f"producer failed: {failure}")
    return result


def run_simulation(
    cfg: Config,
    algorithm: AlgorithmRegistration,
    *,
    work_dir: str,
    prompt_count: int = 0,
    tokens_per_prompt: Optional[int] = None,
    bytes_per_token: int = 40976,
    dispatch_refs: Optional[int] = None,
    target_hidden_size: int = 5120,
    aux_layer_ids: tuple[int, ...] = (37, 38, 39),
    timeout_s: float = 600.0,
    check_restart: bool = True,
    consumer_delay_s: float = 0.002,
) -> SimulationReport:
    """Drive one synthetic online run plus stale-run-id restart checks."""
    report = SimulationReport()
    window = dispatch_refs or _dispatch_refs(cfg)
    tokens = tokens_per_prompt or cfg.data.max_length
    count = prompt_count or max(8, min(2 * window, 512))
    report.metrics.update(
        {
            "run_id": cfg.run_id,
            "dispatch_refs": window,
            "prompt_count": count,
            "tokens_per_prompt": tokens,
            "bytes_per_token": bytes_per_token,
            "dispatch_window_bytes": window * tokens * bytes_per_token,
            "resident_high_watermark_bytes": cfg.runtime.resident_high_watermark_bytes,
            "feature_store_max_resident_bytes": (
                cfg.runtime.feature_store_max_resident_bytes
            ),
        }
    )
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "metadata.sqlite"
    ledger = SQLiteMetadataStore(str(ledger_path))
    try:
        prompts = _make_prompts(count, tokens, cfg.data.max_length)
        first = _run_pass(
            cfg,
            algorithm,
            work_dir=root,
            ledger=ledger,
            run_id=cfg.run_id,
            prompts=prompts,
            dispatch_refs=window,
            bytes_per_token=bytes_per_token,
            tokens_per_prompt=tokens,
            target_hidden_size=target_hidden_size,
            aux_layer_ids=aux_layer_ids,
            timeout_s=timeout_s,
            consumer_delay_s=consumer_delay_s,
        )
        report.metrics.update(
            {
                "produced": first.produced,
                "consumed": first.consumed,
                "fresh": first.fresh,
                "duplicates": first.duplicates,
                "peak_resident_bytes": first.peak_resident_bytes,
                "max_in_flight": first.max_in_flight,
            }
        )
        if first.duplicates:
            report.fail(
                "exactly-once",
                f"{first.duplicates} duplicate refs arrived in a fresh run",
            )
        elif first.consumed != first.produced or first.fresh != first.produced:
            report.fail(
                "exactly-once",
                f"produced={first.produced} consumed={first.consumed} "
                f"fresh={first.fresh}",
            )
        else:
            report.pass_(
                "exactly-once",
                f"{first.produced} refs produced and consumed once",
            )
        if (
            cfg.runtime.feature_store_max_resident_bytes is not None
            and first.peak_resident_bytes > cfg.runtime.feature_store_max_resident_bytes
        ):
            report.fail(
                "store-capacity",
                f"peak resident {first.peak_resident_bytes / GIB:.2f} GiB > "
                f"cap {cfg.runtime.feature_store_max_resident_bytes / GIB:.2f} GiB",
            )
        else:
            report.pass_(
                "store-capacity",
                f"peak resident {first.peak_resident_bytes / GIB:.2f} GiB stayed "
                "under the feature-store cap",
            )
        if first.max_in_flight >= window:
            report.pass_(
                "backpressure",
                f"producer reached the {window}-ref dispatch window and paused "
                f"at most at {first.max_in_flight} in-flight refs",
            )
        else:
            report.warn(
                "backpressure",
                f"peak in-flight {first.max_in_flight} refs never exceeded the "
                f"{window}-ref window; raise --prompt-count to exercise it",
            )
        if check_restart:
            # Restart passes only need one dispatch window of prompts. Running
            # the full stream here measures the duplicate-object leak, not the
            # run-id hazard this check is about.
            restart_prompts = prompts[: max(1, min(len(prompts), window))]
            stale = _run_pass(
                cfg,
                algorithm,
                work_dir=root / "restart-same-id",
                ledger=ledger,
                run_id=cfg.run_id,
                prompts=restart_prompts,
                dispatch_refs=window,
                bytes_per_token=bytes_per_token,
                tokens_per_prompt=tokens,
                target_hidden_size=target_hidden_size,
                aux_layer_ids=aux_layer_ids,
                timeout_s=timeout_s,
                consumer_delay_s=consumer_delay_s,
            )
            report.metrics["stale_restart_duplicates"] = stale.duplicates
            if stale.duplicates == 0:
                report.fail(
                    "stale-run-id",
                    "reusing the same run id did not reproduce duplicate "
                    "sample ids; the simulator is not exercising the hazard",
                )
            else:
                report.pass_(
                    "stale-run-id",
                    f"reusing {cfg.run_id!r} produced {stale.duplicates} "
                    "duplicate refs; always start a fresh run id after an abort",
                )
            fresh_id = f"{cfg.run_id}-simfresh"
            fresh = _run_pass(
                cfg,
                algorithm,
                work_dir=root / "restart-fresh-id",
                ledger=ledger,
                run_id=fresh_id,
                prompts=restart_prompts,
                dispatch_refs=window,
                bytes_per_token=bytes_per_token,
                tokens_per_prompt=tokens,
                target_hidden_size=target_hidden_size,
                aux_layer_ids=aux_layer_ids,
                timeout_s=timeout_s,
                consumer_delay_s=consumer_delay_s,
            )
            report.metrics["fresh_restart_duplicates"] = fresh.duplicates
            if fresh.duplicates == 0 and fresh.fresh == fresh.produced:
                report.pass_(
                    "fresh-run-id",
                    f"fresh run id {fresh_id!r} produced no duplicates",
                )
            else:
                report.fail(
                    "fresh-run-id",
                    f"fresh run id produced {fresh.duplicates} duplicates "
                    f"(fresh={fresh.fresh}, produced={fresh.produced})",
                )
    except MemoryError as exc:
        report.fail("store-capacity", f"feature store rejected the window: {exc}")
    except Exception as exc:  # noqa: BLE001 - reported as a preflight-style failure
        if "hard cap exceeded" in str(exc):
            report.fail(
                "store-capacity",
                "feature store rejected the dispatch window; raise "
                "runtime.feature_store_max_resident_bytes or lower "
                f"accumulation_steps ({exc})",
            )
        else:
            report.fail("simulation", f"{type(exc).__name__}: {exc}")
    finally:
        ledger.close()
    return report


__all__ = [
    "SimulatedCaptureSource",
    "SimulatedFeatureStore",
    "SimulationReport",
    "run_simulation",
]
