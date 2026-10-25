# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The single public SpecForge training entry point.

``specforge train --config run.yaml [section.field=value ...]`` builds the
validated :class:`~specforge.config.Config`, assembles the models, and runs
training through the DataFlow launch builders — the same wiring the
programmatic path uses, behind one typed config.

``deployment.trainer`` defines the process topology. The CLI self-launches
multi-rank workers and recognizes an existing torchrun worker environment
without nesting another launcher.

Model/data assembly lives in :mod:`specforge.training.assembly`; this module is
deliberately limited to click command definitions and distributed process
lifecycle.
"""

from __future__ import annotations

import json
import os
import signal
import socket
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator, Optional, Sequence

import click

from specforge.config import Config, load_config


class _WorkerTermination(BaseException):
    """Translate a process signal into normal Python stack unwinding."""

    def __init__(self, signum: int):
        self.signum = signum


@contextmanager
def _worker_signal_unwind() -> Iterator[None]:
    """Make worker termination run training and distributed cleanup blocks.

    Managed supervisors terminate worker process groups with SIGTERM.  Python's
    default SIGTERM action exits immediately, bypassing ``finally`` blocks.  The
    first managed signal is therefore raised as a ``BaseException``; subsequent
    signals are ignored while cleanup runs, after which the original handlers
    are restored.  A supervising parent may still enforce its grace period with
    SIGKILL if cleanup cannot finish.
    """
    managed_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        managed_signals.append(signal.SIGHUP)
    previous_handlers = {}

    def unwind(signum, _frame):
        for installed in previous_handlers:
            signal.signal(installed, signal.SIG_IGN)
        raise _WorkerTermination(signum)

    try:
        for signum in managed_signals:
            try:
                previous_handlers[signum] = signal.signal(signum, unwind)
            except ValueError:
                # Embedded callers may execute the CLI from a non-main thread,
                # where Python does not permit signal handler installation.
                for installed, handler in previous_handlers.items():
                    signal.signal(installed, handler)
                previous_handlers.clear()
                break
        yield
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _bootstrap_single_process_env() -> None:
    """Provide ``env://`` rendezvous values for a direct one-GPU invocation."""
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    present = [name for name in required if name in os.environ]
    if present:
        missing = [name for name in required if name not in os.environ]
        if missing:
            raise ValueError(
                "distributed environment is incomplete; present="
                f"{present}, missing={missing}. Launch with torchrun or unset the "
                "partial distributed variables for a one-process run."
            )
        return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rendezvous:
        rendezvous.bind(("127.0.0.1", 0))
        port = rendezvous.getsockname()[1]
    os.environ.update(
        {
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )


def _validate_world_size(cfg: Config, world_size: int) -> None:
    cfg.validate_world_size(world_size)


def _train(resolved) -> int:
    from accelerate.utils import set_seed

    cfg = resolved.config
    # Make the typed recipe authoritative for the backend's existing FSDP
    # sharding seam in both direct and managed-local worker processes.
    os.environ["FSDP_SHARDING"] = cfg.training.fsdp_sharding
    if cfg.training.ddp_bucket_cap_mb is None:
        os.environ.pop("SPECFORGE_DDP_BUCKET_CAP_MB", None)
    else:
        os.environ["SPECFORGE_DDP_BUCKET_CAP_MB"] = str(cfg.training.ddp_bucket_cap_mb)
    set_seed(cfg.training.seed)
    if cfg.training.role == "producer":
        # A server-capture/offline-ingest producer owns no trainer process
        # group and must not initialize CUDA merely to publish feature refs.
        from specforge.application import build_application_run

        return build_application_run(resolved).run()

    from specforge.distributed import destroy_distributed, init_distributed

    _bootstrap_single_process_env()
    _validate_world_size(cfg, int(os.environ["WORLD_SIZE"]))
    init_distributed(
        timeout=cfg.training.dist_timeout,
        tp_size=cfg.training.tp_size,
        sp_ulysses_size=cfg.training.sp_ulysses_size,
        sp_ring_size=cfg.training.sp_ring_size,
    )
    failed = True
    try:
        import torch.distributed as dist

        _validate_world_size(cfg, dist.get_world_size())
        from specforge.application import build_application_run

        result = build_application_run(resolved).run()
        failed = False
        return result
    finally:
        # On failure abort instead of collectively destroying; see destroy_distributed.
        destroy_distributed(abort=failed)


def _config_for_role(cfg: Config, role: str) -> Config:
    """Resolve a launch role without changing the persisted run config.

    A shared disaggregated config may contain trainer-only state used by the
    consumer child.  The capture-only producer must ignore that state when the
    launcher derives its role from the shared config.
    """
    raw = cfg.model_dump()
    raw["training"]["role"] = role
    disaggregated = raw["deployment"].get("disaggregated")
    if disaggregated is not None and disaggregated.get("managed_local") is not None:
        # This field describes services owned by the parent supervisor.  A role
        # child consumes the already-derived environment and must not attempt to
        # validate or own that stack again.
        disaggregated["managed_local"] = None
    if role == "producer":
        raw["profiling"]["enabled"] = False
    return Config.model_validate(raw)


_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=_CONTEXT_SETTINGS)
def cli() -> None:
    """SpecForge: speculative decoding training framework."""


@cli.command(short_help="train a draft model from a typed config")
@click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    metavar="PATH",
    help="YAML or JSON run config.",
)
@click.option(
    "--role",
    type=click.Choice(("auto", "all", "producer", "consumer", "both")),
    default="auto",
    show_default=True,
    help=(
        "Launch selection: offline local 'all' or online/disaggregated "
        "producer+consumer when 'auto'."
    ),
)
@click.option(
    "--node-rank",
    type=int,
    default=None,
    help="Node-local rank for an explicit multi-node trainer launch.",
)
@click.option(
    "--plan",
    is_flag=True,
    help="Print the resolved process plan without starting workers.",
)
@click.argument("overrides", nargs=-1)
def train(
    config_path: str,
    role: str,
    node_rank: Optional[int],
    plan: bool,
    overrides: Sequence[str],
) -> int:
    """Train a draft model from a typed run config.

    OVERRIDES are dotted ``section.field=value`` assignments applied on top of
    the config file, e.g. ``training.learning_rate=1e-4``.
    """
    overrides = list(overrides)
    cfg = load_config(config_path, overrides)
    from specforge.application import bind_run, resolve_run
    from specforge.launch_plan import build_launch_plan, run_commands

    resolved = resolve_run(cfg)
    launch = build_launch_plan(
        resolved.config,
        algorithm=resolved.algorithm,
        config_path=config_path,
        overrides=overrides,
        requested_role=role,
        node_rank=node_rank,
    )
    if plan:
        print(launch.render())
        if launch.managed_rdma_error is not None:
            # The plan renders anywhere; launching it on this host would fail.
            click.echo(f"warning: {launch.managed_rdma_error}", err=True)
        return 0
    if launch.kind == "worker":
        for key, value in launch.worker_env.items():
            if value is None:
                # CommandSpec.env contract: None unsets the variable.
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        role_config = _config_for_role(resolved.config, launch.role)
        try:
            with _worker_signal_unwind():
                _train(bind_run(role_config, resolved.algorithm))
        except _WorkerTermination as received:
            return 128 + received.signum
        return 0
    return run_commands(launch)


@cli.command(short_help="validate a run config and size it without GPUs")
@click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    metavar="PATH",
    help="YAML or JSON run config.",
)
@click.option(
    "--data",
    "data_path",
    default=None,
    metavar="PATH",
    help=(
        "Prompt JSONL with input_ids, or a JSON stats file. Defaults to the "
        "config's prompts_path/train_data_path."
    ),
)
@click.option("--prompt-count", type=int, default=None)
@click.option("--p50-prompt-tokens", type=int, default=None)
@click.option("--p95-prompt-tokens", type=int, default=None)
@click.option("--max-prompt-tokens", type=int, default=None)
@click.option("--target-hidden-size", type=int, default=None)
@click.option(
    "--aux-layer-ids",
    default=None,
    metavar="ID,ID,...",
    help="Capture layer ids; use with --target-hidden-size when the checkpoint is remote.",
)
@click.option(
    "--capture-tps",
    type=float,
    default=5000.0,
    show_default=True,
    help="Conservative capture throughput for the timeout estimate.",
)
@click.option(
    "--swa-pool-tokens",
    type=int,
    default=None,
    help="Measured SWA token capacity of a whole-prompt capture server.",
)
@click.option(
    "--external-store-bytes",
    type=int,
    default=None,
    help="Declared capacity of an external Mooncake segment, for the capacity check.",
)
@click.option("--max-rows", type=int, default=0, show_default=True)
@click.option("--allow-existing", is_flag=True)
@click.option("--json-output", is_flag=True)
@click.argument("overrides", nargs=-1)
def preflight(
    config_path: str,
    data_path: Optional[str],
    prompt_count: Optional[int],
    p50_prompt_tokens: Optional[int],
    p95_prompt_tokens: Optional[int],
    max_prompt_tokens: Optional[int],
    target_hidden_size: Optional[int],
    aux_layer_ids: Optional[str],
    capture_tps: float,
    swa_pool_tokens: Optional[int],
    external_store_bytes: Optional[int],
    max_rows: int,
    allow_existing: bool,
    json_output: bool,
    overrides: Sequence[str],
) -> int:
    """Check config, dataset shape, capture budget, and feature-store size."""
    from specforge.application import resolve_run
    from specforge.preflight import (
        load_prompt_stats,
        merge_prompt_stats,
        run_preflight,
    )

    try:
        cfg = load_config(config_path, list(overrides))
        resolved = resolve_run(cfg)
        explicit_data = data_path is not None
        selected_path = data_path
        if selected_path is None and cfg.mode == "online":
            selected_path = cfg.data.prompts_path or cfg.data.train_data_path
        stats = None
        if selected_path and (
            explicit_data or os.path.exists(os.path.expanduser(selected_path))
        ):
            stats = load_prompt_stats(selected_path, max_rows=max_rows)
        stats = merge_prompt_stats(
            stats,
            count=prompt_count,
            p50=p50_prompt_tokens,
            p95=p95_prompt_tokens,
            maximum=max_prompt_tokens,
        )
        contract = None
        if (target_hidden_size is None) != (aux_layer_ids is None):
            raise click.UsageError(
                "--target-hidden-size and --aux-layer-ids must be set together",
                ctx=click.get_current_context(),
            )
        if target_hidden_size is not None:
            from specforge.training.capture_contract import ServerCaptureContract

            try:
                layers = tuple(
                    int(item) for item in aux_layer_ids.split(",") if item.strip()
                )
            except ValueError as exc:
                raise click.UsageError(
                    "--aux-layer-ids must be comma-separated integers",
                    ctx=click.get_current_context(),
                ) from exc
            if not layers:
                raise click.UsageError(
                    "--aux-layer-ids must not be empty",
                    ctx=click.get_current_context(),
                )
            contract = ServerCaptureContract(
                method="preflight-override",
                aux_layer_ids=layers,
                target_hidden_size=target_hidden_size,
                target_vocab_size=0,
                draft_vocab_size=0,
            )
        report = run_preflight(
            resolved.config,
            resolved.algorithm,
            contract=contract,
            prompt_stats=stats,
            capture_tps=capture_tps,
            swa_pool_tokens=swa_pool_tokens,
            allow_existing=allow_existing,
            external_store_bytes=external_store_bytes,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(report.as_dict(), indent=2, sort_keys=True)
        if json_output
        else report.render()
    )
    return 0 if report.ok else 1


@cli.command(short_help="simulate the online capture/training path on CPU")
@click.option("-c", "--config", "config_path", required=True, metavar="PATH")
@click.option(
    "--work-dir",
    default=None,
    metavar="PATH",
    help="Scratch directory for the reference channel and ledger.",
)
@click.option(
    "--prompt-count",
    type=int,
    default=0,
    show_default=True,
    help="Synthetic prompt count; 0 uses two dispatch windows (capped at 512).",
)
@click.option(
    "--tokens-per-prompt",
    type=int,
    default=None,
    help="Tokens per synthetic prompt; defaults to data.max_length.",
)
@click.option(
    "--bytes-per-token",
    type=int,
    default=40976,
    show_default=True,
    help="Synthetic capture bytes per token, from specforge preflight.",
)
@click.option("--refs-per-window", type=int, default=None)
@click.option("--target-hidden-size", type=int, default=None)
@click.option("--aux-layer-ids", default=None, metavar="ID,ID,...")
@click.option("--timeout-s", type=float, default=600.0, show_default=True)
@click.option(
    "--consumer-delay-ms",
    type=float,
    default=2.0,
    show_default=True,
    help="Per-batch consumer pause, to exercise producer backpressure.",
)
@click.option("--no-restart-check", is_flag=True)
@click.option("--json-output", is_flag=True)
@click.argument("overrides", nargs=-1)
def simulate(
    config_path: str,
    work_dir: Optional[str],
    prompt_count: int,
    tokens_per_prompt: Optional[int],
    bytes_per_token: int,
    refs_per_window: Optional[int],
    target_hidden_size: Optional[int],
    aux_layer_ids: Optional[str],
    timeout_s: float,
    consumer_delay_ms: float,
    no_restart_check: bool,
    json_output: bool,
    overrides: Sequence[str],
) -> int:
    """Run the real producer, channel, ledger and dedup path without a GPU."""
    import tempfile

    from specforge.application import resolve_run
    from specforge.simulate import run_simulation

    try:
        cfg = load_config(config_path, list(overrides))
        resolved = resolve_run(cfg)
        layers: tuple[int, ...] = ()
        if aux_layer_ids is not None:
            try:
                layers = tuple(
                    int(item) for item in aux_layer_ids.split(",") if item.strip()
                )
            except ValueError as exc:
                raise click.UsageError(
                    "--aux-layer-ids must be comma-separated integers",
                    ctx=click.get_current_context(),
                ) from exc
        hidden = target_hidden_size
        if hidden is None or not layers:
            try:
                from specforge.training.capture_contract import (
                    resolve_server_capture_contract,
                )

                contract = resolve_server_capture_contract(
                    resolved.config, algorithm=resolved.algorithm
                )
                hidden = hidden or contract.target_hidden_size
                layers = layers or tuple(contract.aux_layer_ids)
            except Exception as exc:  # noqa: BLE001 - explicit flags avoid this
                if hidden is None or not layers:
                    raise click.UsageError(
                        "could not resolve the capture contract without a local "
                        "checkpoint; pass --target-hidden-size and "
                        f"--aux-layer-ids ({exc})",
                        ctx=click.get_current_context(),
                    ) from exc
        scratch = work_dir or tempfile.mkdtemp(prefix="specforge-simulate-")
        report = run_simulation(
            resolved.config,
            resolved.algorithm,
            work_dir=scratch,
            prompt_count=prompt_count,
            tokens_per_prompt=tokens_per_prompt,
            bytes_per_token=bytes_per_token,
            dispatch_refs=refs_per_window,
            target_hidden_size=int(hidden),
            aux_layer_ids=layers,
            timeout_s=timeout_s,
            check_restart=not no_restart_check,
            consumer_delay_s=consumer_delay_ms / 1000.0,
        )
        report.metrics["work_dir"] = scratch
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(report.as_dict(), indent=2, sort_keys=True)
        if json_output
        else report.render()
    )
    return 0 if report.ok else 1


@cli.command(short_help="materialize a runtime checkpoint as a model directory")
@click.option(
    "--to",
    "target",
    type=click.Choice(("hf", "sglang")),
    required=True,
    help="Export layout.",
)
@click.option("--checkpoint", required=True, metavar="PATH")
@click.option("--draft-config", required=True, metavar="PATH")
@click.option("--output-dir", required=True, metavar="PATH")
@click.option("--vocab-mapping", default=None, metavar="PATH")
@click.option(
    "--embedding-source",
    default=None,
    metavar="PATH",
    help="Target model path supplying a frozen embedding for HF export.",
)
@click.option(
    "--embedding-key",
    default="model.embed_tokens.weight",
    show_default=True,
)
@click.pass_context
def export(
    ctx: click.Context,
    target: str,
    checkpoint: str,
    draft_config: str,
    output_dir: str,
    vocab_mapping: Optional[str],
    embedding_source: Optional[str],
    embedding_key: str,
) -> int:
    """Materialize a runtime checkpoint as a model directory."""
    if target == "hf":
        from specforge.export.to_hf import export_to_hf

        export_to_hf(
            checkpoint,
            draft_config,
            output_dir,
            vocab_mapping_path=vocab_mapping,
            embedding_source=embedding_source,
            embedding_key=embedding_key,
        )
        return 0
    if embedding_source is not None:
        raise click.UsageError("--embedding-source is only valid with --to hf", ctx)
    from specforge.export.to_sglang import export_to_sglang

    export_to_sglang(
        checkpoint,
        draft_config,
        output_dir,
        vocab_mapping_path=vocab_mapping,
    )
    return 0


@cli.command(short_help="benchmark a running SGLang server")
@click.option("--model", required=True, help="Tokenizer/model id for prompt rendering.")
@click.option(
    "--dataset",
    type=click.Choice(("gsm8k", "math500", "humaneval", "mbpp", "mt-bench")),
    required=True,
)
@click.option("--max-new-tokens", type=int, default=2048, show_default=True)
@click.option("--temperature", type=float, default=0.0, show_default=True)
@click.option("--top-p", type=float, default=1.0, show_default=True)
@click.option("--top-k", type=int, default=1, show_default=True)
@click.option("--max-samples", type=int, default=None)
@click.option("--num-prompts", type=int, default=1024, show_default=True)
@click.option("--concurrency", type=int, default=1, show_default=True)
@click.option("--base-url", default="http://127.0.0.1:30000", show_default=True)
@click.option("--timeout-seconds", type=int, default=3600, show_default=True)
@click.option("--enable-thinking", is_flag=True)
@click.option("--trust-remote-code", is_flag=True)
@click.option("--output-json", default=None, metavar="PATH")
def benchmark(**options) -> int:
    """Measure throughput and optional speculative-decoding telemetry from a
    running SGLang server."""
    from specforge.benchmarks.sglang import run

    return run(SimpleNamespace(**options))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ``specforge`` CLI and return its exit status.

    Unlike click's standalone mode this never calls ``sys.exit`` itself, so the
    console-script and ``python -m`` entries own process exit while embedded
    callers and tests receive the status as a plain integer.
    """
    try:
        status = cli.main(args=argv, prog_name="specforge", standalone_mode=False)
    except click.ClickException as error:
        error.show()
        return error.exit_code
    return int(status or 0)


if __name__ == "__main__":
    raise SystemExit(main())
