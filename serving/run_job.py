import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from serving.utils import (  # noqa: E402
    build_result,
    copy_video_to_job,
    job_dir,
    load_config,
    make_job_id,
    result_path,
    status_path,
    update_status,
    utc_like_now,
    write_json,
)


def pipeline_run_root(job_path: Path) -> Path:
    return job_path / "pipeline_run"


def build_pipeline_cmd(config: dict[str, Any], input_video: Path, job_path: Path) -> list[str]:
    stage2_root = Path(config["stage2_root"])
    cmd = [
        sys.executable,
        str(stage2_root / "pipeline" / "main.py"),
        "--video",
        str(input_video),
        "--run-id",
        "pipeline_run",
        "--output-root",
        str(job_path),
        "--stage1-root",
        str(config["stage1_root"]),
        "--stage1-pretrain-path",
        str(config["stage1_pretrain_path"]),
        "--stage2-root",
        str(stage2_root),
        "--conda-bin",
        str(config["conda_bin"]),
        "--stage1-env",
        str(config.get("stage1_env", "yowov3")),
        "--stage2-env",
        str(config.get("stage2_env", "qwen35")),
        "--prompt",
        str(config["prompt"]),
        "--vlm-model-path",
        str(config["vlm_model_path"]),
        "--fps",
        str(config.get("fps", 30)),
        "--stage1-window",
        str(config.get("stage1_window", 16)),
        "--stage1-sample-rate",
        str(config.get("stage1_sample_rate", 10)),
        "--stage1-conf-threshold",
        str(config.get("stage1_conf_threshold", 0.3)),
        "--stage1-batch-size",
        str(config.get("stage1_batch_size", 8)),
        "--action-top-k",
        str(config.get("action_top_k", 2)),
        "--stage2-gap",
        str(config.get("stage2_gap", 10)),
        "--stage2-stride",
        str(config.get("stage2_stride", 30)),
        "--max-new-tokens",
        str(config.get("max_new_tokens", 32)),
    ]

    if config.get("include_action_score"):
        cmd.append("--include-action-score")
    if config.get("stage1_make_video"):
        cmd.append("--stage1-make-video")
    if config.get("vlm_limit") is not None:
        cmd.extend(["--vlm-limit", str(config["vlm_limit"])])

    return cmd


def run_job(
    video: Path,
    job_id: str | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    stage2_root = Path(__file__).resolve().parents[1]
    config_path = config_path or stage2_root / "serving" / "config.json"
    config = load_config(config_path)

    job_id = job_id or make_job_id()
    job_path = job_dir(config, job_id)
    job_path.mkdir(parents=True, exist_ok=True)

    if not status_path(job_path).exists():
        write_json(
            status_path(job_path),
            {
                "job_id": job_id,
                "status": "queued",
                "created_at": utc_like_now(),
            },
        )

    input_video = copy_video_to_job(video, job_path)
    update_status(
        job_path,
        job_id=job_id,
        status="running",
        started_at=utc_like_now(),
        input_video=str(input_video),
    )

    cmd = build_pipeline_cmd(config, input_video, job_path)
    try:
        subprocess.run(cmd, cwd=str(config["stage2_root"]), check=True)
        run_root = pipeline_run_root(job_path)
        result = build_result(job_id=job_id, job_path=job_path, run_root=run_root)
        write_json(result_path(job_path), result)
        update_status(
            job_path,
            status="done",
            finished_at=utc_like_now(),
            pipeline_run_root=str(run_root),
        )
        return result
    except Exception as exc:
        update_status(
            job_path,
            status="failed",
            finished_at=utc_like_now(),
            error=str(exc),
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one backend-style AI inference job.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--job-id", default=None, help="Optional fixed job id.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.json"),
        help="Serving config path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_job(
        video=Path(args.video).resolve(),
        job_id=args.job_id,
        config_path=Path(args.config).resolve(),
    )
    print("Saved result:", result_path(job_dir(load_config(Path(args.config)), result["job_id"])))
    print("Overall risk:", result.get("overall_risk_state"))


if __name__ == "__main__":
    main()
