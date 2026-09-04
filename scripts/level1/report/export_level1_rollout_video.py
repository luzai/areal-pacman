from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from maapacman.env import Action, PygamePacmanEnv, PygamePacmanEnvConfig


def compose_frame(
    image: object,
    *,
    title: str,
    step: int,
    total_steps: int,
    action: str,
    score: int,
    open_actions: list[str],
) -> Image.Image:
    game = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
    scale = min(960 / game.width, 720 / game.height)
    game = game.resize(
        (round(game.width * scale), round(game.height * scale)),
        Image.Resampling.NEAREST,
    )
    canvas_width = game.width + (game.width % 2)
    canvas_height = game.height + 92
    canvas_height += canvas_height % 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), (15, 18, 25))
    canvas.paste(game, (0, 92))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((18, 14), title, fill=(244, 246, 250), font=font)
    draw.text(
        (18, 40),
        f"step {step}/{total_steps}   action {action or '-'}   "
        f"score {score}   OPEN {','.join(open_actions)}",
        fill=(185, 198, 220),
        font=font,
    )
    draw.text(
        (18, 65),
        "Qwen3.5-9B initial rollout | no-thinking | dynamic action mask",
        fill=(90, 214, 154),
        font=font,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    args = parser.parse_args()

    episode = json.loads(args.trajectory.read_text(encoding="utf-8"))
    trajectory = episode["trajectory"]
    config = PygamePacmanEnvConfig(
        level=int(episode["level"]),
        curriculum=int(episode["curriculum"]),
        max_steps=int(episode["max_steps"]),
        video_driver="dummy",
        audio_driver="dummy",
    )

    frames: list[Image.Image] = []
    with PygamePacmanEnv(config) as env:
        image, info = env.reset(seed=int(episode["seed"]))
        frames.append(
            compose_frame(
                image,
                title=f"Initial rollout: {episode['id']} | final score {episode['final_score']}",
                step=0,
                total_steps=len(trajectory),
                action="",
                score=int(info["score"]),
                open_actions=list(env.snapshot().get("open") or []),
            )
        )
        for index, recorded in enumerate(trajectory, start=1):
            image, _, terminated, truncated, info = env.step(
                Action(recorded["action"])
            )
            frames.append(
                compose_frame(
                    image,
                    title=f"Initial rollout: {episode['id']} | final score {episode['final_score']}",
                    step=index,
                    total_steps=len(trajectory),
                    action=recorded["action"],
                    score=int(info["score"]),
                    open_actions=list(env.snapshot().get("open") or []),
                )
            )
            if bool(terminated or truncated) != bool(
                recorded["terminated"] or recorded["truncated"]
            ):
                raise RuntimeError(f"replay terminal-state mismatch at step {index}")

    hold_frames = max(1, round(args.fps * args.hold_seconds))
    frames = [frames[0]] * hold_frames + frames + [frames[-1]] * hold_frames
    args.output.parent.mkdir(parents=True, exist_ok=True)
    width, height = frames[0].size
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(
        json.dumps(
            {
                "trajectory": str(args.trajectory),
                "output": str(args.output),
                "frames": len(frames),
                "fps": args.fps,
                "duration_seconds": len(frames) / args.fps,
                "final_score": episode["final_score"],
            }
        )
    )


if __name__ == "__main__":
    main()
