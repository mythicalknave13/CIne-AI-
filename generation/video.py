from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import ffmpeg

from config.settings import Settings

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".webm"}
logger = logging.getLogger("uvicorn.error")
CLIP_RENDER_WORKERS = 4


def _create_ken_burns_clip(
    image_path: Path,
    output_path: Path,
    duration_seconds: float,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    clip_index: int = 0,
) -> None:
    """Fast Ken Burns using scale + animated crop (pan only).

    Instead of zoompan (which re-renders every pixel per frame and takes
    hours), we pre-scale the image 20 % larger and linearly slide the
    crop window across the surface.  This runs in *seconds* per clip.
    """
    start = time.perf_counter()
    scale_factor = 1.2
    sw = int(width * scale_factor)
    sh = int(height * scale_factor)

    # Four pan directions for visual variety across clips.
    directions = [
        (0, 0, sw - width, sh - height),         # top-left → bottom-right
        (sw - width, 0, 0, sh - height),          # top-right → bottom-left
        (0, sh - height, sw - width, 0),           # bottom-left → top-right
        (sw - width, sh - height, 0, 0),           # bottom-right → top-left
    ]
    sx, sy, ex, ey = directions[clip_index % len(directions)]

    # FFmpeg expressions – linear interpolation over time.
    x_expr = f"{sx}+({ex}-{sx})*t/{duration_seconds}"
    y_expr = f"{sy}+({ey}-{sy})*t/{duration_seconds}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    (
        ffmpeg.input(str(image_path), loop=1, t=duration_seconds)
        .filter("scale", sw, sh, force_original_aspect_ratio="increase")
        .filter("crop", sw, sh)          # centre-crop to exact sw×sh
        .filter("crop", width, height, x=x_expr, y=y_expr)  # animated pan
        .filter("fade", type="in", start_time=0, duration=0.5)
        .filter("fade", type="out", start_time=duration_seconds - 0.8, duration=0.8)
        .output(
            str(output_path),
            vcodec="libx264",
            pix_fmt="yuv420p",
            preset="fast",
            movflags="+faststart",
            r=fps,
        )
        .overwrite_output()
        .run(quiet=True)
    )
    logger.info(
        "TIMING ffmpeg ken_burns clip=%s duration=%.2fs render=%.2fs resolution=%dx%d",
        output_path.name,
        duration_seconds,
        time.perf_counter() - start,
        width,
        height,
    )


def create_ken_burns_clip(
    image_path: Path,
    output_path: Path,
    duration_seconds: float,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    clip_index: int = 0,
) -> Path:
    _create_ken_burns_clip(
        image_path=image_path,
        output_path=output_path,
        duration_seconds=duration_seconds,
        width=width,
        height=height,
        fps=fps,
        clip_index=clip_index,
    )
    return output_path


def create_black_card(
    text: str,
    duration_seconds: float,
    output_path: Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    font_size: int = 48,
    fade_duration: float = 0.8,
) -> Path:
    start = time.perf_counter()
    alpha_expr = (
        f"if(lt(t,{fade_duration}),t/{fade_duration},"
        f"if(lt(t,{max(duration_seconds - fade_duration, 0.0)}),1,({duration_seconds}-t)/{fade_duration}))"
    )

    stream = ffmpeg.input(
        f"color=c=black:s={width}x{height}:r={fps}",
        f="lavfi",
        t=duration_seconds,
    )
    if text:
        stream = stream.drawtext(
            text=text,
            fontsize=font_size,
            fontcolor="white",
            x="(w-text_w)/2",
            y="(h-text_h)/2",
            alpha=alpha_expr,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg.output(
            stream,
            str(output_path),
            vcodec="libx264",
            pix_fmt="yuv420p",
            preset="fast",
            movflags="+faststart",
            r=fps,
        )
        .overwrite_output()
        .run(quiet=True)
    )
    logger.info(
        "TIMING ffmpeg black_card duration=%.2fs render=%.2fs output=%s resolution=%dx%d",
        duration_seconds,
        time.perf_counter() - start,
        output_path,
        width,
        height,
    )
    return output_path


def _prepare_existing_video_clip(
    input_path: Path,
    output_path: Path,
    duration_seconds: float,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
) -> None:
    start = time.perf_counter()
    stream = (
        ffmpeg
        .input(str(input_path))
        .video
        .filter("fps", fps)
        .filter("scale", width, height, force_original_aspect_ratio="increase")
        .filter("crop", width, height)
        .filter("setpts", "PTS-STARTPTS")
        .filter("tpad", stop_mode="clone", stop_duration=duration_seconds)
        .filter("trim", duration=duration_seconds)
    )
    (
        stream.output(
            str(output_path),
            vcodec="libx264",
            pix_fmt="yuv420p",
            preset="fast",
            movflags="+faststart",
            r=fps,
            t=duration_seconds,
        )
        .overwrite_output()
        .run(quiet=True)
    )
    logger.info(
        "TIMING ffmpeg existing_clip clip=%s duration=%.2fs render=%.2fs",
        output_path.name,
        duration_seconds,
        time.perf_counter() - start,
    )


def _render_one_clip(
    idx: int,
    media_path: Path,
    clip_dir: Path,
    clip_duration: float,
    width: int,
    height: int,
    fps: int,
) -> Path:
    clip_path = clip_dir / f"clip_{idx:02d}.mp4"
    if media_path.suffix.lower() in VIDEO_SUFFIXES:
        _prepare_existing_video_clip(
            input_path=media_path,
            output_path=clip_path,
            duration_seconds=clip_duration,
            width=width,
            height=height,
            fps=fps,
        )
    else:
        _create_ken_burns_clip(
            image_path=media_path,
            output_path=clip_path,
            duration_seconds=clip_duration,
            width=width,
            height=height,
            fps=fps,
            clip_index=idx - 1,
        )
    return clip_path


def render_all_clips(
    media_paths: Iterable[Path],
    clip_dir: Path,
    scene_duration: float,
    clip_durations: Optional[Iterable[float]] = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> List[Path]:
    ordered_media = [Path(path) for path in media_paths]
    if not ordered_media:
        return []
    durations = list(clip_durations or [scene_duration] * len(ordered_media))
    if len(durations) != len(ordered_media):
        raise ValueError("clip_durations must match media_paths length.")

    clip_dir.mkdir(parents=True, exist_ok=True)
    max_workers = min(CLIP_RENDER_WORKERS, len(ordered_media))
    clip_paths: List[Optional[Path]] = [None] * len(ordered_media)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _render_one_clip,
                idx,
                media_path,
                clip_dir,
                durations[idx - 1],
                width,
                height,
                fps,
            ): idx
            for idx, media_path in enumerate(ordered_media, start=1)
        }
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            clip_paths[idx - 1] = future.result()
            completed += 1
            if progress_callback:
                progress_callback(
                    f"Rendering clips… {completed}/{len(ordered_media)}",
                    75 + int(15 * completed / len(ordered_media)),
                )

    return [path for path in clip_paths if path is not None]


def _write_concat_list(video_paths: Iterable[Path], output_dir: Path) -> Path:
    concat_list_path = output_dir / "clips.txt"
    concat_list_path.write_text(
        "".join(f"file '{Path(path).resolve()}'\n" for path in video_paths),
        encoding="utf-8",
    )
    return concat_list_path


def _create_xfade_transition_clip(
    from_clip: Path,
    to_clip: Path,
    output_path: Path,
    *,
    duration_seconds: float,
    width: int,
    height: int,
    fps: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-sseof",
            f"-{duration_seconds:.3f}",
            "-i",
            str(from_clip),
            "-t",
            f"{duration_seconds:.3f}",
            "-i",
            str(to_clip),
            "-filter_complex",
            (
                f"[0:v]fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},setpts=PTS-STARTPTS[a];"
                f"[1:v]fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},trim=duration={duration_seconds:.3f},setpts=PTS-STARTPTS[b];"
                f"[a][b]xfade=transition=fade:duration={duration_seconds:.3f}:offset=0[v]"
            ),
            "-map",
            "[v]",
            "-t",
            f"{duration_seconds:.3f}",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def _generate_silence_audio(
    output_path: Path,
    *,
    duration_seconds: float,
    sample_rate: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sample_rate}:cl=stereo",
            "-t",
            f"{duration_seconds:.3f}",
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def build_ambient_audio_track(
    *,
    media_paths: Iterable[Path],
    media_has_native_audio: Iterable[bool],
    output_dir: Path,
    sample_rate: int,
    media_durations: Iterable[float],
    intro_duration_seconds: float = 0.0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ambient_parts: List[Path] = []
    durations = list(media_durations)
    paths = [Path(path) for path in media_paths]
    audio_flags = list(media_has_native_audio)
    if not (len(paths) == len(audio_flags) == len(durations)):
        raise ValueError("media_paths, media_has_native_audio, and media_durations must align.")

    if intro_duration_seconds > 0:
        ambient_parts.append(
            _generate_silence_audio(
                output_dir / "ambient_intro.wav",
                duration_seconds=intro_duration_seconds,
                sample_rate=sample_rate,
            )
        )

    for index, (video_path, has_audio, clip_duration) in enumerate(zip(paths, audio_flags, durations), start=1):
        part_path = output_dir / f"ambient_scene_{index:02d}.wav"
        if has_audio:
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(video_path),
                        "-vn",
                        "-af",
                        "apad",
                        "-ac",
                        "2",
                        "-ar",
                        str(sample_rate),
                        "-t",
                        f"{clip_duration:.3f}",
                        str(part_path),
                    ],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                _generate_silence_audio(
                    part_path,
                    duration_seconds=clip_duration,
                    sample_rate=sample_rate,
                )
        else:
            _generate_silence_audio(
                part_path,
                duration_seconds=clip_duration,
                sample_rate=sample_rate,
            )
        ambient_parts.append(part_path)

    ambient_path = output_dir / "ambient_full.wav"
    concat_list_path = _write_concat_list(ambient_parts, output_dir)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c",
            "copy",
            str(ambient_path),
        ],
        check=True,
        capture_output=True,
    )
    logger.info(
        "TIMING ffmpeg ambient_track duration=%.2fs output=%s parts=%d",
        sum(durations) + intro_duration_seconds,
        ambient_path,
        len(ambient_parts),
    )
    return ambient_path


def _run_concat_mux(
    concat_list_path: Path,
    mixed_audio_path: Path,
    final_path: Path,
    *,
    copy_video: bool,
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-i",
        str(mixed_audio_path),
        "-c:v",
        "copy" if copy_video else "libx264",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(final_path),
    ]
    if not copy_video:
        command[12:12] = ["-preset", "fast", "-pix_fmt", "yuv420p"]

    subprocess.run(command, check=True, capture_output=True)


def assemble_video(
    scene_images: Iterable[Path],
    mixed_audio_path: Path,
    settings: Settings,
    output_dir: Path,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    prefix_clips: Iterable[Path] | None = None,
    scene_duration_seconds: float | None = None,
    clip_durations: Iterable[float] | None = None,
    transition_after: dict[int, float] | None = None,
) -> Path:
    total_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    media_paths: List[Path] = [Path(path) for path in scene_images]
    if not media_paths:
        raise ValueError("No scene media provided.")

    intro_paths: List[Path] = [Path(path) for path in (prefix_clips or [])]
    scene_duration = scene_duration_seconds or (settings.film_duration_seconds / len(media_paths))
    durations = list(clip_durations or [scene_duration] * len(media_paths))
    if len(durations) != len(media_paths):
        raise ValueError("clip_durations must match scene_images length.")
    clip_dir = output_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_total_start = time.perf_counter()
    clip_paths = render_all_clips(
        media_paths=media_paths,
        clip_dir=clip_dir,
        scene_duration=scene_duration,
        clip_durations=durations,
        progress_callback=progress_callback,
    )
    logger.info(
        "TIMING ffmpeg clip_stage total=%.2fs clips=%d scene_duration=%.2fs",
        time.perf_counter() - clip_total_start,
        len(media_paths),
        scene_duration,
    )

    if progress_callback:
        progress_callback("Assembling final film…", 92)

    ordered_video_paths = list(intro_paths)
    transition_map = {int(k): float(v) for k, v in (transition_after or {}).items()}
    for index, clip_path in enumerate(clip_paths, start=1):
        ordered_video_paths.append(clip_path)
        transition_duration = float(transition_map.get(index, 0.0))
        if transition_duration > 0 and index < len(clip_paths):
            transition_path = clip_dir / f"transition_{index:02d}_{index + 1:02d}.mp4"
            _create_xfade_transition_clip(
                clip_path,
                clip_paths[index],
                transition_path,
                duration_seconds=transition_duration,
                width=1280,
                height=720,
                fps=24,
            )
            ordered_video_paths.append(transition_path)
    final_path = output_dir / "final_film.mp4"
    concat_list_path = _write_concat_list(ordered_video_paths, output_dir)
    final_mux_start = time.perf_counter()
    stream_copy = True
    try:
        _run_concat_mux(
            concat_list_path=concat_list_path,
            mixed_audio_path=mixed_audio_path,
            final_path=final_path,
            copy_video=True,
        )
    except subprocess.CalledProcessError as exc:
        stream_copy = False
        stderr = (exc.stderr or b"").decode("utf-8", errors="ignore").strip().splitlines()
        preview = stderr[-1] if stderr else "unknown ffmpeg error"
        logger.warning("Final mux stream-copy failed, re-encoding video as fallback. (%s)", preview)
        _run_concat_mux(
            concat_list_path=concat_list_path,
            mixed_audio_path=mixed_audio_path,
            final_path=final_path,
            copy_video=False,
        )
    logger.info(
        "TIMING ffmpeg final_assembly mux=%.2fs total=%.2fs output=%s resolution=1280x720 stream_copy=%s",
        time.perf_counter() - final_mux_start,
        time.perf_counter() - total_start,
        final_path,
        stream_copy,
    )

    return final_path


def assemble_film(
    scene_images: Iterable[Path],
    mixed_audio_path: Path,
    settings: Settings,
    output_dir: Path,
    prefix_clips: Iterable[Path] | None = None,
    scene_duration_seconds: float | None = None,
) -> dict:
    final_path = assemble_video(
        scene_images=scene_images,
        mixed_audio_path=mixed_audio_path,
        settings=settings,
        output_dir=output_dir,
        prefix_clips=prefix_clips,
        scene_duration_seconds=scene_duration_seconds,
    )
    return {
        "video_path": str(final_path),
    }
