#!/usr/bin/env python3
"""
Synchronize N videos/images by audio (all to the first),
extract frames, compute pairwise control points, and build a Hugin PTO.
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import ffmpegio
import kornia
import kornia.feature as KF
import numpy as np
import scipy.signal
import torch
import yaml
from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import rbd
from scipy.signal import correlate

_CONTROL_POINTS_LINE: str = "# control points"
TorchTensor = torch.Tensor


def is_video_file(path: str) -> bool:
    """
    Rudimentary check for video file extensions.
    """
    ext: str = Path(path).suffix.lower()
    return ext in {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".mpg", ".mpeg"}


def load_audio_as_tensor(
    audio: Union[str, np.ndarray, torch.Tensor],
    duration_seconds: float,
    verbose: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Load audio from a file (or array/tensor) and return (waveform, sample_rate).
    """
    sample_rate, waveform = ffmpegio.audio.read(
        audio, t=duration_seconds, show_log=True
    )
    if verbose:
        # waveform shape: [channels, samples]
        print(f"[audio] {audio}: sr={sample_rate}, shape={waveform.shape}")
    return waveform, float(sample_rate)


def get_video_fps_and_duration(video_path: str) -> Tuple[float, float]:
    """
    Retrieve the FPS and duration of a video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Can't open video: {video_path}")
    fps: float = cap.get(cv2.CAP_PROP_FPS)
    frame_count: float = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fps, frame_count / fps


# def synchronize_by_audio(
#     ref_path: str,
#     tgt_path: str,
#     seconds: float = 15.0,
#     verbose: bool = True,
# ) -> float:
#     """
#     Return frame_offset for tgt so it aligns to ref.
#     """
#     fps1, dur1 = get_video_fps_and_duration(ref_path)
#     fps2, dur2 = get_video_fps_and_duration(tgt_path)
#     use_secs: float = min(seconds, dur1 - 0.5, dur2 - 0.5)
#     wf1, sr1 = load_audio_as_tensor(ref_path, use_secs, verbose)
#     wf2, sr2 = load_audio_as_tensor(tgt_path, use_secs, verbose)

#     # use first channel if multichannel
#     a1: np.ndarray = wf1[0] if wf1.ndim > 1 else wf1
#     a2: np.ndarray = wf2[0] if wf2.ndim > 1 else wf2

#     corr: np.ndarray = correlate(a1, a2, mode="full")
#     lag: int = int(np.argmax(corr) - len(a1) + 1)

#     items_per_frame: float = len(a1) / (fps1 * use_secs)
#     frame_offset: float = lag / items_per_frame
#     if verbose:
#         print(f"[sync] {tgt_path} lags by {frame_offset:.1f} frames")
#     return frame_offset


def synchronize_by_audio(
    file1_path: str,
    file2_path: str,
    seconds: int = 15,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    Synchronize two video files by comparing their audio tracks using cross-correlation.

    The function extracts a short audio clip from each video, computes their cross-correlation,
    and calculates the frame offset between the two videos.

    Args:
        file1_path: Path to the first video file.
        file2_path: Path to the second video file.
        seconds: Duration (in seconds) of audio to use for synchronization.
        verbose: If True, prints progress messages.

    Returns:
        A tuple (left_frame_offset, right_frame_offset) representing the number of frames to
        skip in each video so that they are synchronized. The offsets are returned as integers.
    """

    if verbose:
        print("Opening videos...")

    # Get video FPS and duration for both videos.
    video1_fps, video1_duration = get_video_fps_and_duration(file1_path)
    video2_fps, video2_duration = get_video_fps_and_duration(file2_path)

    # Ensure we do not exceed the available duration (leaving a 0.5 sec margin).
    seconds = min(seconds, min(video1_duration - 0.5, video2_duration - 0.5))

    video_1_subclip_frame_count: float = video1_fps * seconds
    video_2_subclip_frame_count: float = video2_fps * seconds

    if verbose:
        print("Loading audio...")

    # Load audio as tensor. The waveform is of shape [channels, samples].
    audio1, sample_rate1 = load_audio_as_tensor(file1_path, duration_seconds=seconds)
    audio2, sample_rate2 = load_audio_as_tensor(file2_path, duration_seconds=seconds)

    # Calculate number of audio samples per video frame.
    # Note: waveform shape is [channels, samples] so we use axis 1 for number of samples.
    audio_items_per_frame_1: float = audio1.shape[0] / video_1_subclip_frame_count
    audio_items_per_frame_2: float = audio2.shape[0] / video_2_subclip_frame_count

    # Check that the computed samples per frame match the expected value.
    assert np.isclose(sample_rate1 / video1_fps, audio_items_per_frame_1)
    assert np.isclose(sample_rate2 / video2_fps, audio_items_per_frame_2)

    if verbose:
        print("Calculating cross-correlation...")

    sum1 = np.sum(audio1[:, 0])
    sum2 = np.sum(audio2[:, 0])

    # Use only the first channel for correlation.
    correlation: np.ndarray = scipy.signal.correlate(
        audio1[:, 0], audio2[:, 0], mode="full"
    )
    sumc = np.sum(correlation)
    # Compute lag: subtract the length of the signal (using axis 1 length)
    lag: int = np.argmax(correlation) - audio1.shape[0] + 1

    # Convert lag (in audio samples) to frame offset.
    fps = video1_fps
    frame_offset: float = lag / audio_items_per_frame_1
    time_offset: float = frame_offset / fps

    if verbose:
        print(f"Calculated frame offset: {frame_offset}")
        print(f"Equivalent time offset: {time_offset} seconds")

    # Determine starting frame for each video.
    left_frame_offset: float = frame_offset if frame_offset > 0 else 0
    right_frame_offset: float = -frame_offset if frame_offset < 0 else 0

    return left_frame_offset, right_frame_offset


def extract_frame(video_path: str, frame_idx: Optional[float]) -> np.ndarray:
    """
    Extract a single frame from a video or load an image.
    """
    if video_path.lower().endswith((".png", ".jpg", ".jpeg")):
        img: Optional[np.ndarray] = cv2.imread(video_path)
        if img is None:
            raise RuntimeError(f"Can't load image {video_path}")
        return img
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx or 0)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Can't grab frame {frame_idx} from {video_path}")
    return frame


def evenly_spaced_indices(n_points: int, n_samples: int) -> torch.Tensor:
    """
    Pick n_samples indices evenly spaced from 0 to n_points-1.
    """
    return torch.linspace(0, n_points - 1, steps=n_samples).long()


def select_evenly_spaced(batch: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Select indices of keypoints evenly spaced along Y.
    """
    _, sorted_idx = torch.sort(batch[:, 1])
    samples_idx: torch.Tensor = evenly_spaced_indices(batch.size(0), n_samples)
    return sorted_idx[samples_idx]


def calculate_control_points(
    im0: np.ndarray,
    im1: np.ndarray,
    max_control_points: int,
    device: Optional[torch.device] = None,
    max_num_keypoints: int = 2048,
) -> Tuple[TorchTensor, TorchTensor]:
    """
    Compute matched keypoints between two frames.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor: SuperPoint = (
        SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
    )
    matcher: LightGlue = (
        LightGlue(features="superpoint", filter_threshold=0.2).eval().to(device)
    )

    rgb0: np.ndarray = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    rgb1: np.ndarray = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    t0: torch.Tensor = (kornia.image_to_tensor(rgb0, False).float() / 255.0).to(device)
    t1: torch.Tensor = (kornia.image_to_tensor(rgb1, False).float() / 255.0).to(device)

    feats0 = extractor.extract(t0)
    feats1 = extractor.extract(t1)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]

    kpts0: torch.Tensor = feats0["keypoints"]
    kpts1: torch.Tensor = feats1["keypoints"]
    matches: torch.Tensor = matches01["matches"]

    m_kpts0: torch.Tensor = kpts0[matches[:, 0]]
    m_kpts1: torch.Tensor = kpts1[matches[:, 1]]

    if m_kpts0.size(0) > max_control_points:
        idx = select_evenly_spaced(m_kpts0, max_control_points)
        m_kpts0 = m_kpts0[idx]
        m_kpts1 = m_kpts1[idx]

    return m_kpts0.cpu(), m_kpts1.cpu()


def load_pto_lines(path: str) -> List[str]:
    """
    Load lines from a PTO file.
    """
    with open(path, "r") as f:
        lines = [L.rstrip() for L in f]
    return lines


def save_pto_lines(path: str, lines: List[str]) -> None:
    """
    Save lines to a PTO file.
    """
    with open(path, "w") as f:
        for L in lines:
            f.write(L + "\n")


def remove_control_points(lines: List[str]) -> Tuple[List[str], int]:
    """
    Remove existing 'c ' control-point lines.
    """
    out: List[str] = []
    cnt: int = 0
    for L in lines:
        if L.startswith(_CONTROL_POINTS_LINE):
            continue
        if L.startswith("c "):
            cnt += 1
            continue
        out.append(L)
    return out, cnt


def update_pto_file_multi(
    pto_file: str,
    image_names: List[str],
    cps: List[Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """
    Update a PTO with control points for each adjacent pair.
    """
    lines, _ = remove_control_points(load_pto_lines(pto_file))
    lines.append("")
    lines.append(_CONTROL_POINTS_LINE)

    def fmt(v: float) -> str:
        s = f"{v:.12f}"
        return s.rstrip("0").rstrip(".")

    for i, (mk0, mk1) in enumerate(cps):
        for (x0, y0), (x1, y1) in zip(mk0.tolist(), mk1.tolist()):
            lines.append(
                f"c n{i} N{i+1} x{fmt(x0)} y{fmt(y0)} X{fmt(x1)} Y{fmt(y1)} t0"
            )

    save_pto_lines(pto_file, lines)
    total = sum(cp[0].size(0) for cp in cps)
    print(f"[PTO] wrote {total} control points")


def configure_stitching_multi(
    frames: List[np.ndarray],
    image_files: List[str],
    directory: str,
    force: bool = True,
    max_control_points: int = 240,
    fov: float = 108.0,
    scale: Optional[float] = None,
) -> None:
    """
    Save frames, generate PTO, compute cps & visuals, run Hugin tools.
    """
    # save frames
    names: List[str] = [f"i{i}.png" for i in range(len(frames))]
    for im, name in zip(frames, names):
        cv2.imwrite(os.path.join(directory, name), im)

    pto: str = os.path.join(directory, "hm_project.pto")
    autooptimiser: str = os.path.join(directory, "autooptimiser.pto")

    if force or not os.path.exists(pto):
        cmd = ["pto_gen", "-o", pto, "-f", str(fov)] + [
            os.path.join(directory, n) for n in names
        ]
        os.system(" ".join(cmd))

    # compute CPS and save visuals
    cps: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(len(frames) - 1):
        cp0, cp1 = calculate_control_points(
            frames[i],
            frames[i + 1],
            max_control_points=max_control_points,
        )
        cps.append((cp0, cp1))

        pair_dir: str = os.path.join(directory, f"pair_{i}_{i+1}")
        os.makedirs(pair_dir, exist_ok=True)

        # matches
        axes = viz2d.plot_images([frames[i], frames[i + 1]])
        viz2d.plot_matches(cp0, cp1, color="lime", lw=0.2)
        viz2d.save_plot(os.path.join(pair_dir, "matches.png"))

        # keypoints
        axes = viz2d.plot_images([frames[i], frames[i + 1]])
        viz2d.plot_keypoints([cp0, cp1], colors=["red", "blue"], ps=5)
        viz2d.save_plot(os.path.join(pair_dir, "keypoints.png"))

    # update PTO
    update_pto_file_multi(pto, names, cps)

    # autooptimiser
    cmd = ["autooptimiser", "-a", "-m", "-l", "-s", "-o", autooptimiser, pto]
    if scale is not None:
        cmd += ["-x", str(scale)]
    os.system(" ".join(cmd))

    # nona & enblend
    os.system(
        f"nona --bigtiff -m TIFF_m -z NONE -c -o {os.path.join(directory, 'mapping_')} {autooptimiser}"
    )
    os.system(
        f"multiblend --save-seams={os.path.join(directory, 'seam_file.png')} -o {os.path.join(directory, 'panorama.tif')} {os.path.join(directory, 'mapping_????.tif')}"
    )


def compute_global_offsets(pairs: List[Tuple[float, float]]) -> List[float]:
    """
    Given a list of pairwise offsets [(off0, off1), (off1, off2), ...],
    return a list of N global offsets [g0, g1, ..., gN-1] such that:
      - g0 == 0
      - gi+1 == gi + (off_{i+1} - off_i)
      - all gi >= 0 (shifted so the earliest is zero)
    """
    # Number of videos
    n = len(pairs) + 1
    # Step 1: compute pairwise delays
    delays = [off1 - off0 for off0, off1 in pairs]

    # Step 2: accumulate
    globals_ = [0.0]
    for d in delays:
        globals_.append(globals_[-1] + d)

    # Shift so the minimum is zero
    min_off = min(globals_)
    if min_off < 0:
        globals_ = [g - min_off for g in globals_]

    return globals_


def main() -> None:
    """
    Parse args, sync inputs, extract frames, run stitching.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to videos or images (at least 2)",
    )
    parser.add_argument(
        "--max-control-points",
        type=int,
        default=500,
        help="Maximum number of control points",
    )
    parser.add_argument("--scale", type=float, default=None, help="Panorama scale")
    args = parser.parse_args()

    files: List[str] = args.inputs
    if len(files) < 2:
        print("Need at least 2 inputs")
        exit(1)

    # compute offsets to files[0]
    offsets: List[float] = []
    ref: str = files[0]
    for tgt in files[1:]:
        if is_video_file(ref) and is_video_file(tgt):
            offsets.append(synchronize_by_audio(ref, tgt))
    assert len(offsets) == len(files) - 1

    final_offsets: List[float] = compute_global_offsets(offsets)

    # extract frames
    frames: List[np.ndarray] = [
        extract_frame(path, off) for path, off in zip(files, final_offsets)
    ]

    configure_stitching_multi(
        frames,
        files,
        directory=str(Path(files[0]).parent),
        force=True,
        max_control_points=args.max_control_points,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
