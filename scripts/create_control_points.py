#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import ffmpegio
import kornia
import kornia.feature as KF
import numpy as np
import scipy
import torch
from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd
from scipy.signal import correlate

try:
    import lightglue
except:
    print("Please install lightglue package from https://github.com/cvg/LightGlue")
    exit(1)

_CONTROL_POINTS_LINE = "# control points"


def load_audio_as_tensor(
    audio: Union[str, np.ndarray, torch.Tensor],
    duration_seconds: float,
    verbose: Optional[bool] = False,
) -> Tuple[torch.Tensor, float]:
    """
    Loads audio from a file and returns it as a PyTorch tensor.

    Args:
        audio_file_path (str): Path to the audio file.

    Returns:
        waveform (torch.Tensor): The audio as a tensor [channels, samples].
        sample_rate (int): The sample rate of the audio.
    """
    smaples_per_second, waveform = ffmpegio.audio.read(
        audio, t=duration_seconds, show_log=True
    )
    if verbose:
        # The waveform is now a PyTorch tensor with shape [channels, samples]
        print(f"Waveform shape: {waveform.shape}")
        print(f"Sample rate: {smaples_per_second}")

    return waveform, smaples_per_second


def synchronize_by_audio(
    file1_path: str,
    file2_path: str,
    seconds: int = 15,
    verbose: bool = True,
):
    # Load the videos
    if verbose:
        print("Openning videos...")

    video1_fps, video1_duration = get_video_fps_and_duration(file1_path)
    video2_fps, video2_duration = get_video_fps_and_duration(file2_path)

    seconds = min(seconds, min(video1_duration - 0.5, video2_duration - 0.5))

    video_1_subclip_frame_count = video1_fps * seconds
    video_2_subclip_frame_count = video2_fps * seconds

    audio1, sample_rate1 = load_audio_as_tensor(file1_path, duration_seconds=seconds)
    audio2, sample_rate2 = load_audio_as_tensor(file2_path, duration_seconds=seconds)

    if verbose:
        print("Loading audio...")

    audio_items_per_frame_1 = audio1.shape[0] / video_1_subclip_frame_count
    audio_items_per_frame_2 = audio2.shape[0] / video_2_subclip_frame_count

    assert np.isclose(sample_rate1 / video1_fps, audio_items_per_frame_1)
    assert np.isclose(sample_rate2 / video2_fps, audio_items_per_frame_2)

    # Calculate the cross-correlation of audio1 and audio2
    if verbose:
        print("Calculating cross-correlation...")
    # correlation = np.correlate(audio1[:, 0], audio2[:, 0], mode="full")
    correlation = scipy.signal.correlate(audio1[:, 0], audio2[:, 0], mode="full")
    lag = np.argmax(correlation) - len(audio1) + 1

    # Calculate the time offset in seconds
    fps = video1_fps
    frame_offset = lag / audio_items_per_frame_1
    time_offset = frame_offset / fps

    if verbose:
        print(f"Left frame offset: {frame_offset}")
        print(f"Time offset: {time_offset} seconds")

    # Adjust to the starting frame number in each video (i.e. frame_offset might be a negative number)
    left_frame_offset = frame_offset if frame_offset > 0 else 0
    right_frame_offset = -frame_offset if frame_offset < 0 else 0

    return left_frame_offset, right_frame_offset


def find_sync_offset(audio1, audio2, sample_rate=44100):
    """
    Use cross-correlation (full mode) to find the best relative offset between two audio signals.
    Returns the lag (in samples). A positive lag means that audio1 is delayed relative to audio2.
    """
    corr = correlate(audio1, audio2, mode="full")
    # The lags run from -(len(audio2)-1) to len(audio1)-1.
    lag_arr = np.arange(-len(audio2) + 1, len(audio1))
    best_lag = lag_arr[np.argmax(corr)]
    return best_lag


def get_video_fps_and_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frame_count / fps


def extract_frame(video_path, frame_idx):
    """
    Open a video file with OpenCV and return the frame at frame_idx.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not extract frame {frame_idx} from {video_path}")
    return frame


def evenly_spaced_indices(n_points, n_samples):
    """Generate indices to pick n_samples evenly spaced from n_points."""
    return torch.linspace(0, n_points - 1, steps=n_samples).long()


def select_evenly_spaced(batch, n_samples):
    """
    Selects a subset of points that are most evenly spaced over the Y range.

    Args:
    - batch (torch.Tensor): A tensor of shape (N, 2) where N is the number of points,
                            and the second dimension represents (X, Y) coordinates.
    - n_samples (int): Number of samples to select.

    Returns:
    - torch.Tensor: Indices of the selected points in the original batch.
    """
    # Sort the points based on Y values
    _, sorted_indices = torch.sort(batch[:, 1])

    # Calculate indices that would space the points evenly
    sample_indices = evenly_spaced_indices(batch.size(0), n_samples)

    # Select indices of the original batch
    selected_indices = sorted_indices[sample_indices]

    return selected_indices


def calculate_control_points(
    frame0: Union[str, Path, torch.Tensor],
    frame1: Union[str, Path, torch.Tensor],
    max_control_points: int,
    device: Optional[torch.device] = None,
    max_num_keypoints: int = 2048,
    output_directory: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor: SuperPoint = (
        SuperPoint(
            max_num_keypoints=max_num_keypoints,
        )
        .eval()
        .to(device)
    )  # load the extractor
    matcher = (
        LightGlue(
            features="superpoint",
            # depth_confidence=0.95,
            depth_confidence=-1,
            width_confidence=-1,
            filter_threshold=0.2,
        )
        .eval()
        .to(device)
    )

    # Convert OpenCV BGR frames to RGB.
    img1_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    # Convert to torch tensors (shape [C,H,W]) and scale to [0,1].
    image0 = kornia.image_to_tensor(img1_rgb, keepdim=False).float() / 255.0
    image1 = kornia.image_to_tensor(img2_rgb, keepdim=False).float() / 255.0

    if device is not None:
        if image0.device != device:
            image0 = image0.to(device)
        if image1.device != device:
            image1 = image1.to(device)

    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = (
        feats0["keypoints"],
        feats1["keypoints"],
        matches01["matches"],
    )
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    indices = select_evenly_spaced(m_kpts0, max_control_points)
    m_kpts0 = m_kpts0[indices]
    m_kpts1 = m_kpts1[indices]

    if output_directory:
        axes = viz2d.plot_images([frame0, frame1])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
        viz2d.save_plot(os.path.join(output_directory, "matches.png"))

        kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(
            matches01["prune1"]
        )
        viz2d.plot_images([frame0, frame1])
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
        viz2d.save_plot(os.path.join(output_directory, "keypoints.png"))
    control_points = dict(m_kpts0=m_kpts0, m_kpts1=m_kpts1)
    return control_points


def load_pto_file(file_path: str) -> List[str]:
    """Load the content of a .pto file into a list of lines."""
    with open(file_path, "r") as file:
        lines = file.readlines()
    # trim trailing whitespace
    for i, line in enumerate(lines):
        lines[i] = line.rstrip()
    return lines


def save_pto_file(file_path: str, data: List[str]):
    """Save modified data back to a .pto file."""
    with open(file_path, "w") as file:
        for line in data:
            file.write(f"{line}\n")


def remove_control_points(lines: List[str]) -> Tuple[List[str], int]:
    prev_control_point_count: int = 0
    new_lines: List[str] = []
    for line in lines:
        if line.startswith(_CONTROL_POINTS_LINE):
            continue
        if line.startswith("c "):
            prev_control_point_count += 1
            continue
        new_lines.append(line)

    return (new_lines, prev_control_point_count)


def is_older_than(file1: str, file2: str):
    try:
        mtime1 = os.path.getmtime(file1)
        mtime2 = os.path.getmtime(file2)
        return mtime2 < mtime1
    except OSError:
        return None


def strip(s: str) -> str:
    return re.sub(r"\s+", "", s)


def update_pto_file(pto_file, control_points):
    pts0 = control_points["m_kpts0"]
    pts1 = control_points["m_kpts1"]
    assert len(pts0) == len(pts1)
    print(f"Found {len(pts0)} control points")
    assert len(pts0) and len(pts1)
    pto_lines = load_pto_file(pto_file)
    pto_lines, _ = remove_control_points(lines=pto_lines)
    pto_lines.append("")
    pto_lines.append(_CONTROL_POINTS_LINE)

    def _to_hugin_decimal(val: str) -> str:
        val = float(val)
        if val == float(int(val)):
            return f"{int(val)}"
        return f"{val:.12f}"

    for i in range(len(pts0)):
        point0 = [float(c) for c in pts0[i]]
        point1 = [float(c) for c in pts1[i]]
        line = f"c n0 N1 x{_to_hugin_decimal(point0[0])} y{_to_hugin_decimal(point0[1])} X{_to_hugin_decimal(point1[0])} Y{_to_hugin_decimal(point1[1])} t0"
        pto_lines.append(line)
    save_pto_file(file_path=pto_file, data=pto_lines)
    print("Done with control points")


def configure_stitching(
    frame1: str,
    frame2: str,
    directory: str,
    force: bool = False,
    skip_if_exists: bool = False,
    fov: float = 108,  # 108 is Gopro Wide
    max_control_points: int = 240,
    device: Optional[torch.device] = None,
) -> str:
    left_image_file = "left.png"
    right_image_file = "right.png"
    f1 = os.path.join(directory, left_image_file)
    f2 = os.path.join(directory, right_image_file)
    cv2.imwrite(f1, frame1)
    cv2.imwrite(f2, frame2)
    project_file_path = os.path.join(directory, "hm_project.pto")
    pto_path = Path(project_file_path)
    dir_name = pto_path.parent
    hm_project = project_file_path
    autooptimiser_out = os.path.join(dir_name, "autooptimiser_out.pto")
    assert autooptimiser_out != hm_project
    if skip_if_exists and (
        os.path.exists(project_file_path)
        and os.path.exists(autooptimiser_out)
        and not is_older_than(project_file_path, autooptimiser_out)
    ):
        print(
            f"Project file already exists (skipping project creation): {autooptimiser_out}"
        )
        return True

    curr_dir = os.getcwd()
    os.chdir(dir_name)
    try:
        if not os.path.exists(hm_project) or force:
            cmd = [
                "pto_gen",
                "-p",
                "0",
                "-o",
                hm_project,
                "-f",
                str(fov),
                left_image_file,
                right_image_file,
            ]
            cmd_str = " ".join(cmd)
            os.system(cmd_str)

        control_points: Dict[str, torch.Tensor] = calculate_control_points(
            frame1,
            frame2,
            max_control_points=max_control_points,
            device=device,
            max_num_keypoints=2048,
            output_directory=directory,
        )
        update_pto_file(project_file_path, control_points)

        # autooptimiser (RANSAC?)
        cmd = [
            "autooptimiser",
            "-a",
            "-m",
            "-l",
            "-s",
            "-o",
            autooptimiser_out,
            hm_project,
        ]
        os.system(" ".join(cmd))

        # Output mapping files
        cmd = [
            "nona",
            "-m",
            "TIFF_m",
            "-z",
            "NONE",
            "--bigtiff",
            "-c",
            "-o",
            "mapping_",
            autooptimiser_out,
        ]
        os.system(" ".join(cmd))

        cmd = [
            "enblend",
            "--save-masks=seam_file.png",
            "-o",
            os.path.join(dir_name, "panorama.tif"),
            os.path.join(dir_name, "mapping_????.tif"),
        ]
        os.system(" ".join(cmd))
    finally:
        os.chdir(curr_dir)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize two videos using audio cross-correlation, extract sync frames, "
        "compute control points using Kornia SuperPoint+LightGlue, and update a Hugin PTO file."
    )
    parser.add_argument("video1", help="Path to first video file")
    parser.add_argument("video2", help="Path to second video file")
    parser.add_argument("pto_file", help="Path to the Hugin PTO file to update")
    args = parser.parse_args()

    lfo, rfo = synchronize_by_audio(args.video1, args.video2)

    print("Extracting frames at the sync points...")
    frame1 = extract_frame(args.video1, lfo)
    frame2 = extract_frame(args.video2, rfo)

    # If the stitching looks bad, try adjusting max_control_points.
    # Numbers to try: 7, 20 50 150, 500, 1000, etc.
    print("Running SuperPoint and LightGlue to obtain control point matches...")
    configure_stitching(
        frame1, frame2, max_control_points=240, directory=Path(args.video1).parent
    )


if __name__ == "__main__":
    main()
