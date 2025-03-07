#!/usr/bin/env python3
"""
This script synchronizes two videos using audio cross-correlation, extracts the corresponding frames,
computes control points using LightGlue and SuperPoint (from the lightglue package), and updates a Hugin
PTO file with the newly computed control points.
"""

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
import scipy.signal
import torch
import yaml
from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd
from scipy.signal import correlate

# Ensure that the lightglue package is available.
try:
    import lightglue  # noqa: F401
except ImportError:
    print("Please install the lightglue package from https://github.com/cvg/LightGlue")
    exit(1)

# Constant marker used in PTO files to denote control points.
_CONTROL_POINTS_LINE = "# control points"


def load_audio_as_tensor(
    audio: Union[str, np.ndarray, torch.Tensor],
    duration_seconds: float,
    verbose: Optional[bool] = False,
) -> Tuple[torch.Tensor, float]:
    """
    Load audio from a file (or other supported source) using ffmpegio and return it as a PyTorch tensor.

    Args:
        audio: Either a file path or an array/tensor representing audio.
        duration_seconds: Duration (in seconds) to read from the audio.
        verbose: If True, prints additional debug information.

    Returns:
        A tuple (waveform, sample_rate) where waveform is a tensor of shape [channels, samples]
        and sample_rate is the number of samples per second.
    """
    sample_rate, waveform = ffmpegio.audio.read(
        audio, t=duration_seconds, show_log=True
    )
    if verbose:
        # waveform shape: [channels, samples]
        print(f"Waveform shape: {waveform.shape}")
        print(f"Sample rate: {sample_rate}")
    return waveform, sample_rate


def get_video_fps_and_duration(video_path: str) -> Tuple[float, float]:
    """
    Retrieve the frames-per-second (FPS) and duration (in seconds) of a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        A tuple (fps, duration) where duration is computed as frame_count / fps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        exit(1)
    fps: float = cap.get(cv2.CAP_PROP_FPS)
    frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frame_count / fps


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
    Extract a single frame from a video file using OpenCV.

    Args:
        video_path: Path to the video file.
        frame_idx: Index of the frame to extract.

    Returns:
        The extracted frame as a NumPy array (BGR format).

    Raises:
        ValueError: If the frame cannot be extracted.
    """
    if video_path.endswith(".png"):
        return cv2.imread(video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not extract frame {frame_idx} from {video_path}")
    return frame


def evenly_spaced_indices(n_points: int, n_samples: int) -> torch.Tensor:
    """
    Generate indices to pick n_samples evenly spaced points from a total of n_points.

    Args:
        n_points: Total number of available points.
        n_samples: Number of indices to select.

    Returns:
        A torch.Tensor of selected indices.
    """
    return torch.linspace(0, n_points - 1, steps=n_samples).long()


def select_evenly_spaced(batch: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Select a subset of keypoints that are evenly spaced along the Y axis.

    Args:
        batch: A tensor of shape (N, 2) containing (X, Y) coordinates of keypoints.
        n_samples: Number of keypoints to select.

    Returns:
        A tensor of indices corresponding to the selected keypoints.
    """
    # Sort the keypoints based on the Y coordinate.
    _, sorted_indices = torch.sort(batch[:, 1])
    # Compute evenly spaced indices over the sorted keypoints.
    sample_indices: torch.Tensor = evenly_spaced_indices(batch.size(0), n_samples)
    # Map back to original indices.
    selected_indices: torch.Tensor = sorted_indices[sample_indices]
    return selected_indices


def calculate_control_points(
    frame0: np.ndarray,
    frame1: np.ndarray,
    max_control_points: int,
    device: Optional[torch.device] = None,
    max_num_keypoints: int = 2048,
    output_directory: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute control points (matched keypoints) between two frames using SuperPoint and LightGlue.

    Args:
        frame0: First input frame (BGR NumPy array).
        frame1: Second input frame (BGR NumPy array).
        max_control_points: Maximum number of control point matches to return.
        device: Torch device to perform computation on (defaults to CUDA if available).
        max_num_keypoints: Maximum number of keypoints to extract.
        output_directory: Directory where visualizations (matches and keypoints) will be saved.
                          If None, no visual output is generated.

    Returns:
        A dictionary with keys "m_kpts0" and "m_kpts1" containing the matched keypoints as torch.Tensors.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the SuperPoint extractor.
    extractor: SuperPoint = (
        SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
    )
    # Initialize the LightGlue matcher.
    matcher: LightGlue = (
        LightGlue(
            features="superpoint",
            depth_confidence=-1,
            width_confidence=-1,
            filter_threshold=0.2,
        )
        .eval()
        .to(device)
    )

    # Convert frames from BGR to RGB.
    img1_rgb: np.ndarray = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    img2_rgb: np.ndarray = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    # Convert images to torch tensors with shape [C, H, W] and normalize to [0, 1].
    image0: torch.Tensor = (
        kornia.image_to_tensor(img1_rgb, keepdim=False).float() / 255.0
    )
    image1: torch.Tensor = (
        kornia.image_to_tensor(img2_rgb, keepdim=False).float() / 255.0
    )

    # Move tensors to the specified device if necessary.
    if image0.device != device:
        image0 = image0.to(device)
    if image1.device != device:
        image1 = image1.to(device)

    # Extract features using SuperPoint.
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    # Run LightGlue to match features.
    matches01 = matcher({"image0": feats0, "image1": feats1})
    # Remove batch dimensions.
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    # Retrieve keypoints and matching indices.
    kpts0: torch.Tensor = feats0["keypoints"]
    kpts1: torch.Tensor = feats1["keypoints"]
    matches: torch.Tensor = matches01["matches"]

    # Select the matched keypoints.
    m_kpts0: torch.Tensor = kpts0[matches[..., 0]]
    m_kpts1: torch.Tensor = kpts1[matches[..., 1]]

    # If there are more matches than desired, select evenly spaced ones.
    indices: torch.Tensor = select_evenly_spaced(m_kpts0, max_control_points)
    m_kpts0 = m_kpts0[indices]
    m_kpts1 = m_kpts1[indices]

    # Optionally generate and save visualizations.
    if output_directory:
        axes = viz2d.plot_images([frame0, frame1])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
        viz2d.save_plot(os.path.join(output_directory, "matches.png"))

        kpc0 = viz2d.cm_prune(matches01["prune0"])
        kpc1 = viz2d.cm_prune(matches01["prune1"])
        viz2d.plot_images([frame0, frame1])
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
        viz2d.save_plot(os.path.join(output_directory, "keypoints.png"))

    control_points: Dict[str, torch.Tensor] = {"m_kpts0": m_kpts0, "m_kpts1": m_kpts1}
    return control_points


def load_pto_file(file_path: str) -> List[str]:
    """
    Load the contents of a Hugin PTO file.

    Args:
        file_path: Path to the PTO file.

    Returns:
        A list of strings representing the lines in the file (with trailing whitespace removed).
    """
    with open(file_path, "r") as file:
        lines: List[str] = file.readlines()
    # Remove trailing whitespace from each line.
    lines = [line.rstrip() for line in lines]
    return lines


def save_pto_file(file_path: str, data: List[str]) -> None:
    """
    Save a list of lines back into a PTO file.

    Args:
        file_path: Path to the PTO file.
        data: List of lines to write.
    """
    with open(file_path, "w") as file:
        for line in data:
            file.write(f"{line}\n")


def remove_control_points(lines: List[str]) -> Tuple[List[str], int]:
    """
    Remove existing control point lines (lines starting with "c ") from a PTO file content.

    Args:
        lines: List of strings representing the PTO file lines.

    Returns:
        A tuple (new_lines, count) where new_lines is the list without control point lines,
        and count is the number of control point lines removed.
    """
    prev_control_point_count: int = 0
    new_lines: List[str] = []
    for line in lines:
        if line.startswith(_CONTROL_POINTS_LINE):
            continue
        if line.startswith("c "):
            prev_control_point_count += 1
            continue
        new_lines.append(line)
    return new_lines, prev_control_point_count


def is_older_than(file1: str, file2: str) -> Optional[bool]:
    """
    Compare the modification times of two files.

    Args:
        file1: Path to the first file.
        file2: Path to the second file.

    Returns:
        True if file2 is older than file1, False if not, or None if there is an error.
    """
    try:
        mtime1 = os.path.getmtime(file1)
        mtime2 = os.path.getmtime(file2)
        return mtime2 < mtime1
    except OSError:
        return None


def strip(s: str) -> str:
    """
    Remove all whitespace from a string.

    Args:
        s: Input string.

    Returns:
        The string with all whitespace removed.
    """
    return re.sub(r"\s+", "", s)


def update_pto_file(pto_file: str, control_points: Dict[str, torch.Tensor]) -> None:
    """
    Update a Hugin PTO file by replacing existing control points with new ones.

    Args:
        pto_file: Path to the PTO file.
        control_points: Dictionary containing matched keypoints with keys "m_kpts0" and "m_kpts1".
    """
    pts0: torch.Tensor = control_points["m_kpts0"]
    pts1: torch.Tensor = control_points["m_kpts1"]
    assert len(pts0) == len(
        pts1
    ), "The number of control points in both images must match."
    print(f"Found {len(pts0)} control points")
    assert len(pts0) > 0 and len(pts1) > 0, "No control points found."

    # Load the current PTO file and remove old control point lines.
    pto_lines: List[str] = load_pto_file(pto_file)
    pto_lines, _ = remove_control_points(pto_lines)
    pto_lines.append("")
    pto_lines.append(_CONTROL_POINTS_LINE)

    def _to_hugin_decimal(val: Union[str, float]) -> str:
        # Convert value to float and then format.
        val = float(val)
        if val == float(int(val)):
            return f"{int(val)}"
        return f"{val:.12f}"

    # Append new control point lines.
    for i in range(len(pts0)):
        point0 = [float(c) for c in pts0[i]]
        point1 = [float(c) for c in pts1[i]]
        line = (
            f"c n0 N1 x{_to_hugin_decimal(point0[0])} "
            f"y{_to_hugin_decimal(point0[1])} "
            f"X{_to_hugin_decimal(point1[0])} "
            f"Y{_to_hugin_decimal(point1[1])} t0"
        )
        pto_lines.append(line)
    save_pto_file(pto_file, pto_lines)
    print("Done updating control points in the PTO file.")


def configure_stitching(
    frame1: np.ndarray,
    frame2: np.ndarray,
    directory: str,
    force: bool = True,
    skip_if_exists: bool = False,
    fov: float = 108,  # Default FOV (e.g., GoPro Wide)
    max_control_points: int = 240,
    scale: float = None,
    device: Optional[torch.device] = None,
) -> bool:
    """
    Configure and run the stitching pipeline. This includes:
      - Saving input frames as images.
      - Generating a Hugin PTO project file.
      - Computing control points and updating the PTO file.
      - Running auto-optimisation and generating mapping and panorama images.

    Args:
        frame1: First input frame (BGR image as a NumPy array).
        frame2: Second input frame (BGR image as a NumPy array).
        directory: Directory where output files will be saved.
        force: If True, force re-creation of the project file.
        skip_if_exists: If True, skip creation if output files already exist and are up-to-date.
        fov: Field-of-view parameter for the project generation.
        max_control_points: Maximum number of control points to compute.
        device: Torch device for computations.

    Returns:
        True if the process completes successfully.
    """
    # Define file names for saved images.
    left_image_file: str = "left.png"
    right_image_file: str = "right.png"
    f1: str = os.path.join(directory, left_image_file)
    f2: str = os.path.join(directory, right_image_file)

    # Save the frames to disk.
    cv2.imwrite(f1, frame1)
    cv2.imwrite(f2, frame2)

    # Define paths for the project file and autooptimiser output.
    project_file_path: str = os.path.join(directory, "hm_project.pto")
    pto_path: Path = Path(project_file_path)
    dir_name: str = str(pto_path.parent)
    hm_project: str = project_file_path
    autooptimiser_out: str = os.path.join(dir_name, "autooptimiser_out.pto")
    assert (
        autooptimiser_out != hm_project
    ), "Output project file conflicts with input project file."

    # Optionally skip processing if outputs already exist and are up-to-date.
    if skip_if_exists and (
        os.path.exists(project_file_path)
        and os.path.exists(autooptimiser_out)
        and not is_older_than(project_file_path, autooptimiser_out)
    ):
        print(
            f"Project file already exists (skipping project creation): {autooptimiser_out}"
        )
        return True

    curr_dir: str = os.getcwd()
    os.chdir(dir_name)
    try:
        # Generate the initial PTO project file if it doesn't exist or if forced.
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

        # Calculate control points using the provided frames.
        control_points: Dict[str, torch.Tensor] = calculate_control_points(
            frame1,
            frame2,
            max_control_points=max_control_points,
            device=device,
            max_num_keypoints=2048,
            output_directory=directory,
        )
        # Update the PTO file with the new control points.
        update_pto_file(project_file_path, control_points)

        # Run autooptimiser (e.g., using RANSAC) on the project file.
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
        if scale and scale != 1.0:
            cmd += [
                "-x",
                str(scale),
            ]
        os.system(" ".join(cmd))

        # Generate mapping files using nona.
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

        # Blend the mappings into a panorama using enblend.
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


def main() -> None:
    """
    Main entry point:
      - Parses command-line arguments.
      - Synchronizes the two videos by audio.
      - Extracts frames at the synchronization points.
      - Computes control points and runs the stitching pipeline.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize two videos using audio cross-correlation, extract sync frames, "
            "compute control points using LightGlue and SuperPoint, and update a Hugin PTO file."
        )
    )
    parser.add_argument(
        "--game-id",
        default=None,
        help="Game ID (everything being in $HOME/Videos/game-id)",
    )
    parser.add_argument("--left", default=None, help="Path to left video file")
    parser.add_argument("--right", default=None, help="Path to left video file")
    parser.add_argument("--max-control-points", type=int, default=500, help="Maximum number of control points")
    parser.add_argument("--lfo", default=None, help="Left frame offset")
    parser.add_argument("--rfo", default=None, help="Right frame offset")
    parser.add_argument(
        "--synchronize-only",
        action="store_true",
        help="Only synchronize and print out the frame offsets",
    )
    parser.add_argument(
        "--scale",
        default=None,
        help="Scale of the final panorama (i.e. for downsizing)",
    )
    args = parser.parse_args()

    if (not args.left or not args.right) and not args.game_id:
        print("You must supply either left and right videos or a game-id")
        exit(1)

    if (not args.left or not args.right) and args.game_id:
        game_dir: str = os.path.join(os.environ["HOME"], "Videos", args.game_id)
        config_file: str = os.path.join(game_dir, "config.yaml")
        if not os.path.exists(config_file):
            print(f"Could not find config file: {config_file}")
            exit(1)
        with open(config_file, "r") as file:
            config_yaml = yaml.safe_load(file)
        args.left = config_yaml["game"]["videos"]["left"][0]
        if "/" not in args.left:
            args.left = os.path.join(game_dir, args.left)
        args.right = config_yaml["game"]["videos"]["right"][0]
        if "/" not in args.right:
            args.right = os.path.join(game_dir, args.right)

    is_image = False
    if args.left.endswith(".png") and args.right.endswith(".png"):
        is_image = True

    if not is_image:
        # Determine frame offsets by synchronizing audio.
        if (args.lfo is None and args.rfo is None) or args.synchronize_only:
            lfo, rfo = synchronize_by_audio(args.left, args.right)
        else:
            lfo, rfo = args.lfo, args.rfo

        if args.synchronize_only:
            print(f"Left frame offset: {lfo}")
            print(f"Right frame offset: {rfo}")
            exit(0)

        print("Extracting frames at the sync points...")
    else:
        lfo, rfo = None, None

    # Ensure frame indices are integers.
    frame1: np.ndarray = extract_frame(args.left, lfo)
    frame2: np.ndarray = extract_frame(args.right, rfo)

    # Run the stitching pipeline which includes control point computation and PTO update.
    print("Running SuperPoint and LightGlue to obtain control point matches...")
    configure_stitching(
        frame1,
        frame2,
        directory=str(Path(args.left).parent),
        max_control_points=args.max_control_points,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
