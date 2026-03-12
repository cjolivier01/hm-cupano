#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile

MASK_ONE = 0xFFFF
UNMAPPED = 0xFFFF
Q_MIN = -(1 << 17)
Q_MAX = (1 << 17) - 1


@dataclass(frozen=True)
class CaseShape:
    input_w: int
    input_h: int
    overlap: int
    pad: int

    @property
    def x2(self) -> int:
        return self.input_w - self.overlap

    @property
    def canvas_w(self) -> int:
        return self.input_w + self.x2

    @property
    def canvas_h(self) -> int:
        return self.input_h

    @property
    def blend_w(self) -> int:
        return self.overlap + (2 * self.pad)

    @property
    def blend_h(self) -> int:
        return self.input_h

    @property
    def low_w(self) -> int:
        return self.blend_w // 2

    @property
    def low_h(self) -> int:
        return self.blend_h // 2

    def validate(self) -> None:
        if self.input_w <= 0 or self.input_h <= 0:
            raise ValueError("input width and height must be positive")
        if self.pad < 0:
            raise ValueError("pad must be non-negative")
        if self.overlap <= 0 or self.overlap >= self.input_w:
            raise ValueError(f"overlap={self.overlap} must satisfy 0 < overlap < input_w={self.input_w}")
        if self.input_h % 2 != 0:
            raise ValueError(f"input_h={self.input_h} must be even for the 2x2 downsample path")
        if self.blend_w % 2 != 0:
            raise ValueError(
                f"blend_w={self.blend_w} must be even; choose overlap/pad so overlap + 2*pad is even"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_w": self.input_w,
            "input_h": self.input_h,
            "overlap": self.overlap,
            "pad": self.pad,
            "x2": self.x2,
            "canvas_w": self.canvas_w,
            "canvas_h": self.canvas_h,
            "blend_w": self.blend_w,
            "blend_h": self.blend_h,
            "low_w": self.low_w,
            "low_h": self.low_h,
        }


@dataclass(frozen=True)
class ControlData:
    left_img: np.ndarray
    right_img: np.ndarray
    map1_x: np.ndarray
    map1_y: np.ndarray
    map2_x: np.ndarray
    map2_y: np.ndarray
    seam_mask: np.ndarray
    pos0: tuple[int, int]
    pos1: tuple[int, int]


@dataclass(frozen=True)
class WindowRange:
    start: int
    stop: int

    def clamp(self, value: int) -> int:
        return max(self.start, min(self.stop, value))


def default_overlap(width: int) -> int:
    overlap = max(4, min(width // 4, 64))
    if overlap >= width:
        overlap = width - 1
    if overlap % 2 != 0:
        overlap -= 1
    return max(2, overlap)


def center_crop(image: np.ndarray, width: int, height: int) -> np.ndarray:
    y0 = max((image.shape[0] - height) // 2, 0)
    x0 = max((image.shape[1] - width) // 2, 0)
    return image[y0 : y0 + height, x0 : x0 + width].copy()


def pack_rgba_word(bgr: np.ndarray) -> int:
    b, g, r = [int(v) for v in bgr]
    return (255 << 24) | (r << 16) | (g << 8) | b


def unpack_rgba_word(word: int) -> tuple[int, int, int, int]:
    b = word & 0xFF
    g = (word >> 8) & 0xFF
    r = (word >> 16) & 0xFF
    a = (word >> 24) & 0xFF
    return b, g, r, a


def word_to_qpixel(word: int) -> list[int]:
    b, g, r, a = unpack_rgba_word(word)
    return [b << 8, g << 8, r << 8, a << 8]


def clamp_q(value: int) -> int:
    return max(Q_MIN, min(Q_MAX, int(value)))


def q_to_u8(value: int) -> int:
    shifted = int(value) >> 8
    if shifted < 0:
        return 0
    if shifted > 255:
        return 255
    return shifted


def qpixel_to_word(qpixel: list[int]) -> int:
    b = q_to_u8(qpixel[0])
    g = q_to_u8(qpixel[1])
    r = q_to_u8(qpixel[2])
    a = q_to_u8(qpixel[3])
    return (a << 24) | (r << 16) | (g << 8) | b


def identity_map_x(width: int, height: int) -> np.ndarray:
    return np.tile(np.arange(width, dtype=np.uint16), (height, 1))


def identity_map_y(width: int, height: int) -> np.ndarray:
    return np.tile(np.arange(height, dtype=np.uint16)[:, None], (1, width))


def write_position_tiff(path: Path, width: int, height: int, xpos: int, ypos: int) -> None:
    tifffile.imwrite(
        path,
        np.zeros((height, width), dtype=np.uint8),
        resolution=(1.0, 1.0),
        extratags=[
            (286, 5, 1, (xpos, 1), False),
            (287, 5, 1, (ypos, 1), False),
        ],
    )


def read_tiff_position(path: Path) -> tuple[int, int]:
    with tifffile.TiffFile(str(path)) as tif:
        tags = tif.pages[0].tags

        def to_float(value: object) -> float:
            if isinstance(value, tuple) and len(value) == 2:
                num, den = value
                return float(num) / float(den or 1)
            if isinstance(value, (list, tuple)) and value:
                return to_float(value[0])
            return float(value)

        xres = to_float(tags["XResolution"].value) if "XResolution" in tags else 0.0
        yres = to_float(tags["YResolution"].value) if "YResolution" in tags else 0.0
        xpos = to_float(tags["XPosition"].value) if "XPosition" in tags else 0.0
        ypos = to_float(tags["YPosition"].value) if "YPosition" in tags else 0.0
    return int(round(xpos * xres)), int(round(ypos * yres))


def normalize_positions(pos0: tuple[int, int], pos1: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    min_x = min(pos0[0], pos1[0])
    min_y = min(pos0[1], pos1[1])
    return (pos0[0] - min_x, pos0[1] - min_y), (pos1[0] - min_x, pos1[1] - min_y)


def normalize_binary_seam(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    min_val = int(mask.min())
    max_val = int(mask.max())
    out = np.zeros_like(mask, dtype=np.uint8)
    out[mask == min_val] = 1
    out[mask == max_val] = 0
    return out


def pad_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape[0] > height or mask.shape[1] > width:
        raise ValueError(f"Seam mask is larger than target canvas: {mask.shape} vs {(height, width)}")
    pad_bottom = height - mask.shape[0]
    pad_right = width - mask.shape[1]
    if pad_bottom == 0 and pad_right == 0:
        return mask
    return cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)


def downsample_pixel(image: np.ndarray, x: int, y: int) -> list[int]:
    pixels = [
        image[(2 * y) + 0, (2 * x) + 0],
        image[(2 * y) + 0, (2 * x) + 1],
        image[(2 * y) + 1, (2 * x) + 0],
        image[(2 * y) + 1, (2 * x) + 1],
    ]
    out = [0, 0, 0, 0]
    for channel in range(3):
        values = [pix[channel] for pix in pixels if pix[3] != 0]
        out[channel] = 0 if not values else sum(values) // len(values)
    out[3] = max(pix[3] for pix in pixels)
    return out


def downsample_mask(mask: np.ndarray, x: int, y: int) -> int:
    return int(
        (
            int(mask[(2 * y) + 0, (2 * x) + 0])
            + int(mask[(2 * y) + 0, (2 * x) + 1])
            + int(mask[(2 * y) + 1, (2 * x) + 0])
            + int(mask[(2 * y) + 1, (2 * x) + 1])
        )
        >> 2
    )


def upsample_weights(x_odd: int, y_odd: int) -> tuple[int, int, int, int]:
    if not x_odd and not y_odd:
        return 4, 0, 0, 0
    if x_odd and not y_odd:
        return 2, 2, 0, 0
    if not x_odd and y_odd:
        return 2, 0, 2, 0
    return 1, 1, 1, 1


def laplacian_pixel(high: list[int], low00: list[int], low10: list[int], low01: list[int], low11: list[int], x: int, y: int) -> list[int]:
    w00, w10, w01, w11 = upsample_weights(x & 1, y & 1)
    out = [0, 0, 0, high[3]]
    low_pixels = [(low00, w00), (low10, w10), (low01, w01), (low11, w11)]
    for channel in range(3):
        weighted_sum = 0
        weight_sum = 0
        for pixel, weight in low_pixels:
            if pixel[3] != 0:
                weighted_sum += pixel[channel] * weight
                weight_sum += weight
        upsampled = 0 if weight_sum == 0 else weighted_sum // weight_sum
        out[channel] = clamp_q(high[channel] - upsampled)
    return out


def blend_pixel(lap1: list[int], lap2: list[int], mask_weight: int) -> list[int]:
    out = [0, 0, 0, lap1[3] if lap1[3] != 0 else lap2[3]]
    for channel in range(3):
        if lap1[3] == 0:
            out[channel] = lap2[channel]
        elif lap2[3] == 0:
            out[channel] = lap1[channel]
        else:
            blend_value = (lap1[channel] * mask_weight) + (lap2[channel] * (MASK_ONE - mask_weight))
            out[channel] = clamp_q(blend_value // MASK_ONE)
    return out


def reconstruct_pixel(lower00: list[int], lower10: list[int], lower01: list[int], lower11: list[int], lap: list[int], x: int, y: int) -> list[int]:
    w00, w10, w01, w11 = upsample_weights(x & 1, y & 1)
    out = [0, 0, 0, lap[3]]
    low_pixels = [(lower00, w00), (lower10, w10), (lower01, w01), (lower11, w11)]
    for channel in range(3):
        weighted_sum = 0
        weight_sum = 0
        for pixel, weight in low_pixels:
            if pixel[3] != 0:
                weighted_sum += pixel[channel] * weight
                weight_sum += weight
        upsampled = 0 if weight_sum == 0 else weighted_sum // weight_sum
        out[channel] = clamp_q(upsampled + lap[channel])
    return out


def low_neighbor(image: np.ndarray, x: int, y: int) -> tuple[list[int], list[int], list[int], list[int]]:
    lx = x >> 1
    ly = y >> 1
    lx1 = min(lx + 1, image.shape[1] - 1)
    ly1 = min(ly + 1, image.shape[0] - 1)
    return image[ly, lx].tolist(), image[ly, lx1].tolist(), image[ly1, lx].tolist(), image[ly1, lx1].tolist()


def write_words32(path: Path, words: list[int]) -> None:
    path.write_text("\n".join(f"{word:08x}" for word in words) + "\n", encoding="ascii")


def write_words16(path: Path, words: list[int]) -> None:
    path.write_text("\n".join(f"{word:04x}" for word in words) + "\n", encoding="ascii")


def read_words(path: Path) -> list[str]:
    words: list[str] = []
    for line in path.read_text(encoding="ascii").splitlines():
        word = line.strip().lower()
        if not word or word.startswith("//") or word.startswith("@"):
            continue
        words.append(word)
    return words


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def load_u16_tiff(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise FileNotFoundError(path)
    if image.dtype != np.uint16:
        raise ValueError(f"Expected CV_16U TIFF at {path}, got {image.dtype}")
    return image


def load_control_data(control_dir: Path, left_path: Path | None = None, right_path: Path | None = None) -> ControlData:
    left_candidate = control_dir / "left.png"
    right_candidate = control_dir / "right.png"
    if left_candidate.exists():
        left_img = load_image(left_candidate)
    elif left_path is not None:
        left_img = load_image(left_path)
    else:
        raise FileNotFoundError(left_candidate)
    if right_candidate.exists():
        right_img = load_image(right_candidate)
    elif right_path is not None:
        right_img = load_image(right_path)
    else:
        raise FileNotFoundError(right_candidate)

    map1_x = load_u16_tiff(control_dir / "mapping_0000_x.tif")
    map1_y = load_u16_tiff(control_dir / "mapping_0000_y.tif")
    map2_x = load_u16_tiff(control_dir / "mapping_0001_x.tif")
    map2_y = load_u16_tiff(control_dir / "mapping_0001_y.tif")
    if map1_x.shape != map1_y.shape or map2_x.shape != map2_y.shape:
        raise ValueError("map x/y TIFF shapes do not match")

    pos0 = read_tiff_position(control_dir / "mapping_0000.tif")
    pos1 = read_tiff_position(control_dir / "mapping_0001.tif")
    norm0, norm1 = normalize_positions(pos0, pos1)
    full_canvas_w = max(norm0[0] + map1_x.shape[1], norm1[0] + map2_x.shape[1])
    full_canvas_h = max(norm0[1] + map1_x.shape[0], norm1[1] + map2_x.shape[0])

    seam_raw = cv2.imread(str(control_dir / "seam_file.png"), cv2.IMREAD_GRAYSCALE)
    if seam_raw is None:
        raise FileNotFoundError(control_dir / "seam_file.png")
    seam_mask = pad_mask(normalize_binary_seam(seam_raw), full_canvas_h, full_canvas_w)

    return ControlData(
        left_img=left_img,
        right_img=right_img,
        map1_x=map1_x,
        map1_y=map1_y,
        map2_x=map2_x,
        map2_y=map2_y,
        seam_mask=seam_mask,
        pos0=norm0,
        pos1=norm1,
    )


def infer_shape_from_control(control_dir: Path, pad: int) -> CaseShape:
    data = load_control_data(control_dir)
    width = int(data.map1_x.shape[1])
    height = int(data.map1_x.shape[0])
    if data.pos0 != (0, 0) or data.pos1[1] != 0:
        raise ValueError(f"Control dir {control_dir} is not a normalized two-image horizontal layout: {data.pos0}, {data.pos1}")
    overlap = width - int(data.pos1[0])
    shape = CaseShape(width, height, overlap, pad)
    shape.validate()
    return shape


def write_case_manifest(outdir: Path, shape: CaseShape) -> None:
    (outdir / "case_manifest.json").write_text(json.dumps(shape.to_dict(), indent=2) + "\n", encoding="ascii")


def load_case_manifest(outdir: Path) -> CaseShape:
    raw = json.loads((outdir / "case_manifest.json").read_text(encoding="ascii"))
    shape = CaseShape(raw["input_w"], raw["input_h"], raw["overlap"], raw["pad"])
    shape.validate()
    return shape


def write_control_dir(
    control_dir: Path,
    left_img: np.ndarray,
    right_img: np.ndarray,
    map1_x: np.ndarray,
    map1_y: np.ndarray,
    map2_x: np.ndarray,
    map2_y: np.ndarray,
    seam_mask: np.ndarray,
    shape: CaseShape,
) -> None:
    control_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(control_dir / "left.png"), left_img):
        raise RuntimeError(f"Failed to write {control_dir / 'left.png'}")
    if not cv2.imwrite(str(control_dir / "right.png"), right_img):
        raise RuntimeError(f"Failed to write {control_dir / 'right.png'}")
    if not cv2.imwrite(str(control_dir / "mapping_0000_x.tif"), map1_x):
        raise RuntimeError(f"Failed to write {control_dir / 'mapping_0000_x.tif'}")
    if not cv2.imwrite(str(control_dir / "mapping_0000_y.tif"), map1_y):
        raise RuntimeError(f"Failed to write {control_dir / 'mapping_0000_y.tif'}")
    if not cv2.imwrite(str(control_dir / "mapping_0001_x.tif"), map2_x):
        raise RuntimeError(f"Failed to write {control_dir / 'mapping_0001_x.tif'}")
    if not cv2.imwrite(str(control_dir / "mapping_0001_y.tif"), map2_y):
        raise RuntimeError(f"Failed to write {control_dir / 'mapping_0001_y.tif'}")

    write_position_tiff(control_dir / "mapping_0000.tif", shape.input_w, shape.input_h, 0, 0)
    write_position_tiff(control_dir / "mapping_0001.tif", shape.input_w, shape.input_h, shape.x2, 0)

    seam_file = np.where(seam_mask != 0, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(control_dir / "seam_file.png"), seam_file):
        raise RuntimeError(f"Failed to write {control_dir / 'seam_file.png'}")

    (control_dir / "case_shape.json").write_text(json.dumps(shape.to_dict(), indent=2) + "\n", encoding="ascii")


def write_generated_control_dir(left_path: Path, right_path: Path, control_dir: Path, shape: CaseShape) -> None:
    left = load_image(left_path)
    right = load_image(right_path)
    left_crop = center_crop(left, shape.input_w, shape.input_h)
    right_crop = center_crop(right, shape.input_w, shape.input_h)

    seam_mask = np.zeros((shape.canvas_h, shape.canvas_w), dtype=np.uint8)
    seam_mask[:, : shape.x2 + (shape.overlap // 2)] = 1

    write_control_dir(
        control_dir,
        left_crop,
        right_crop,
        identity_map_x(shape.input_w, shape.input_h),
        identity_map_y(shape.input_w, shape.input_h),
        identity_map_x(shape.input_w, shape.input_h),
        identity_map_y(shape.input_w, shape.input_h),
        seam_mask,
        shape,
    )


def stage_control_dir(source_control_dir: Path, staged_control_dir: Path) -> None:
    staged_control_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "mapping_0000.tif",
        "mapping_0000_x.tif",
        "mapping_0000_y.tif",
        "mapping_0001.tif",
        "mapping_0001_x.tif",
        "mapping_0001_y.tif",
        "seam_file.png",
    ]:
        shutil.copy2(source_control_dir / name, staged_control_dir / name)
    for name in ["left.png", "right.png", "case_shape.json"]:
        candidate = source_control_dir / name
        if candidate.exists():
            shutil.copy2(candidate, staged_control_dir / name)


def seam_transition_columns(mask: np.ndarray, row: int) -> np.ndarray:
    row_values = mask[row].astype(np.int16)
    return np.where(row_values[:-1] != row_values[1:])[0] + 1


def rebase_source_and_maps(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = (map_x != UNMAPPED) & (map_y != UNMAPPED)
    if not np.any(valid):
        raise ValueError("Extracted map window has no valid source coordinates")
    min_x = int(map_x[valid].min())
    max_x = int(map_x[valid].max())
    min_y = int(map_y[valid].min())
    max_y = int(map_y[valid].max())
    if min_x < 0 or min_y < 0 or max_x >= image.shape[1] or max_y >= image.shape[0]:
        raise ValueError("Map window references pixels outside the source image bounds")
    rebased_x = np.where(valid, map_x - min_x, UNMAPPED).astype(np.uint16)
    rebased_y = np.where(valid, map_y - min_y, UNMAPPED).astype(np.uint16)
    return image[min_y : max_y + 1, min_x : max_x + 1].copy(), rebased_x, rebased_y


def extract_control_window(
    left_path: Path,
    right_path: Path,
    source_control_dir: Path,
    outdir: Path,
    shape: CaseShape,
) -> None:
    shape.validate()
    data = load_control_data(source_control_dir, left_path, right_path)
    map_h0, map_w0 = data.map1_x.shape
    map_h1, map_w1 = data.map2_x.shape

    x_range = WindowRange(
        max(data.pos0[0], data.pos1[0] - shape.x2),
        min(data.pos0[0] + map_w0 - shape.input_w, data.pos1[0] + map_w1 - shape.input_w - shape.x2),
    )
    y_range = WindowRange(
        max(data.pos0[1], data.pos1[1]),
        min(data.pos0[1] + map_h0 - shape.input_h, data.pos1[1] + map_h1 - shape.input_h),
    )
    if x_range.start > x_range.stop or y_range.start > y_range.stop:
        raise ValueError("Requested window does not fit inside the source control geometry")

    seam_rows = [row for row in range(y_range.start, y_range.stop + 1) if seam_transition_columns(data.seam_mask, row).size > 0]
    if not seam_rows:
        raise ValueError("Unable to find a row where the seam crosses inside the feasible window range")
    seam_mid_row = seam_rows[len(seam_rows) // 2]
    preferred_y0 = seam_mid_row - (shape.input_h // 2)
    window_y0 = y_range.clamp(preferred_y0)

    seam_cols = seam_transition_columns(data.seam_mask, seam_mid_row)
    seam_x = int(seam_cols[len(seam_cols) // 2])
    preferred_x0 = seam_x - (shape.x2 + (shape.overlap // 2))
    window_x0 = x_range.clamp(preferred_x0)

    left_crop_x = window_x0 - data.pos0[0]
    left_crop_y = window_y0 - data.pos0[1]
    right_crop_x = window_x0 + shape.x2 - data.pos1[0]
    right_crop_y = window_y0 - data.pos1[1]

    map1_x = data.map1_x[left_crop_y : left_crop_y + shape.input_h, left_crop_x : left_crop_x + shape.input_w].copy()
    map1_y = data.map1_y[left_crop_y : left_crop_y + shape.input_h, left_crop_x : left_crop_x + shape.input_w].copy()
    map2_x = data.map2_x[right_crop_y : right_crop_y + shape.input_h, right_crop_x : right_crop_x + shape.input_w].copy()
    map2_y = data.map2_y[right_crop_y : right_crop_y + shape.input_h, right_crop_x : right_crop_x + shape.input_w].copy()

    left_img, map1_x, map1_y = rebase_source_and_maps(data.left_img, map1_x, map1_y)
    right_img, map2_x, map2_y = rebase_source_and_maps(data.right_img, map2_x, map2_y)
    seam_crop = data.seam_mask[window_y0 : window_y0 + shape.canvas_h, window_x0 : window_x0 + shape.canvas_w].copy()

    write_control_dir(outdir, left_img, right_img, map1_x, map1_y, map2_x, map2_y, seam_crop, shape)


def load_matching_control_data(control_dir: Path, left_path: Path, right_path: Path, shape: CaseShape) -> ControlData:
    data = load_control_data(control_dir, left_path, right_path)
    if data.map1_x.shape != (shape.input_h, shape.input_w):
        raise ValueError(f"Expected map1 shape {(shape.input_h, shape.input_w)}, got {data.map1_x.shape}")
    if data.map2_x.shape != (shape.input_h, shape.input_w):
        raise ValueError(f"Expected map2 shape {(shape.input_h, shape.input_w)}, got {data.map2_x.shape}")
    if data.pos0 != (0, 0) or data.pos1 != (shape.x2, 0):
        raise ValueError(f"Expected normalized positions {(0, 0)} and {(shape.x2, 0)}, got {data.pos0} and {data.pos1}")
    if data.seam_mask.shape != (shape.canvas_h, shape.canvas_w):
        raise ValueError(f"Expected seam shape {(shape.canvas_h, shape.canvas_w)}, got {data.seam_mask.shape}")
    return data


def apply_remap(canvas: np.ndarray, source: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, offset_x: int, offset_y: int, shape: CaseShape) -> None:
    src_h, src_w = source.shape
    for y in range(shape.input_h):
        for x in range(shape.input_w):
            sx = int(map_x[y, x])
            sy = int(map_y[y, x])
            if sx == UNMAPPED or sy == UNMAPPED:
                pixel = 0
            elif sx < 0 or sy < 0 or sx >= src_w or sy >= src_h:
                pixel = 0
            else:
                pixel = int(source[sy, sx])
            canvas[offset_y + y, offset_x + x] = pixel


def build_case(left_path: Path, right_path: Path, outdir: Path, shape: CaseShape, control_dir: Path | None = None) -> None:
    shape.validate()
    outdir.mkdir(parents=True, exist_ok=True)
    staged_control_dir = outdir / "control"
    if control_dir is None:
        write_generated_control_dir(left_path, right_path, staged_control_dir, shape)
    else:
        control_dir = control_dir.resolve()
        try:
            control_shape = infer_shape_from_control(control_dir, shape.pad)
        except Exception:
            control_shape = None
        if control_shape is not None and control_shape.input_w == shape.input_w and control_shape.input_h == shape.input_h and control_shape.overlap == shape.overlap:
            stage_control_dir(control_dir, staged_control_dir)
        else:
            extract_control_window(left_path, right_path, control_dir, staged_control_dir, shape)

    data = load_matching_control_data(staged_control_dir, left_path, right_path, shape)

    left_words = np.array(
        [[pack_rgba_word(data.left_img[y, x]) for x in range(shape.input_w)] for y in range(shape.input_h)],
        dtype=np.uint32,
    )
    right_words = np.array(
        [[pack_rgba_word(data.right_img[y, x]) for x in range(shape.input_w)] for y in range(shape.input_h)],
        dtype=np.uint32,
    )

    canvas = np.zeros((shape.canvas_h, shape.canvas_w), dtype=np.uint32)
    apply_remap(canvas, left_words, data.map1_x, data.map1_y, 0, 0, shape)

    blend_left = np.zeros((shape.blend_h, shape.blend_w), dtype=np.uint32)
    blend_right = np.zeros((shape.blend_h, shape.blend_w), dtype=np.uint32)
    blend_left[:, 0 : shape.overlap + shape.pad] = canvas[:, shape.x2 - shape.pad : shape.x2 - shape.pad + shape.overlap + shape.pad]

    apply_remap(canvas, right_words, data.map2_x, data.map2_y, shape.x2, 0, shape)
    blend_right[:, shape.pad : shape.pad + shape.overlap + shape.pad] = canvas[:, shape.x2 : shape.x2 + shape.overlap + shape.pad]

    mask_high_bin = data.seam_mask[:, shape.x2 - shape.pad : shape.x2 - shape.pad + shape.blend_w]
    mask_high = np.where(mask_high_bin != 0, MASK_ONE, 0).astype(np.uint16)

    left_q = np.array(
        [[word_to_qpixel(int(blend_left[y, x])) for x in range(shape.blend_w)] for y in range(shape.blend_h)],
        dtype=np.int32,
    )
    right_q = np.array(
        [[word_to_qpixel(int(blend_right[y, x])) for x in range(shape.blend_w)] for y in range(shape.blend_h)],
        dtype=np.int32,
    )

    low_left = np.zeros((shape.low_h, shape.low_w, 4), dtype=np.int32)
    low_right = np.zeros((shape.low_h, shape.low_w, 4), dtype=np.int32)
    low_mask = np.zeros((shape.low_h, shape.low_w), dtype=np.uint16)
    for y in range(shape.low_h):
        for x in range(shape.low_w):
            low_left[y, x] = downsample_pixel(left_q, x, y)
            low_right[y, x] = downsample_pixel(right_q, x, y)
            low_mask[y, x] = downsample_mask(mask_high, x, y)

    high_blend = np.zeros((shape.blend_h, shape.blend_w, 4), dtype=np.int32)
    for y in range(shape.blend_h):
        for x in range(shape.blend_w):
            low00, low10, low01, low11 = low_neighbor(low_left, x, y)
            lap1 = laplacian_pixel(left_q[y, x].tolist(), low00, low10, low01, low11, x, y)
            low00, low10, low01, low11 = low_neighbor(low_right, x, y)
            lap2 = laplacian_pixel(right_q[y, x].tolist(), low00, low10, low01, low11, x, y)
            high_blend[y, x] = blend_pixel(lap1, lap2, int(mask_high[y, x]))

    low_blend = np.zeros((shape.low_h, shape.low_w, 4), dtype=np.int32)
    for y in range(shape.low_h):
        for x in range(shape.low_w):
            low_blend[y, x] = blend_pixel(low_left[y, x].tolist(), low_right[y, x].tolist(), int(low_mask[y, x]))

    blend_out = np.zeros((shape.blend_h, shape.blend_w), dtype=np.uint32)
    for y in range(shape.blend_h):
        for x in range(shape.blend_w):
            low00, low10, low01, low11 = low_neighbor(low_blend, x, y)
            recon = reconstruct_pixel(low00, low10, low01, low11, high_blend[y, x].tolist(), x, y)
            blend_out[y, x] = qpixel_to_word(recon)

    canvas[:, shape.x2 - shape.pad : shape.x2 - shape.pad + shape.blend_w] = blend_out

    write_words32(outdir / "left.hex", left_words.reshape(-1).tolist())
    write_words32(outdir / "right.hex", right_words.reshape(-1).tolist())
    write_words16(outdir / "map1_x.hex", data.map1_x.reshape(-1).tolist())
    write_words16(outdir / "map1_y.hex", data.map1_y.reshape(-1).tolist())
    write_words16(outdir / "map2_x.hex", data.map2_x.reshape(-1).tolist())
    write_words16(outdir / "map2_y.hex", data.map2_y.reshape(-1).tolist())
    write_words16(outdir / "mask_high.hex", mask_high.reshape(-1).tolist())
    write_words32(outdir / "expected_canvas.hex", canvas.reshape(-1).tolist())
    write_case_manifest(outdir, shape)


def compare_case(outdir: Path, actual_path: Path | None = None) -> None:
    shape = load_case_manifest(outdir)
    expected = read_words(outdir / "expected_canvas.hex")
    actual = read_words(actual_path if actual_path is not None else outdir / "canvas_out.hex")
    if len(expected) != len(actual):
        raise SystemExit(f"word-count mismatch: expected {len(expected)} got {len(actual)}")
    for index, (exp, got) in enumerate(zip(expected, actual)):
        if exp.lower() != got.lower():
            y = index // shape.canvas_w
            x = index % shape.canvas_w
            raise SystemExit(f"canvas mismatch at ({x}, {y}): expected 0x{exp} got 0x{got}")


def resolve_shape(args: argparse.Namespace, control_dir: Path | None = None) -> CaseShape:
    pad = args.pad
    if args.width is None or args.height is None:
        if control_dir is None:
            width = args.width if args.width is not None else 16
            height = args.height if args.height is not None else 8
            overlap = args.overlap if args.overlap is not None else default_overlap(width)
            shape = CaseShape(width, height, overlap, pad)
            shape.validate()
            return shape
        control_shape = infer_shape_from_control(control_dir, pad)
        width = control_shape.input_w if args.width is None else args.width
        height = control_shape.input_h if args.height is None else args.height
        overlap = control_shape.overlap if args.overlap is None else args.overlap
        shape = CaseShape(width, height, overlap, pad)
        shape.validate()
        return shape

    overlap = args.overlap if args.overlap is not None else default_overlap(args.width)
    shape = CaseShape(args.width, args.height, overlap, pad)
    shape.validate()
    return shape


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate, extract, or compare two-image RTL stitch cases")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--left", required=True)
    gen.add_argument("--right", required=True)
    gen.add_argument("--outdir", required=True)
    gen.add_argument("--control-dir", default=None)
    gen.add_argument("--width", type=int, default=None)
    gen.add_argument("--height", type=int, default=None)
    gen.add_argument("--overlap", type=int, default=None)
    gen.add_argument("--pad", type=int, default=2)

    extract = sub.add_parser("extract-control")
    extract.add_argument("--left", required=True)
    extract.add_argument("--right", required=True)
    extract.add_argument("--control-dir", required=True)
    extract.add_argument("--outdir", required=True)
    extract.add_argument("--width", type=int, required=True)
    extract.add_argument("--height", type=int, required=True)
    extract.add_argument("--overlap", type=int, default=None)
    extract.add_argument("--pad", type=int, default=2)

    cmp = sub.add_parser("compare")
    cmp.add_argument("--outdir", required=True)
    cmp.add_argument("--actual", default=None)

    args = parser.parse_args()
    if args.cmd == "compare":
        compare_case(Path(args.outdir), None if args.actual is None else Path(args.actual))
        return

    control_dir = None if getattr(args, "control_dir", None) is None else Path(args.control_dir)
    shape = resolve_shape(args, control_dir)
    if args.cmd == "extract-control":
        extract_control_window(
            Path(args.left),
            Path(args.right),
            Path(args.control_dir),
            Path(args.outdir),
            shape,
        )
        return

    build_case(
        Path(args.left),
        Path(args.right),
        Path(args.outdir),
        shape,
        control_dir,
    )


if __name__ == "__main__":
    main()
