#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import tifffile

INPUT_W = 16
INPUT_H = 8
OVERLAP = 4
PAD = 2
X2 = INPUT_W - OVERLAP
CANVAS_W = INPUT_W + X2
CANVAS_H = INPUT_H
BLEND_W = OVERLAP + (2 * PAD)
BLEND_H = INPUT_H
LOW_W = BLEND_W // 2
LOW_H = BLEND_H // 2
MASK_ONE = 0xFFFF
UNMAPPED = 0xFFFF
Q_MIN = -(1 << 17)
Q_MAX = (1 << 17) - 1


def center_crop(image: np.ndarray, width: int, height: int) -> np.ndarray:
    y0 = max((image.shape[0] - height) // 2, 0)
    x0 = max((image.shape[1] - width) // 2, 0)
    return image[y0:y0 + height, x0:x0 + width].copy()


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


def normalize_binary_seam(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    min_val = int(mask.min())
    max_val = int(mask.max())
    out = np.zeros_like(mask, dtype=np.uint8)
    out[mask == min_val] = 1
    out[mask == max_val] = 0
    return out


def pad_mask_to_canvas(mask: np.ndarray) -> np.ndarray:
    if mask.shape[0] > CANVAS_H or mask.shape[1] > CANVAS_W:
        raise ValueError(f"Seam mask is larger than reduced canvas: {mask.shape} vs {(CANVAS_H, CANVAS_W)}")
    pad_bottom = CANVAS_H - mask.shape[0]
    pad_right = CANVAS_W - mask.shape[1]
    if pad_bottom == 0 and pad_right == 0:
        return mask
    return cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)


def downsample_pixel(image: np.ndarray, x: int, y: int) -> list[int]:
    pixels = [
        image[2 * y + 0, 2 * x + 0],
        image[2 * y + 0, 2 * x + 1],
        image[2 * y + 1, 2 * x + 0],
        image[2 * y + 1, 2 * x + 1],
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
            int(mask[2 * y + 0, 2 * x + 0])
            + int(mask[2 * y + 0, 2 * x + 1])
            + int(mask[2 * y + 1, 2 * x + 0])
            + int(mask[2 * y + 1, 2 * x + 1])
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


def write_reduced_control_dir(left_path: Path, right_path: Path, control_dir: Path) -> None:
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left is None:
        raise FileNotFoundError(left_path)
    if right is None:
        raise FileNotFoundError(right_path)

    left_crop = center_crop(left, INPUT_W, INPUT_H)
    right_crop = center_crop(right, INPUT_W, INPUT_H)

    control_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(control_dir / "left.png"), left_crop):
        raise RuntimeError(f"Failed to write {control_dir / 'left.png'}")
    if not cv2.imwrite(str(control_dir / "right.png"), right_crop):
        raise RuntimeError(f"Failed to write {control_dir / 'right.png'}")

    map_x = identity_map_x(INPUT_W, INPUT_H)
    map_y = identity_map_y(INPUT_W, INPUT_H)
    if not cv2.imwrite(str(control_dir / "mapping_0000_x.tif"), map_x):
        raise RuntimeError(f"Failed to write {control_dir / 'mapping_0000_x.tif'}")
    if not cv2.imwrite(str(control_dir / "mapping_0000_y.tif"), map_y):
        raise RuntimeError(f"Failed to write {control_dir / 'mapping_0000_y.tif'}")
    if not cv2.imwrite(str(control_dir / "mapping_0001_x.tif"), map_x):
        raise RuntimeError(f"Failed to write {control_dir / 'mapping_0001_x.tif'}")
    if not cv2.imwrite(str(control_dir / "mapping_0001_y.tif"), map_y):
        raise RuntimeError(f"Failed to write {control_dir / 'mapping_0001_y.tif'}")

    write_position_tiff(control_dir / "mapping_0000.tif", INPUT_W, INPUT_H, 0, 0)
    write_position_tiff(control_dir / "mapping_0001.tif", INPUT_W, INPUT_H, X2, 0)

    seam = np.full((CANVAS_H, CANVAS_W), 255, dtype=np.uint8)
    seam[:, : X2 + (OVERLAP // 2)] = 0
    if not cv2.imwrite(str(control_dir / "seam_file.png"), seam):
        raise RuntimeError(f"Failed to write {control_dir / 'seam_file.png'}")


def stage_external_control_dir(left_path: Path, right_path: Path, source_control_dir: Path, staged_control_dir: Path) -> None:
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

    left_candidate = source_control_dir / "left.png"
    right_candidate = source_control_dir / "right.png"
    if left_candidate.exists():
        shutil.copy2(left_candidate, staged_control_dir / "left.png")
    else:
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        if left is None:
            raise FileNotFoundError(left_path)
        if not cv2.imwrite(str(staged_control_dir / "left.png"), left):
            raise RuntimeError(f"Failed to write {staged_control_dir / 'left.png'}")
    if right_candidate.exists():
        shutil.copy2(right_candidate, staged_control_dir / "right.png")
    else:
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if right is None:
            raise FileNotFoundError(right_path)
        if not cv2.imwrite(str(staged_control_dir / "right.png"), right):
            raise RuntimeError(f"Failed to write {staged_control_dir / 'right.png'}")


def load_case_images(left_path: Path, right_path: Path, control_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    left_candidate = control_dir / "left.png"
    right_candidate = control_dir / "right.png"
    left_img = cv2.imread(str(left_candidate if left_candidate.exists() else left_path), cv2.IMREAD_COLOR)
    right_img = cv2.imread(str(right_candidate if right_candidate.exists() else right_path), cv2.IMREAD_COLOR)
    if left_img is None:
        raise FileNotFoundError(left_candidate if left_candidate.exists() else left_path)
    if right_img is None:
        raise FileNotFoundError(right_candidate if right_candidate.exists() else right_path)
    if left_img.shape[:2] != (INPUT_H, INPUT_W):
        raise ValueError(f"Reduced RTL case requires left image shape {(INPUT_H, INPUT_W)}, got {left_img.shape[:2]}")
    if right_img.shape[:2] != (INPUT_H, INPUT_W):
        raise ValueError(f"Reduced RTL case requires right image shape {(INPUT_H, INPUT_W)}, got {right_img.shape[:2]}")
    return left_img, right_img


def load_u16_map(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise FileNotFoundError(path)
    if image.dtype != np.uint16:
        raise ValueError(f"Expected CV_16U TIFF at {path}, got {image.dtype}")
    if image.shape != (INPUT_H, INPUT_W):
        raise ValueError(f"Expected reduced map shape {(INPUT_H, INPUT_W)} at {path}, got {image.shape}")
    return image


def load_binary_seam(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return pad_mask_to_canvas(normalize_binary_seam(image))


def load_positions(control_dir: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    pos0 = read_tiff_position(control_dir / "mapping_0000.tif")
    pos1 = read_tiff_position(control_dir / "mapping_0001.tif")
    min_x = min(pos0[0], pos1[0])
    min_y = min(pos0[1], pos1[1])
    norm0 = (pos0[0] - min_x, pos0[1] - min_y)
    norm1 = (pos1[0] - min_x, pos1[1] - min_y)
    if norm0 != (0, 0) or norm1 != (X2, 0):
        raise ValueError(
            f"Reduced RTL case expects normalized positions {(0, 0)} and {(X2, 0)}, got {norm0} and {norm1}"
        )
    return norm0, norm1


def apply_remap(canvas: np.ndarray, source: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, offset_x: int, offset_y: int) -> None:
    src_h, src_w = source.shape
    for y in range(INPUT_H):
        for x in range(INPUT_W):
            sx = int(map_x[y, x])
            sy = int(map_y[y, x])
            if sx == UNMAPPED or sy == UNMAPPED:
                pixel = 0
            elif sx < 0 or sy < 0 or sx >= src_w or sy >= src_h:
                pixel = 0
            else:
                pixel = int(source[sy, sx])
            canvas[offset_y + y, offset_x + x] = pixel


def build_case(left_path: Path, right_path: Path, outdir: Path, control_dir: Path | None = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    staged_control_dir = outdir / "control"
    if control_dir is None:
        write_reduced_control_dir(left_path, right_path, staged_control_dir)
    else:
        stage_external_control_dir(left_path, right_path, control_dir.resolve(), staged_control_dir)
    control_dir = staged_control_dir

    left_img, right_img = load_case_images(left_path, right_path, control_dir)
    map1_x = load_u16_map(control_dir / "mapping_0000_x.tif")
    map1_y = load_u16_map(control_dir / "mapping_0000_y.tif")
    map2_x = load_u16_map(control_dir / "mapping_0001_x.tif")
    map2_y = load_u16_map(control_dir / "mapping_0001_y.tif")
    seam_mask = load_binary_seam(control_dir / "seam_file.png")
    load_positions(control_dir)

    left_words = np.array([[pack_rgba_word(left_img[y, x]) for x in range(INPUT_W)] for y in range(INPUT_H)], dtype=np.uint32)
    right_words = np.array([[pack_rgba_word(right_img[y, x]) for x in range(INPUT_W)] for y in range(INPUT_H)], dtype=np.uint32)

    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint32)
    apply_remap(canvas, left_words, map1_x, map1_y, 0, 0)

    blend_left = np.zeros((BLEND_H, BLEND_W), dtype=np.uint32)
    blend_right = np.zeros((BLEND_H, BLEND_W), dtype=np.uint32)
    blend_left[:, 0 : OVERLAP + PAD] = canvas[:, X2 - PAD : X2 - PAD + OVERLAP + PAD]

    apply_remap(canvas, right_words, map2_x, map2_y, X2, 0)
    blend_right[:, PAD : PAD + OVERLAP + PAD] = canvas[:, X2 : X2 + OVERLAP + PAD]

    mask_high_bin = seam_mask[:, X2 - PAD : X2 - PAD + BLEND_W]
    mask_high = np.where(mask_high_bin != 0, MASK_ONE, 0).astype(np.uint16)

    left_q = np.array([[word_to_qpixel(int(blend_left[y, x])) for x in range(BLEND_W)] for y in range(BLEND_H)], dtype=np.int32)
    right_q = np.array([[word_to_qpixel(int(blend_right[y, x])) for x in range(BLEND_W)] for y in range(BLEND_H)], dtype=np.int32)

    low_left = np.zeros((LOW_H, LOW_W, 4), dtype=np.int32)
    low_right = np.zeros((LOW_H, LOW_W, 4), dtype=np.int32)
    low_mask = np.zeros((LOW_H, LOW_W), dtype=np.uint16)
    for y in range(LOW_H):
        for x in range(LOW_W):
            low_left[y, x] = downsample_pixel(left_q, x, y)
            low_right[y, x] = downsample_pixel(right_q, x, y)
            low_mask[y, x] = downsample_mask(mask_high, x, y)

    high_blend = np.zeros((BLEND_H, BLEND_W, 4), dtype=np.int32)
    for y in range(BLEND_H):
        for x in range(BLEND_W):
            low00, low10, low01, low11 = low_neighbor(low_left, x, y)
            lap1 = laplacian_pixel(left_q[y, x].tolist(), low00, low10, low01, low11, x, y)
            low00, low10, low01, low11 = low_neighbor(low_right, x, y)
            lap2 = laplacian_pixel(right_q[y, x].tolist(), low00, low10, low01, low11, x, y)
            high_blend[y, x] = blend_pixel(lap1, lap2, int(mask_high[y, x]))

    low_blend = np.zeros((LOW_H, LOW_W, 4), dtype=np.int32)
    for y in range(LOW_H):
        for x in range(LOW_W):
            low_blend[y, x] = blend_pixel(low_left[y, x].tolist(), low_right[y, x].tolist(), int(low_mask[y, x]))

    blend_out = np.zeros((BLEND_H, BLEND_W), dtype=np.uint32)
    for y in range(BLEND_H):
        for x in range(BLEND_W):
            low00, low10, low01, low11 = low_neighbor(low_blend, x, y)
            recon = reconstruct_pixel(low00, low10, low01, low11, high_blend[y, x].tolist(), x, y)
            blend_out[y, x] = qpixel_to_word(recon)

    canvas[:, X2 - PAD : X2 - PAD + BLEND_W] = blend_out

    write_words32(outdir / "left.hex", left_words.reshape(-1).tolist())
    write_words32(outdir / "right.hex", right_words.reshape(-1).tolist())
    write_words16(outdir / "map1_x.hex", map1_x.reshape(-1).tolist())
    write_words16(outdir / "map1_y.hex", map1_y.reshape(-1).tolist())
    write_words16(outdir / "map2_x.hex", map2_x.reshape(-1).tolist())
    write_words16(outdir / "map2_y.hex", map2_y.reshape(-1).tolist())
    write_words16(outdir / "mask_high.hex", mask_high.reshape(-1).tolist())
    write_words32(outdir / "expected_canvas.hex", canvas.reshape(-1).tolist())


def compare_case(outdir: Path) -> None:
    expected = read_words(outdir / "expected_canvas.hex")
    actual = read_words(outdir / "canvas_out.hex")
    if len(expected) != len(actual):
        raise SystemExit(f"word-count mismatch: expected {len(expected)} got {len(actual)}")
    for index, (exp, got) in enumerate(zip(expected, actual)):
        if exp.lower() != got.lower():
            y = index // CANVAS_W
            x = index % CANVAS_W
            raise SystemExit(f"canvas mismatch at ({x}, {y}): expected 0x{exp} got 0x{got}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or compare reduced RTL stitch cases for two-image assets")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--left", required=True)
    gen.add_argument("--right", required=True)
    gen.add_argument("--outdir", required=True)
    gen.add_argument("--control-dir", default=None)

    cmp = sub.add_parser("compare")
    cmp.add_argument("--outdir", required=True)

    args = parser.parse_args()
    if args.cmd == "generate":
        build_case(
            Path(args.left),
            Path(args.right),
            Path(args.outdir),
            None if args.control_dir is None else Path(args.control_dir),
        )
    else:
        compare_case(Path(args.outdir))


if __name__ == "__main__":
    main()
