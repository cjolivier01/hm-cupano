#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

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


def write_words(path: Path, words: list[int]) -> None:
    path.write_text("\n".join(f"{word:08x}" for word in words) + "\n", encoding="ascii")


def read_words(path: Path) -> list[str]:
    words: list[str] = []
    for line in path.read_text(encoding="ascii").splitlines():
        word = line.strip().lower()
        if not word or word.startswith("//") or word.startswith("@"):
            continue
        words.append(word)
    return words


def build_case(left_path: Path, right_path: Path, outdir: Path) -> None:
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left is None:
        raise FileNotFoundError(left_path)
    if right is None:
        raise FileNotFoundError(right_path)

    left_crop = center_crop(left, INPUT_W, INPUT_H)
    right_crop = center_crop(right, INPUT_W, INPUT_H)

    left_words = np.array([[pack_rgba_word(left_crop[y, x]) for x in range(INPUT_W)] for y in range(INPUT_H)], dtype=np.uint32)
    right_words = np.array([[pack_rgba_word(right_crop[y, x]) for x in range(INPUT_W)] for y in range(INPUT_H)], dtype=np.uint32)

    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint32)
    canvas[:, 0:INPUT_W] = left_words

    blend_left = np.zeros((BLEND_H, BLEND_W), dtype=np.uint32)
    blend_right = np.zeros((BLEND_H, BLEND_W), dtype=np.uint32)
    blend_left[:, 0:OVERLAP + PAD] = canvas[:, X2 - PAD:X2 - PAD + OVERLAP + PAD]

    canvas[:, X2:X2 + INPUT_W] = right_words
    blend_right[:, PAD:PAD + OVERLAP + PAD] = canvas[:, X2:X2 + OVERLAP + PAD]

    mask_high = np.zeros((BLEND_H, BLEND_W), dtype=np.uint16)
    mask_high[:, :PAD + (OVERLAP // 2)] = MASK_ONE

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

    canvas[:, X2 - PAD:X2 - PAD + BLEND_W] = blend_out

    outdir.mkdir(parents=True, exist_ok=True)
    write_words(outdir / "left.hex", left_words.reshape(-1).tolist())
    write_words(outdir / "right.hex", right_words.reshape(-1).tolist())
    write_words(outdir / "expected_canvas.hex", canvas.reshape(-1).tolist())


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
    parser = argparse.ArgumentParser(description="Generate or compare reduced RTL stitch cases for left/right assets")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--left", required=True)
    gen.add_argument("--right", required=True)
    gen.add_argument("--outdir", required=True)

    cmp = sub.add_parser("compare")
    cmp.add_argument("--outdir", required=True)

    args = parser.parse_args()
    if args.cmd == "generate":
        build_case(Path(args.left), Path(args.right), Path(args.outdir))
    else:
        compare_case(Path(args.outdir))


if __name__ == "__main__":
    main()
