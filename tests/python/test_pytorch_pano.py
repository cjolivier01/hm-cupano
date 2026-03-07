from __future__ import annotations

import numpy as np
import pytest
import torch

from cupano import ControlMasks, ControlMasksN, CudaStitchPano, CudaStitchPanoN, SpatialTiff


def identity_map_x(width: int, height: int) -> np.ndarray:
    return np.tile(np.arange(width, dtype=np.uint16), (height, 1))


def identity_map_y(width: int, height: int) -> np.ndarray:
    return np.tile(np.arange(height, dtype=np.uint16)[:, None], (1, width))


def constant_image(width: int, height: int, rgba: tuple[float, float, float, float], device: torch.device) -> torch.Tensor:
    image = torch.zeros((1, height, width, 4), dtype=torch.float32, device=device)
    image[..., 0] = rgba[0]
    image[..., 1] = rgba[1]
    image[..., 2] = rgba[2]
    image[..., 3] = rgba[3]
    return image


def patterned_image(width: int, height: int, idx: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(height, dtype=torch.float32, device=device).view(1, height, 1, 1)
    xs = torch.arange(width, dtype=torch.float32, device=device).view(1, 1, width, 1)
    base = float(idx * 50)
    image = torch.zeros((1, height, width, 4), dtype=torch.float32, device=device)
    image[..., 0] = base + (xs[..., 0] % 17)
    image[..., 1] = base + (ys[..., 0] % 19)
    image[..., 2] = base + ((xs[..., 0] + ys[..., 0]) % 23)
    image[..., 3] = 255.0
    return image


def make_two_masks(width: int, height: int, seam: np.ndarray, x2: int) -> ControlMasks:
    masks = ControlMasks()
    masks.img1_col = identity_map_x(width, height)
    masks.img1_row = identity_map_y(width, height)
    masks.img2_col = identity_map_x(width, height)
    masks.img2_row = identity_map_y(width, height)
    masks.whole_seam_mask_image = seam
    masks.positions = [SpatialTiff(0.0, 0.0), SpatialTiff(float(x2), 0.0)]
    assert masks.is_valid()
    return masks


def make_n_masks(sizes: list[tuple[int, int]], positions: list[tuple[int, int]], seam_index: np.ndarray) -> ControlMasksN:
    masks = ControlMasksN()
    masks.img_col = [identity_map_x(w, h) for w, h in sizes]
    masks.img_row = [identity_map_y(w, h) for w, h in sizes]
    masks.positions = [SpatialTiff(float(x), float(y)) for x, y in positions]
    masks.whole_seam_mask_indexed = seam_index.astype(np.uint8)
    assert masks.is_valid()
    return masks


def assert_tensor_equal(a: torch.Tensor, b: torch.Tensor, tol: float = 0.0) -> None:
    if tol == 0.0:
        assert torch.equal(a, b), (a - b).abs().max().item()
    else:
        assert torch.allclose(a, b, atol=tol, rtol=0.0), (a - b).abs().max().item()


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_cuda_pano_hard_seam_matches_expected_selection(device: torch.device) -> None:
    width = 256
    height = 32
    x2 = 128
    canvas_width = width + x2
    seam = np.zeros((height, canvas_width), dtype=np.uint8)
    seam[:, :192] = 1
    masks = make_two_masks(width, height, seam, x2)

    image1 = constant_image(width, height, (10.0, 20.0, 30.0, 255.0), device)
    image2 = constant_image(width, height, (100.0, 110.0, 120.0, 255.0), device)
    pano = CudaStitchPano(1, 0, masks, quiet=True)
    out = pano.process(image1, image2)

    expected = torch.zeros((1, height, canvas_width, 4), dtype=torch.float32, device=device)
    expected[:, :, :192, :] = image1[:, :, :192, :]
    expected[:, :, 192:, :] = image2[:, :, 64:, :]
    assert_tensor_equal(out, expected)



def test_cuda_pano_soft_seam_single_level_matches_binary_mask(device: torch.device) -> None:
    width = 256
    height = 32
    x2 = 128
    canvas_width = width + x2
    seam = np.zeros((height, canvas_width), dtype=np.uint8)
    seam[:, :192] = 1
    masks = make_two_masks(width, height, seam, x2)

    image1 = constant_image(width, height, (10.0, 20.0, 30.0, 255.0), device)
    image2 = constant_image(width, height, (100.0, 110.0, 120.0, 255.0), device)
    pano = CudaStitchPano(1, 1, masks, quiet=True)
    out = pano.process(image1, image2)

    expected = torch.zeros((1, height, canvas_width, 4), dtype=torch.float32, device=device)
    expected[:, :, :192, :] = image1[:, :, :192, :]
    expected[:, :, 192:, :] = image2[:, :, 64:, :]
    assert_tensor_equal(out, expected, tol=1e-5)



def test_cuda_pano_n_minimize_blend_matches_full_blend(device: torch.device) -> None:
    width = 64
    height = 64
    seam = np.zeros((height, width), dtype=np.uint8)
    seam[:, width // 2 :] = 1
    masks = make_n_masks([(width, height), (width, height)], [(0, 0), (0, 0)], seam)
    images = [patterned_image(width, height, 0, device), patterned_image(width, height, 1, device)]

    pano_full = CudaStitchPanoN(1, 4, masks, quiet=True, minimize_blend=False)
    pano_mini = CudaStitchPanoN(1, 4, masks, quiet=True, minimize_blend=True)
    out_full = pano_full.process(images)
    out_mini = pano_mini.process(images)

    assert_tensor_equal(out_full, out_mini, tol=1e-5)



def test_cuda_pano_n_hard_seam_selects_indexed_image(device: torch.device) -> None:
    width = 48
    height = 24
    seam = np.zeros((height, width), dtype=np.uint8)
    seam[:, 16:32] = 1
    seam[:, 32:] = 2
    masks = make_n_masks([(width, height), (width, height), (width, height)], [(0, 0), (0, 0), (0, 0)], seam)
    images = [
        constant_image(width, height, (10.0, 0.0, 0.0, 255.0), device),
        constant_image(width, height, (0.0, 20.0, 0.0, 255.0), device),
        constant_image(width, height, (0.0, 0.0, 30.0, 255.0), device),
    ]

    pano = CudaStitchPanoN(1, 0, masks, quiet=True)
    out = pano.process(images)
    expected = torch.zeros((1, height, width, 4), dtype=torch.float32, device=device)
    expected[:, :, :16, :] = images[0][:, :, :16, :]
    expected[:, :, 16:32, :] = images[1][:, :, 16:32, :]
    expected[:, :, 32:, :] = images[2][:, :, 32:, :]
    assert_tensor_equal(out, expected)
