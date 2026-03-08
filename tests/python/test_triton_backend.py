from __future__ import annotations

import numpy as np
import pytest
import torch

from cupano import ControlMasks, ControlMasksN, CudaStitchPano, CudaStitchPanoN, SpatialTiff
from cupano.geometry import Rect
from cupano.ops import (
    _copy_roi_torch,
    _remap_to_canvas_torch,
    _remap_to_canvas_with_dest_map_torch,
    compute_laplacian,
    copy_roi,
    remap_to_canvas,
    remap_to_canvas_with_dest_map,
)
from cupano.triton_ops import triton_available

_MIN_TEST_FREE_BYTES = 1 << 30


def _has_enough_cuda_memory(min_free_bytes: int = _MIN_TEST_FREE_BYTES) -> bool:
    if not torch.cuda.is_available():
        return False
    free_bytes, _ = torch.cuda.mem_get_info()
    return free_bytes >= min_free_bytes


@pytest.fixture(scope="module")
def gpu_device() -> torch.device:
    if not torch.cuda.is_available() or not triton_available():
        pytest.skip("Triton GPU tests require torch CUDA/ROCm + Triton")
    if not _has_enough_cuda_memory():
        pytest.skip("Triton GPU tests require at least 1 GiB of free GPU memory")
    return torch.device("cuda")


def identity_map_x(width: int, height: int) -> np.ndarray:
    return np.tile(np.arange(width, dtype=np.uint16), (height, 1))


def identity_map_y(width: int, height: int) -> np.ndarray:
    return np.tile(np.arange(height, dtype=np.uint16)[:, None], (1, width))


def patterned_image(width: int, height: int, idx: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(height, dtype=torch.float32, device=device).view(1, height, 1, 1)
    xs = torch.arange(width, dtype=torch.float32, device=device).view(1, 1, width, 1)
    image = torch.zeros((1, height, width, 4), dtype=torch.uint8, device=device)
    image[..., 0] = (idx * 20 + xs[..., 0]).to(torch.uint8)
    image[..., 1] = (idx * 30 + ys[..., 0]).to(torch.uint8)
    image[..., 2] = ((xs[..., 0] + ys[..., 0]) % 255).to(torch.uint8)
    image[..., 3] = 255
    return image.contiguous()


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


def assert_equal(a: torch.Tensor, b: torch.Tensor) -> None:
    assert torch.equal(a, b), (a.to(torch.int16) - b.to(torch.int16)).abs().max().item()


def test_copy_roi_triton_matches_reference(gpu_device: torch.device) -> None:
    src = torch.arange(1 * 16 * 16 * 4, device=gpu_device, dtype=torch.float32).reshape(1, 16, 16, 4).contiguous()
    dest_ref = torch.zeros((1, 20, 20, 4), device=gpu_device, dtype=torch.uint8)
    dest_triton = dest_ref.clone()

    _copy_roi_torch(src, dest_ref, Rect(0, 0, 8, 7), 3, 4, 5, 6)
    copy_roi(src, dest_triton, Rect(0, 0, 8, 7), 3, 4, 5, 6, backend="triton")
    assert_equal(dest_ref, dest_triton)


def test_copy_roi_negative_src_offset_matches_reference(gpu_device: torch.device) -> None:
    src = patterned_image(16, 12, 0, gpu_device)
    dest_ref = torch.zeros((1, 12, 20, 4), device=gpu_device, dtype=torch.uint8)
    dest_triton = dest_ref.clone()

    _copy_roi_torch(src, dest_ref, Rect(0, 0, 14, 8), -5, 2, 3, 1)
    copy_roi(src, dest_triton, Rect(0, 0, 14, 8), -5, 2, 3, 1, backend="triton")
    assert_equal(dest_ref, dest_triton)


def test_remap_triton_matches_reference(gpu_device: torch.device) -> None:
    src = patterned_image(32, 24, 0, gpu_device)
    dest_ref = torch.zeros((1, 32, 40, 4), device=gpu_device, dtype=torch.uint8)
    dest_triton = dest_ref.clone()
    map_x = torch.from_numpy(identity_map_x(32, 24)).to(device=gpu_device, dtype=torch.int32).contiguous()
    map_y = torch.from_numpy(identity_map_y(32, 24)).to(device=gpu_device, dtype=torch.int32).contiguous()

    _remap_to_canvas_torch(src, dest_ref, map_x, map_y, 4, 3)
    remap_to_canvas(src, dest_triton, map_x, map_y, 4, 3, backend="triton")
    assert_equal(dest_ref, dest_triton)


def test_remap_with_dest_map_triton_matches_reference(gpu_device: torch.device) -> None:
    src = patterned_image(24, 24, 1, gpu_device)
    dest_ref = torch.zeros((1, 24, 24, 4), device=gpu_device, dtype=torch.uint8)
    dest_triton = dest_ref.clone()
    map_x = torch.from_numpy(identity_map_x(24, 24)).to(device=gpu_device, dtype=torch.int32).contiguous()
    map_y = torch.from_numpy(identity_map_y(24, 24)).to(device=gpu_device, dtype=torch.int32).contiguous()
    seam = torch.zeros((24, 24), device=gpu_device, dtype=torch.uint8)
    seam[:, 12:] = 1

    _remap_to_canvas_with_dest_map_torch(src, dest_ref, map_x, map_y, 0, seam, 0, 0)
    remap_to_canvas_with_dest_map(src, dest_triton, map_x, map_y, 0, seam, 0, 0, backend="triton")
    assert_equal(dest_ref, dest_triton)


def test_compute_laplacian_renormalizes_fractional_alpha_weights(gpu_device: torch.device) -> None:
    high = torch.zeros((1, 4, 4, 4), dtype=torch.float32, device=gpu_device)
    low = torch.zeros((1, 2, 2, 4), dtype=torch.float32, device=gpu_device)
    low[:, 1, 1, :] = torch.tensor([100.0, 110.0, 120.0, 255.0], device=gpu_device)

    lap_triton = compute_laplacian(high, low, backend="triton")
    lap_ref = compute_laplacian(high.cpu(), low.cpu(), backend="auto").to(gpu_device)

    assert torch.allclose(lap_triton, lap_ref, atol=1e-5, rtol=0.0)
    assert torch.allclose(
        lap_triton[0, 1, 1, :3],
        torch.tensor([-100.0, -110.0, -120.0], device=gpu_device),
        atol=1e-5,
        rtol=0.0,
    )
    assert lap_triton[0, 1, 1, 3].item() == 0.0


def test_cuda_pano_triton_matches_reference(gpu_device: torch.device) -> None:
    width = 96
    height = 32
    x2 = 48
    seam = np.zeros((height, width + x2), dtype=np.uint8)
    seam[:, :72] = 1
    masks = make_two_masks(width, height, seam, x2)
    image1_gpu = patterned_image(width, height, 0, gpu_device)
    image2_gpu = patterned_image(width, height, 2, gpu_device)
    image1_cpu = image1_gpu.cpu()
    image2_cpu = image2_gpu.cpu()

    ref_pano = CudaStitchPano(1, 1, masks, quiet=True, backend="auto")
    triton_pano = CudaStitchPano(1, 1, masks, quiet=True, backend="triton")
    out_ref = ref_pano.process(image1_cpu, image2_cpu).cpu()
    out_triton = triton_pano.process(image1_gpu, image2_gpu).cpu()
    assert_equal(out_ref, out_triton)


def test_cuda_pano_n_triton_matches_reference(gpu_device: torch.device) -> None:
    width = 48
    height = 24
    seam = np.zeros((height, width), dtype=np.uint8)
    seam[:, 16:32] = 1
    seam[:, 32:] = 2
    masks = make_n_masks([(width, height), (width, height), (width, height)], [(0, 0), (0, 0), (0, 0)], seam)
    inputs_gpu = [patterned_image(width, height, i, gpu_device) for i in range(3)]
    inputs_cpu = [image.cpu() for image in inputs_gpu]

    ref_pano = CudaStitchPanoN(1, 0, masks, quiet=True, backend="auto")
    triton_pano = CudaStitchPanoN(1, 0, masks, quiet=True, backend="triton")
    out_ref = ref_pano.process(inputs_cpu).cpu()
    out_triton = triton_pano.process(inputs_gpu).cpu()
    assert_equal(out_ref, out_triton)
