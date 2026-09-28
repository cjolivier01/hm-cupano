from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from .canvas import CanvasManager, CanvasManagerN
from .feather import DEFAULT_FEATHER_FRACTION, FeatherParams, build_weights as build_feather_weights
from .geometry import CanvasInfo, Rect
from .masks import UNMAPPED_POSITION_VALUE, ControlMasks, ControlMasksN
from .ops import (
    Backend,
    LaplacianBlendWorkspace,
    cast_like,
    copy_roi,
    ensure_batched,
    BlendMode,
    laplacian_blend_n,
    remap_to_canvas,
    remap_to_canvas_with_dest_map,
)
from .status import CudaStatus, CudaStatusError


def _scaled_overlap_padding(original_canvas_width: int, scaled_canvas_width: int, base_pad: int = 128) -> int:
    if original_canvas_width <= 0 or scaled_canvas_width <= 0 or scaled_canvas_width >= original_canvas_width:
        return base_pad
    return max(1, int(math.floor(base_pad * scaled_canvas_width / original_canvas_width)))


@dataclass
class StitchingContext:
    batch_size: int
    is_hard_seam: bool
    remap_1_x: np.ndarray | None = None
    remap_1_y: np.ndarray | None = None
    remap_2_x: np.ndarray | None = None
    remap_2_y: np.ndarray | None = None
    hard_seam: np.ndarray | None = None
    blend_seam: np.ndarray | None = None
    blend_mask: np.ndarray | None = None


@dataclass
class StitchingContextN:
    batch_size: int
    is_hard_seam: bool
    n_images: int
    remap_x: list[np.ndarray]
    remap_y: list[np.ndarray]
    blend_mask: np.ndarray | None = None
    seam_index: np.ndarray | None = None


@dataclass
class RemapRoiInfo:
    roi: Rect = field(default_factory=Rect)
    offset_x: int = 0
    offset_y: int = 0


@dataclass
class _TwoImageSoftScratch:
    full1: torch.Tensor
    full2: torch.Tensor
    blend_workspace: LaplacianBlendWorkspace = field(
        default_factory=LaplacianBlendWorkspace
    )


@dataclass
class _NImageSoftScratch:
    compute_buffers: list[torch.Tensor]
    blend_workspace: LaplacianBlendWorkspace = field(
        default_factory=LaplacianBlendWorkspace
    )


@dataclass
class _GraphState:
    key: tuple[object, ...]
    graph: object
    static_inputs: list[torch.Tensor]
    static_output: torch.Tensor


def _status_or_raise(status: CudaStatus) -> None:
    if not status.ok():
        raise CudaStatusError(status)


class _DeviceCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], torch.Tensor] = {}

    def tensor(
        self, array: np.ndarray, device: torch.device, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        use_dtype = dtype or torch.from_numpy(array).dtype
        key = (str(device), f"{array.__array_interface__['data'][0]}:{use_dtype}")
        if key not in self._cache:
            self._cache[key] = torch.as_tensor(array, device=device, dtype=use_dtype)
        return self._cache[key]


def _cuda_graphs_supported(device: torch.device) -> bool:
    return (
        device.type == "cuda"
        and torch.cuda.is_available()
        and torch.version.hip is None
        and hasattr(torch.cuda, "CUDAGraph")
    )


def _write_batched_tiffs(tensor: torch.Tensor, out_dir: Path, stem: str) -> None:
    array = tensor.detach().cpu().numpy()
    if array.ndim == 3:
        if not cv2.imwrite(
            str(out_dir / f"{stem}.tiff"), array.astype(np.float32, copy=False)
        ):
            raise RuntimeError(f"Unable to write {(out_dir / f'{stem}.tiff')}")
        return
    if array.ndim != 4:
        raise ValueError(f"Expected HWC or BHWC tensor, got shape {tuple(array.shape)}")
    for batch_item in range(array.shape[0]):
        path = out_dir / f"{stem}_batch_{batch_item}.tiff"
        if not cv2.imwrite(str(path), array[batch_item].astype(np.float32, copy=False)):
            raise RuntimeError(f"Unable to write {path}")


class CudaStitchPano:
    def __init__(
        self,
        batch_size: int,
        num_levels: int,
        control_masks: ControlMasks,
        quiet: bool = False,
        minimize_blend: bool = True,
        max_output_width: int = 0,
        backend: Backend = "auto",
        enable_cuda_graphs: bool = True,
        blend_mode: BlendMode | None = None,
        feather_fraction: float = DEFAULT_FEATHER_FRACTION,
    ) -> None:
        self._status = CudaStatus()
        # `blend_mode=None` reproduces the historical convention where num_levels == 0 means a hard
        # seam. Mirrors BlendSettings' implicit int constructor in blendMode.h.
        self._blend_mode: BlendMode = blend_mode or ("hard" if num_levels == 0 else "laplacian")
        feather_fraction = float(feather_fraction)
        self._feather_radius_px = 0.0
        if self._blend_mode == "alpha":
            num_levels = 1
        self._num_levels = num_levels
        self._minimize_blend = bool(minimize_blend and self._blend_mode != "hard")
        # A status rather than a raise, matching the two checks below and BlendSettings::Validate.
        # A negative count used to build a soft stitcher that then ran max(1, num_levels) levels.
        if num_levels < 0:
            self._status = CudaStatus(1, "Blend level count must not be negative")
            return
        if self._blend_mode == "laplacian" and num_levels < 1:
            # Matches BlendSettings::Laplacian(0).Validate(). Reachable only by naming the mode
            # explicitly; a bare 0 still means a hard seam.
            self._status = CudaStatus(1, "Laplacian blending requires at least one pyramid level")
            return
        if self._blend_mode not in ("hard", "laplacian", "alpha"):
            self._status = CudaStatus(1, "blend_mode must be hard, laplacian or alpha")
            return
        if self._blend_mode == "alpha" and (
            not math.isfinite(feather_fraction) or not 0.0 <= feather_fraction <= 1.0
        ):
            self._status = CudaStatus(1, "Alpha blend feather fraction must be in [0, 1]")
            return
        self._backend = backend
        self._enable_cuda_graphs = enable_cuda_graphs
        self._cache = _DeviceCache()
        self._soft_scratch: dict[tuple[str, int], _TwoImageSoftScratch] = {}
        self._graph_state: _GraphState | None = None
        self._graph_disabled = False
        if not control_masks.is_valid():
            self._status = CudaStatus(1, "Stitching masks were not able to be loaded")
            return
        original_canvas_width = control_masks.canvas_width()
        if max_output_width > 0 and original_canvas_width > max_output_width:
            control_masks = copy.copy(control_masks)
            if not control_masks.scale_to_max_output_width(max_output_width):
                self._status = CudaStatus(1, "Stitching masks lost a seam class while applying max_output_width")
                return

        self._context = StitchingContext(
            batch_size=batch_size, is_hard_seam=(self._blend_mode == "hard")
        )
        canvas_w = control_masks.canvas_width()
        canvas_h = control_masks.canvas_height()
        if not quiet:
            print(f"Stitched canvas size: {canvas_w} x {canvas_h}")
        x0 = int(control_masks.positions[0].xpos)
        x1 = int(control_masks.positions[1].xpos)
        scaled_overlap = x0 + control_masks.img1_col.shape[1] - x1
        if x0 > x1 or scaled_overlap <= 0:
            self._minimize_blend = False

        self._canvas_manager = CanvasManager(
            CanvasInfo(
                width=canvas_w,
                height=canvas_h,
                positions=[
                    (
                        int(control_masks.positions[0].xpos),
                        int(control_masks.positions[0].ypos),
                    ),
                    (
                        int(control_masks.positions[1].xpos),
                        int(control_masks.positions[1].ypos),
                    ),
                ],
            ),
            minimize_blend=self._minimize_blend,
            overlap_pad=_scaled_overlap_padding(original_canvas_width, canvas_w),
        )
        self._canvas_manager._remapper_1.width = control_masks.img1_col.shape[1]
        self._canvas_manager._remapper_1.height = control_masks.img1_col.shape[0]
        self._canvas_manager._remapper_2.width = control_masks.img2_col.shape[1]
        self._canvas_manager._remapper_2.height = control_masks.img2_col.shape[0]
        self._canvas_manager.updateMinimizeBlend(
            (control_masks.img1_col.shape[1], control_masks.img1_col.shape[0]),
            (control_masks.img2_col.shape[1], control_masks.img2_col.shape[0]),
        )

        # The feather needs the whole canvas: distance transforms are non-local, so building one
        # from a blend-ROI crop invents a boundary at the crop edge, and the camera positions
        # below are canvas coordinates that a cropped mask no longer shares.
        full_seam = self._canvas_manager.padMaskMat(control_masks.whole_seam_mask_image)
        blend_seam = self._canvas_manager.cropToBlendRoi(full_seam)
        self._context.remap_1_x = control_masks.img1_col
        self._context.remap_1_y = control_masks.img1_row
        self._context.remap_2_x = control_masks.img2_col
        self._context.remap_2_y = control_masks.img2_row
        self._context.hard_seam = blend_seam.astype(np.uint8, copy=False)
        self._context.blend_seam = blend_seam.astype(np.float32, copy=False)
        if not self._context.is_hard_seam:
            if self._blend_mode == "alpha":
                # Two-image seam labels double as blend weights: label 1 owns image 1, label 0 owns
                # image 2. Order the feather inputs to match so channel 1 is image 1's weight.
                try:
                    feathered = build_feather_weights(
                        full_seam.astype(np.uint8, copy=False),
                        [control_masks.img2_col, control_masks.img1_col],
                        [control_masks.img2_row, control_masks.img1_row],
                        [
                            (
                                int(control_masks.positions[1].xpos),
                                int(control_masks.positions[1].ypos),
                            ),
                            (
                                int(control_masks.positions[0].xpos),
                                int(control_masks.positions[0].ypos),
                            ),
                        ],
                        2,
                        FeatherParams(fraction=feather_fraction),
                    )
                except ValueError as error:
                    # A status, not a raise: the C++ reports this through Result::error, and
                    # the validations above set _status too.
                    self._status = CudaStatus(1, str(error))
                    return
                self._feather_radius_px = feathered.radius_px
                # This path crops to the geometric overlap band; it has no seam-derived ROI.
                # The per-pixel cap keeps R(p) <= 2*coverage, so
                # the band cannot leave the cameras' shared coverage, which is inside that band.
                weights = self._canvas_manager.cropToBlendRoi(feathered.weights)
                blend_mask = np.empty(self._context.blend_seam.shape + (2,), dtype=np.float32)
                blend_mask[..., 0] = weights[..., 1]
                blend_mask[..., 1] = weights[..., 0]
            else:
                blend_mask = np.empty(
                    self._context.blend_seam.shape + (2,), dtype=np.float32
                )
                blend_mask[..., 0] = self._context.blend_seam
                blend_mask[..., 1] = 1.0 - self._context.blend_seam
            self._context.blend_mask = blend_mask

    @property
    def status(self) -> CudaStatus:
        return self._status

    @property
    def blend_mode(self) -> BlendMode:
        return self._blend_mode

    @property
    def feather_radius_px(self) -> float:
        """Widest crossfade any seam pixel got, in canvas pixels. Zero unless alpha mode."""
        return self._feather_radius_px

    def canvas_width(self) -> int:
        return self._canvas_manager.canvas_width()

    def canvas_height(self) -> int:
        return self._canvas_manager.canvas_height()

    def batch_size(self) -> int:
        return self._context.batch_size

    def _tensor(
        self, array: np.ndarray, device: torch.device, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        return self._cache.tensor(array, device=device, dtype=dtype)

    def _soft_scratch_for(
        self, device: torch.device, channels: int
    ) -> _TwoImageSoftScratch:
        key = (str(device), channels)
        scratch = self._soft_scratch.get(key)
        blend_h = int(self._context.blend_seam.shape[0])
        blend_w = int(self._context.blend_seam.shape[1])
        shape = (self._context.batch_size, blend_h, blend_w, channels)
        if scratch is None:
            scratch = _TwoImageSoftScratch(
                full1=torch.empty(shape, device=device, dtype=torch.float32),
                full2=torch.empty(shape, device=device, dtype=torch.float32),
            )
            self._soft_scratch[key] = scratch
        return scratch

    def _graph_key(
        self, input_image_1: torch.Tensor, input_image_2: torch.Tensor
    ) -> tuple[object, ...]:
        return (
            str(input_image_1.device),
            tuple(input_image_1.shape),
            tuple(input_image_2.shape),
            input_image_1.dtype,
            input_image_2.dtype,
        )

    def _process_impl(
        self,
        input_image_1: torch.Tensor,
        input_image_2: torch.Tensor,
        canvas: torch.Tensor,
    ) -> torch.Tensor:
        device = input_image_1.device
        channels = input_image_1.shape[-1]
        canvas.zero_()

        if self._context.is_hard_seam:
            seam = self._tensor(self._context.hard_seam, device, torch.uint8)
            remap_to_canvas_with_dest_map(
                input_image_1,
                canvas,
                self._tensor(self._context.remap_1_x, device, torch.int32),
                self._tensor(self._context.remap_1_y, device, torch.int32),
                1,
                seam,
                self._canvas_manager._x1,
                self._canvas_manager._y1,
                backend=self._backend,
            )
            remap_to_canvas_with_dest_map(
                input_image_2,
                canvas,
                self._tensor(self._context.remap_2_x, device, torch.int32),
                self._tensor(self._context.remap_2_y, device, torch.int32),
                0,
                seam,
                self._canvas_manager._x2,
                self._canvas_manager._y2,
                backend=self._backend,
            )
            return canvas

        scratch = self._soft_scratch_for(device, channels)
        full1 = scratch.full1
        full2 = scratch.full2
        full1.zero_()
        full2.zero_()

        remap_to_canvas(
            input_image_1,
            canvas if self._minimize_blend else full1,
            self._tensor(self._context.remap_1_x, device, torch.int32),
            self._tensor(self._context.remap_1_y, device, torch.int32),
            self._canvas_manager._x1,
            self._canvas_manager._y1,
            fill_invalid_alpha=True,
            backend=self._backend,
        )
        if self._minimize_blend:
            remap_to_canvas(
                input_image_1,
                full1,
                self._tensor(self._context.remap_1_x, device, torch.int32),
                self._tensor(self._context.remap_1_y, device, torch.int32),
                self._canvas_manager._remapper_1.xpos
                - self._canvas_manager.remapped_image_roi_blend_1.x,
                self._canvas_manager._y1,
                roi=self._canvas_manager.remapped_image_roi_blend_1,
                fill_invalid_alpha=True,
                backend=self._backend,
            )

        remap_to_canvas(
            input_image_2,
            canvas if self._minimize_blend else full2,
            self._tensor(self._context.remap_2_x, device, torch.int32),
            self._tensor(self._context.remap_2_y, device, torch.int32),
            self._canvas_manager._x2,
            self._canvas_manager._y2,
            fill_invalid_alpha=True,
            backend=self._backend,
        )
        if self._minimize_blend:
            remap_to_canvas(
                input_image_2,
                full2,
                self._tensor(self._context.remap_2_x, device, torch.int32),
                self._tensor(self._context.remap_2_y, device, torch.int32),
                self._canvas_manager._remapper_2.xpos
                - self._canvas_manager.remapped_image_roi_blend_2.x,
                self._canvas_manager._y2,
                roi=self._canvas_manager.remapped_image_roi_blend_2,
                fill_invalid_alpha=True,
                backend=self._backend,
            )

        mask = self._tensor(self._context.blend_mask, device, torch.float32)
        blended = laplacian_blend_n(
            [full1, full2],
            mask,
            self._num_levels,
            backend=self._backend,
            workspace=scratch.blend_workspace,
        )
        copy_roi(
            blended,
            canvas,
            Rect(0, 0, blended.shape[2], blended.shape[1]),
            0,
            0,
            (
                self._canvas_manager._x2 - self._canvas_manager.overlap_padding()
                if self._minimize_blend
                else 0
            ),
            0,
            backend=self._backend,
        )
        return cast_like(canvas, input_image_1.dtype)

    def _capture_graph(
        self, input_image_1: torch.Tensor, input_image_2: torch.Tensor
    ) -> _GraphState:
        static_input_1 = torch.empty_like(input_image_1)
        static_input_2 = torch.empty_like(input_image_2)
        static_output = torch.empty(
            (
                self._context.batch_size,
                self.canvas_height(),
                self.canvas_width(),
                input_image_1.shape[-1],
            ),
            device=input_image_1.device,
            dtype=input_image_1.dtype,
        )
        static_input_1.copy_(input_image_1)
        static_input_2.copy_(input_image_2)
        warmup_stream = torch.cuda.Stream(device=input_image_1.device)
        current_stream = torch.cuda.current_stream(device=input_image_1.device)
        with torch.cuda.stream(warmup_stream):
            self._process_impl(static_input_1, static_input_2, static_output)
            self._process_impl(static_input_1, static_input_2, static_output)
        current_stream.wait_stream(warmup_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._process_impl(static_input_1, static_input_2, static_output)
        return _GraphState(
            key=self._graph_key(input_image_1, input_image_2),
            graph=graph,
            static_inputs=[static_input_1, static_input_2],
            static_output=static_output,
        )

    def _maybe_run_graph(
        self,
        input_image_1: torch.Tensor,
        input_image_2: torch.Tensor,
        canvas: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if (
            self._graph_disabled
            or not self._enable_cuda_graphs
            or not _cuda_graphs_supported(input_image_1.device)
        ):
            return None
        if input_image_1.device != input_image_2.device:
            return None
        key = self._graph_key(input_image_1, input_image_2)
        if self._graph_state is None or self._graph_state.key != key:
            try:
                self._graph_state = self._capture_graph(input_image_1, input_image_2)
            except Exception:
                self._graph_disabled = True
                self._graph_state = None
                return None
        assert self._graph_state is not None
        self._graph_state.static_inputs[0].copy_(input_image_1)
        self._graph_state.static_inputs[1].copy_(input_image_2)
        self._graph_state.graph.replay()
        if canvas is not None:
            canvas.copy_(self._graph_state.static_output)
            return canvas
        out = torch.empty_like(self._graph_state.static_output)
        out.copy_(self._graph_state.static_output)
        return out

    def process(
        self,
        input_image_1: torch.Tensor,
        input_image_2: torch.Tensor,
        canvas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _status_or_raise(self._status)
        input_image_1 = ensure_batched(input_image_1)
        input_image_2 = ensure_batched(input_image_2)
        if (
            input_image_1.shape[0] != self._context.batch_size
            or input_image_2.shape[0] != self._context.batch_size
        ):
            raise ValueError("Input batch size does not match stitcher batch size")
        if input_image_1.shape[-1] not in (3, 4):
            raise ValueError("Only 3- and 4-channel inputs are supported")

        if canvas is not None:
            canvas = ensure_batched(canvas)
        graph_out = self._maybe_run_graph(input_image_1, input_image_2, canvas)
        if graph_out is not None:
            return graph_out

        device = input_image_1.device
        if canvas is None:
            canvas = torch.empty(
                (
                    self._context.batch_size,
                    self.canvas_height(),
                    self.canvas_width(),
                    input_image_1.shape[-1],
                ),
                device=device,
                dtype=input_image_1.dtype,
            )
        return self._process_impl(input_image_1, input_image_2, canvas)

    def dump_soft_blend_pyramid(
        self,
        directory: str | Path,
        *,
        device: torch.device | None = None,
        channels: int | None = None,
    ) -> None:
        if self._context.is_hard_seam:
            raise ValueError("Soft-blend pyramid is only available when num_levels > 0")
        if device is None or channels is None:
            if len(self._soft_scratch) != 1:
                raise ValueError(
                    "device and channels must be provided when multiple scratch buffers exist"
                )
            (device_key, channels_key), scratch = next(iter(self._soft_scratch.items()))
            device = torch.device(device_key)
            channels = channels_key
        else:
            scratch = self._soft_scratch.get((str(device), channels))
            if scratch is None:
                raise ValueError(
                    f"No soft scratch buffers available for device={device} channels={channels}"
                )

        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        workspace = scratch.blend_workspace
        levels = len(workspace.mask_pyr)
        if levels == 0:
            raise ValueError("No Laplacian workspace is populated yet")

        (out_dir / "metadata.txt").write_text(
            "\n".join(
                [
                    f"num_levels={levels}",
                    f"batch_size={self._context.batch_size}",
                    f"channels={channels}",
                    *[
                        f"level_{level}={workspace.mask_pyr[level].shape[1]}x{workspace.mask_pyr[level].shape[0]}"
                        for level in range(levels)
                    ],
                ]
            )
            + "\n"
        )

        for subdir in (
            "gauss1",
            "gauss2",
            "mask",
            "lap1",
            "lap2",
            "blend",
            "reconstruct",
        ):
            (out_dir / subdir).mkdir(parents=True, exist_ok=True)

        for level in range(levels):
            _write_batched_tiffs(
                workspace.gauss[0][level], out_dir / "gauss1", f"level_{level}"
            )
            _write_batched_tiffs(
                workspace.gauss[1][level], out_dir / "gauss2", f"level_{level}"
            )
            _write_batched_tiffs(
                workspace.mask_pyr[level], out_dir / "mask", f"level_{level}"
            )
            lap1 = (
                workspace.gauss[0][level]
                if level == levels - 1
                else workspace.laps[0][level]
            )
            lap2 = (
                workspace.gauss[1][level]
                if level == levels - 1
                else workspace.laps[1][level]
            )
            _write_batched_tiffs(lap1, out_dir / "lap1", f"level_{level}")
            _write_batched_tiffs(lap2, out_dir / "lap2", f"level_{level}")
            _write_batched_tiffs(
                workspace.blended[level], out_dir / "blend", f"level_{level}"
            )
            _write_batched_tiffs(
                workspace.recon[level], out_dir / "reconstruct", f"level_{level}"
            )


class CudaStitchPanoN:
    def __init__(
        self,
        batch_size: int,
        num_levels: int,
        control_masks: ControlMasksN,
        quiet: bool = False,
        minimize_blend: bool = True,
        backend: Backend = "auto",
        enable_cuda_graphs: bool = True,
        max_output_width: int = 0,
        blend_mode: BlendMode | None = None,
        feather_fraction: float = DEFAULT_FEATHER_FRACTION,
    ) -> None:
        self._status = CudaStatus()
        # `blend_mode=None` reproduces the historical convention where num_levels == 0 means a hard
        # seam. Mirrors BlendSettings' implicit int constructor in blendMode.h.
        self._blend_mode: BlendMode = blend_mode or ("hard" if num_levels == 0 else "laplacian")
        feather_fraction = float(feather_fraction)
        self._feather_radius_px = 0.0
        feather_roi_radius_px = 0.0
        if self._blend_mode == "alpha":
            num_levels = 1
        self._num_levels = num_levels
        self._minimize_blend = bool(minimize_blend and self._blend_mode != "hard")
        # A status rather than a raise, matching the two checks below and BlendSettings::Validate.
        # A negative count used to build a soft stitcher that then ran max(1, num_levels) levels.
        if num_levels < 0:
            self._status = CudaStatus(1, "Blend level count must not be negative")
            return
        if self._blend_mode == "laplacian" and num_levels < 1:
            # Matches BlendSettings::Laplacian(0).Validate(). Reachable only by naming the mode
            # explicitly; a bare 0 still means a hard seam.
            self._status = CudaStatus(1, "Laplacian blending requires at least one pyramid level")
            return
        if self._blend_mode not in ("hard", "laplacian", "alpha"):
            self._status = CudaStatus(1, "blend_mode must be hard, laplacian or alpha")
            return
        if self._blend_mode == "alpha" and (
            not math.isfinite(feather_fraction) or not 0.0 <= feather_fraction <= 1.0
        ):
            self._status = CudaStatus(1, "Alpha blend feather fraction must be in [0, 1]")
            return
        self._backend = backend
        self._enable_cuda_graphs = enable_cuda_graphs
        self._cache = _DeviceCache()
        self._blend_roi_canvas = Rect()
        self._write_roi_canvas = Rect()
        self._remap_rois: list[RemapRoiInfo] = []
        self._soft_scratch: dict[tuple[str, int, bool], _NImageSoftScratch] = {}
        self._graph_state: _GraphState | None = None
        self._graph_disabled = False

        if not control_masks.is_valid():
            self._status = CudaStatus(
                1, "Stitching masks (N-image) could not be loaded"
            )
            return
        original_canvas_width = control_masks.canvas_width()
        if max_output_width > 0 and original_canvas_width > max_output_width:
            control_masks = copy.copy(control_masks)
            if not control_masks.scale_to_max_output_width(max_output_width):
                self._status = CudaStatus(
                    2, "max_output_width removes one or more N-image seam classes"
                )
                return

        n = len(control_masks.img_col)
        if n < 2 or n > 8:
            self._status = CudaStatus(2, "Unsupported N for blend (supported 2..8)")
            return

        canvas_w = control_masks.canvas_width()
        canvas_h = control_masks.canvas_height()
        if not quiet:
            print(f"Stitched (N-image) canvas size: {canvas_w} x {canvas_h}")
        positions = [(int(p.xpos), int(p.ypos)) for p in control_masks.positions]
        self._canvas_manager = CanvasManagerN(
            CanvasInfo(canvas_w, canvas_h, positions),
            minimize_blend=self._minimize_blend,
            overlap_pad=_scaled_overlap_padding(original_canvas_width, canvas_w),
        )
        for idx, remap in enumerate(control_masks.img_col):
            self._canvas_manager.set_remap_size(idx, (remap.shape[1], remap.shape[0]))

        seam_index_padded = self._canvas_manager.convertMaskMat(
            control_masks.whole_seam_mask_indexed.copy()
        )
        seam_for_blend = seam_index_padded
        self._context = StitchingContextN(
            batch_size=batch_size,
            is_hard_seam=(self._blend_mode == "hard"),
            n_images=n,
            remap_x=list(control_masks.img_col),
            remap_y=list(control_masks.img_row),
            seam_index=seam_index_padded,
        )

        # Alpha mode resolves its weights on the full canvas before any ROI cropping: distance
        # transforms are non-local, so building them from a cropped seam would invent a boundary at
        # the crop edge. Resolving first also yields the radius the ROI padding below must cover.
        feather_weights_full = None
        seam_index_for_roi = seam_index_padded
        if not self._context.is_hard_seam and self._blend_mode == "alpha":
            try:
                feathered = build_feather_weights(
                    seam_index_padded.astype(np.uint8, copy=False),
                    list(control_masks.img_col),
                    list(control_masks.img_row),
                    [
                        (int(p[0]), int(p[1]))
                        for p in self._canvas_manager.canvas_positions()
                    ],
                    n,
                    FeatherParams(fraction=feather_fraction),
                )
            except ValueError as error:
                # A status, not a raise: the C++ reports this through Result::error, and
                # the validations above set _status too.
                self._status = CudaStatus(1, str(error))
                return
            feather_weights_full = feathered.weights
            self._feather_radius_px = feathered.radius_px
            feather_roi_radius_px = feathered.requested_radius_px
            # A coverage hole inside an overlap moves a seam, so the ROI has to be derived from
            # the labels the field was actually built from, not from the ones handed in.
            if feathered.corrected_labels is not None:
                seam_index_for_roi = feathered.corrected_labels

        if not self._context.is_hard_seam and self._minimize_blend:
            boundary_bbox = seam_boundary_bbox(seam_index_for_roi)
            covered = False
            if boundary_bbox is not None:
                # The crossfade spans radius/2 either side of the seam, so the write ROI must
                # cover it.
                feather_pad = (
                    int(math.ceil(feather_roi_radius_px / 2.0)) + 1
                    if self._blend_mode == "alpha"
                    else 0
                )
                self._write_roi_canvas = expand_and_clamp(
                    boundary_bbox,
                    max(self._canvas_manager.overlap_padding(), feather_pad),
                    canvas_w,
                    canvas_h,
                )
                self._blend_roi_canvas = expand_and_clamp(
                    self._write_roi_canvas,
                    pyramid_margin(0 if self._blend_mode == "alpha" else num_levels),
                    canvas_w,
                    canvas_h,
                )
                self._blend_roi_canvas = align_and_clamp(
                    self._blend_roi_canvas,
                    pyramid_alignment(0 if self._blend_mode == "alpha" else num_levels),
                    canvas_w,
                    canvas_h,
                )
                covered = hard_baseline_covers_soft_owners_outside_write(
                    seam_index_padded,
                    [(int(p[0]), int(p[1])) for p in self._canvas_manager.canvas_positions()],
                    list(control_masks.img_col),
                    list(control_masks.img_row),
                    self._write_roi_canvas,
                )
            if boundary_bbox is not None and covered:
                seam_for_blend = seam_index_padded[
                    self._blend_roi_canvas.y : self._blend_roi_canvas.bottom,
                    self._blend_roi_canvas.x : self._blend_roi_canvas.right,
                ]
                if feather_weights_full is not None:
                    feather_weights_full = feather_weights_full[
                        self._blend_roi_canvas.y : self._blend_roi_canvas.bottom,
                        self._blend_roi_canvas.x : self._blend_roi_canvas.right,
                    ]
                self._remap_rois = [RemapRoiInfo() for _ in range(n)]
                for i, pos in enumerate(self._canvas_manager.canvas_positions()):
                    size = (
                        control_masks.img_col[i].shape[1],
                        control_masks.img_col[i].shape[0],
                    )
                    img_rect = Rect(pos[0], pos[1], size[0], size[1])
                    inter = img_rect.intersect(self._blend_roi_canvas)
                    self._remap_rois[i].offset_x = pos[0] - self._blend_roi_canvas.x
                    self._remap_rois[i].offset_y = pos[1] - self._blend_roi_canvas.y
                    if not inter.empty:
                        self._remap_rois[i].roi = Rect(
                            inter.x - pos[0],
                            inter.y - pos[1],
                            inter.width,
                            inter.height,
                        )
            else:
                # Either there is no seam at all, or the hard baseline cannot stand in for the
                # soft one outside the write ROI. Either way, fall back to the full canvas.
                self._write_roi_canvas = Rect(0, 0, 0, 0)
                self._blend_roi_canvas = Rect(0, 0, 0, 0)
                self._remap_rois = [RemapRoiInfo() for _ in range(n)]

        if not self._context.is_hard_seam:
            self._context.blend_mask = (
                feather_weights_full
                if feather_weights_full is not None
                else ControlMasksN.split_to_channels(seam_for_blend, n).astype(np.float32, copy=False)
            )

    @property
    def status(self) -> CudaStatus:
        return self._status

    @property
    def blend_mode(self) -> BlendMode:
        return self._blend_mode

    @property
    def feather_radius_px(self) -> float:
        """Widest crossfade any seam pixel got, in canvas pixels. Zero unless alpha mode."""
        return self._feather_radius_px

    def canvas_width(self) -> int:
        return self._canvas_manager.canvas_width()

    def canvas_height(self) -> int:
        return self._canvas_manager.canvas_height()

    def batch_size(self) -> int:
        return self._context.batch_size

    def _tensor(
        self, array: np.ndarray, device: torch.device, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        return self._cache.tensor(array, device=device, dtype=dtype)

    def _use_minimized_blend(self) -> bool:
        return (
            self._minimize_blend
            and not self._blend_roi_canvas.empty
            and not self._write_roi_canvas.empty
        )

    def _soft_scratch_for(
        self, device: torch.device, channels: int, minimized: bool
    ) -> _NImageSoftScratch:
        key = (str(device), channels, minimized)
        scratch = self._soft_scratch.get(key)
        if minimized:
            blend_h = self._blend_roi_canvas.height
            blend_w = self._blend_roi_canvas.width
        else:
            blend_h = self.canvas_height()
            blend_w = self.canvas_width()
        shape = (self._context.batch_size, blend_h, blend_w, channels)
        if scratch is None:
            scratch = _NImageSoftScratch(
                compute_buffers=[
                    torch.empty(shape, device=device, dtype=torch.float32)
                    for _ in range(self._context.n_images)
                ],
            )
            self._soft_scratch[key] = scratch
        return scratch

    def _graph_key(self, inputs: list[torch.Tensor]) -> tuple[object, ...]:
        return (
            str(inputs[0].device),
            tuple(tuple(inp.shape) for inp in inputs),
            tuple(inp.dtype for inp in inputs),
            self._use_minimized_blend(),
        )

    def _process_impl(
        self, batched_inputs: list[torch.Tensor], canvas: torch.Tensor
    ) -> torch.Tensor:
        device = batched_inputs[0].device
        channels = batched_inputs[0].shape[-1]
        canvas.zero_()
        seam_index = self._tensor(self._context.seam_index, device, torch.uint8)
        if self._context.is_hard_seam:
            for i in range(self._context.n_images):
                remap_to_canvas_with_dest_map(
                    batched_inputs[i],
                    canvas,
                    self._tensor(self._context.remap_x[i], device, torch.int32),
                    self._tensor(self._context.remap_y[i], device, torch.int32),
                    i,
                    seam_index,
                    self._canvas_manager.canvas_positions()[i][0],
                    self._canvas_manager.canvas_positions()[i][1],
                    backend=self._backend,
                )
            return canvas

        minimized = self._use_minimized_blend()
        if minimized:
            for i in range(self._context.n_images):
                remap_to_canvas_with_dest_map(
                    batched_inputs[i],
                    canvas,
                    self._tensor(self._context.remap_x[i], device, torch.int32),
                    self._tensor(self._context.remap_y[i], device, torch.int32),
                    i,
                    seam_index,
                    self._canvas_manager.canvas_positions()[i][0],
                    self._canvas_manager.canvas_positions()[i][1],
                    backend=self._backend,
                )
            scratch = self._soft_scratch_for(device, channels, True)
            for i, buf in enumerate(scratch.compute_buffers):
                buf.zero_()
                ri = self._remap_rois[i]
                remap_to_canvas(
                    batched_inputs[i],
                    buf,
                    self._tensor(self._context.remap_x[i], device, torch.int32),
                    self._tensor(self._context.remap_y[i], device, torch.int32),
                    ri.offset_x,
                    ri.offset_y,
                    roi=ri.roi,
                    fill_invalid_alpha=True,
                    backend=self._backend,
                )
            mask = self._tensor(self._context.blend_mask, device, torch.float32)
            blended = laplacian_blend_n(
                scratch.compute_buffers,
                mask,
                self._num_levels,
                backend=self._backend,
                workspace=scratch.blend_workspace,
            )
            copy_roi(
                blended,
                canvas,
                Rect(0, 0, self._write_roi_canvas.width, self._write_roi_canvas.height),
                self._write_roi_canvas.x - self._blend_roi_canvas.x,
                self._write_roi_canvas.y - self._blend_roi_canvas.y,
                self._write_roi_canvas.x,
                self._write_roi_canvas.y,
                backend=self._backend,
            )
            return cast_like(canvas, batched_inputs[0].dtype)

        scratch = self._soft_scratch_for(device, channels, False)
        for i, buf in enumerate(scratch.compute_buffers):
            buf.zero_()
            remap_to_canvas(
                batched_inputs[i],
                buf,
                self._tensor(self._context.remap_x[i], device, torch.int32),
                self._tensor(self._context.remap_y[i], device, torch.int32),
                self._canvas_manager.canvas_positions()[i][0],
                self._canvas_manager.canvas_positions()[i][1],
                fill_invalid_alpha=True,
                backend=self._backend,
            )
        mask = self._tensor(self._context.blend_mask, device, torch.float32)
        blended = laplacian_blend_n(
            scratch.compute_buffers,
            mask,
            self._num_levels,
            backend=self._backend,
            workspace=scratch.blend_workspace,
        )
        copy_roi(
            blended,
            canvas,
            Rect(0, 0, blended.shape[2], blended.shape[1]),
            0,
            0,
            0,
            0,
            backend=self._backend,
        )
        return cast_like(canvas, batched_inputs[0].dtype)

    def _capture_graph(self, inputs: list[torch.Tensor]) -> _GraphState:
        static_inputs = [torch.empty_like(inp) for inp in inputs]
        for static, inp in zip(static_inputs, inputs, strict=True):
            static.copy_(inp)
        static_output = torch.empty(
            (
                self._context.batch_size,
                self.canvas_height(),
                self.canvas_width(),
                inputs[0].shape[-1],
            ),
            device=inputs[0].device,
            dtype=inputs[0].dtype,
        )
        warmup_stream = torch.cuda.Stream(device=inputs[0].device)
        current_stream = torch.cuda.current_stream(device=inputs[0].device)
        with torch.cuda.stream(warmup_stream):
            self._process_impl(static_inputs, static_output)
            self._process_impl(static_inputs, static_output)
        current_stream.wait_stream(warmup_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._process_impl(static_inputs, static_output)
        return _GraphState(
            key=self._graph_key(inputs),
            graph=graph,
            static_inputs=static_inputs,
            static_output=static_output,
        )

    def _maybe_run_graph(
        self, inputs: list[torch.Tensor], canvas: torch.Tensor | None
    ) -> torch.Tensor | None:
        if (
            self._graph_disabled
            or not self._enable_cuda_graphs
            or not _cuda_graphs_supported(inputs[0].device)
        ):
            return None
        if any(inp.device != inputs[0].device for inp in inputs):
            return None
        key = self._graph_key(inputs)
        if self._graph_state is None or self._graph_state.key != key:
            try:
                self._graph_state = self._capture_graph(inputs)
            except Exception:
                self._graph_disabled = True
                self._graph_state = None
                return None
        assert self._graph_state is not None
        for static, inp in zip(self._graph_state.static_inputs, inputs, strict=True):
            static.copy_(inp)
        self._graph_state.graph.replay()
        if canvas is not None:
            canvas.copy_(self._graph_state.static_output)
            return canvas
        out = torch.empty_like(self._graph_state.static_output)
        out.copy_(self._graph_state.static_output)
        return out

    def process(
        self, inputs: Iterable[torch.Tensor], canvas: torch.Tensor | None = None
    ) -> torch.Tensor:
        _status_or_raise(self._status)
        batched_inputs = [ensure_batched(inp) for inp in inputs]
        if len(batched_inputs) != self._context.n_images:
            raise ValueError("inputs size != N")
        batch = batched_inputs[0].shape[0]
        channels = batched_inputs[0].shape[-1]
        if any(inp.shape[0] != batch for inp in batched_inputs):
            raise ValueError("Mismatched batch sizes")
        if batch != self._context.batch_size:
            raise ValueError("Input batch size does not match stitcher batch size")
        if any(inp.shape[-1] != channels for inp in batched_inputs):
            raise ValueError("Mismatched channel counts")

        if canvas is not None:
            canvas = ensure_batched(canvas)
        graph_out = self._maybe_run_graph(batched_inputs, canvas)
        if graph_out is not None:
            return graph_out

        device = batched_inputs[0].device
        if canvas is None:
            canvas = torch.empty(
                (batch, self.canvas_height(), self.canvas_width(), channels),
                device=device,
                dtype=batched_inputs[0].dtype,
            )
        return self._process_impl(batched_inputs, canvas)


def pyramid_margin(num_levels: int) -> int:
    if num_levels <= 0:
        return 0
    return 1 << min(num_levels, 30)


def pyramid_alignment(num_levels: int) -> int:
    if num_levels <= 1:
        return 1
    return 1 << min(num_levels - 1, 30)


def hard_baseline_covers_soft_owners_outside_write(
    seam_index: np.ndarray,
    positions: list[tuple[int, int]],
    remap_x: list[np.ndarray],
    remap_y: list[np.ndarray],
    write_roi_canvas: Rect,
) -> bool:
    """Mirrors blend_roi::hard_baseline_covers_soft_owners_outside_write.

    Outside the write ROI the minimized path keeps whatever the hard baseline produced, and that
    baseline is remapped from ``seam_index``. If the labelled camera has no data at such a pixel
    the baseline is transparent there while the full-canvas run is not, so minimizing has to be
    abandoned. Deliberately the labels handed in, not the corrected ones: correction fires exactly
    where the original owner has no data and some other camera does, so a corrected owner covers
    by construction and checking those would only ever fail where no camera covers at all.
    """
    if seam_index.dtype != np.uint8:
        raise ValueError("Expected indexed uint8 seam")
    height, width = seam_index.shape

    # One canvas bool rather than a gather over every outside pixel: this runs at construction on
    # every soft N-camera blend, and the gather cost 3.9 s and 2.4 GiB on a 14220x4938 canvas.
    # A pixel is answered for when its own owner has data there, so anything an out-of-range
    # owner, a mismatched remap pair, or a footprint that does not reach the pixel leaves unset
    # stays False and fails the check, matching the C++ early returns.
    answered = np.zeros((height, width), dtype=bool)
    for i, (pos_x, pos_y) in enumerate(positions):
        if remap_x[i].shape != remap_y[i].shape:
            continue
        rows_i, cols_i = remap_x[i].shape
        x0, x1 = max(0, pos_x), min(width, pos_x + cols_i)
        y0, y1 = max(0, pos_y), min(height, pos_y + rows_i)
        if x1 <= x0 or y1 <= y0:
            continue
        window_x = remap_x[i][y0 - pos_y : y1 - pos_y, x0 - pos_x : x1 - pos_x]
        window_y = remap_y[i][y0 - pos_y : y1 - pos_y, x0 - pos_x : x1 - pos_x]
        mapped = (window_x != UNMAPPED_POSITION_VALUE) & (window_y != UNMAPPED_POSITION_VALUE)
        answered[y0:y1, x0:x1] |= mapped & (seam_index[y0:y1, x0:x1] == i)

    # Inside the write ROI the soft field is written directly, so the baseline does not matter.
    # Both ends clamped: a negative bottom/right would otherwise be a negative numpy stop index
    # and mark a large region answered. expand_and_clamp never produces one, but this is a
    # module-level function.
    answered[
        max(0, write_roi_canvas.y) : max(0, min(height, write_roi_canvas.bottom)),
        max(0, write_roi_canvas.x) : max(0, min(width, write_roi_canvas.right)),
    ] = True
    return bool(answered.all())


def seam_boundary_bbox(seam_index: np.ndarray) -> Rect | None:
    if seam_index.dtype != np.uint8:
        raise ValueError("Expected indexed uint8 seam")
    h, w = seam_index.shape
    if w <= 1 or h <= 1:
        return None
    min_x = w
    min_y = h
    max_x = -1
    max_y = -1
    for y in range(h):
        row = seam_index[y]
        row_down = seam_index[y + 1] if y + 1 < h else None
        for x in range(w):
            value = row[x]
            if x + 1 < w and value != row[x + 1]:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + 1)
                max_y = max(max_y, y)
            if row_down is not None and value != row_down[x]:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y + 1)
    if max_x < 0 or max_y < 0:
        return None
    return Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)


def expand_and_clamp(rect: Rect, pad: int, max_w: int, max_h: int) -> Rect:
    if rect.empty:
        return Rect()
    x0 = max(0, rect.x - pad)
    y0 = max(0, rect.y - pad)
    x1 = min(max_w, rect.right + pad)
    y1 = min(max_h, rect.bottom + pad)
    return Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def align_and_clamp(rect: Rect, align: int, max_w: int, max_h: int) -> Rect:
    if rect.empty or align <= 1:
        return rect
    x0 = max(0, (rect.x // align) * align)
    y0 = max(0, (rect.y // align) * align)
    x1 = min(max_w, ((rect.right + align - 1) // align) * align)
    y1 = min(max_h, ((rect.bottom + align - 1) // align) * align)
    return Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))
