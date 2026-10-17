# Low-memory CUDA stitching

`CudaStitchPano`, `CudaStitchPano3`, and `CudaStitchPanoN` accept packed `CudaMat<Rgb10A2>` camera inputs with `half4` pipeline and compute types. Wrap existing pitched device surfaces with `SurfaceInfo` to avoid a full source-frame allocation. Batch images are contiguous at `pitch * height` bytes per image. Inputs must remain valid until work on the supplied stream completes; serialize calls on each context.

Remapping loads each packed source sample directly into FP16 compute scratch. R/G/B occupy bits 0–9, 10–19, and 20–29. Conversion preserves the staged path's float multiply by `255.0f / 1023.0f`, followed by round-to-nearest half conversion. Packed alpha is ignored: valid source samples receive alpha 255. Invalid mappings retain the selected remapper's existing behavior, including the distinct reference and fused three-camera routes. Existing half4 input calls remain available for unpack-then-remap callers.

For two 7680 × 4320 sources, eliminating the two half4 staging images saves exactly 530,841,600 bytes (506.25 MiB). Canvas resolution, remap coordinates, blend levels, and downstream grading are unchanged. No host readback is introduced.

## Compact workspace

The trailing constructor option `compact_workspace` defaults to false. When enabled, writable remap scratch is reused for Gaussian, Laplacian, blended, and reconstruction values after their last use. Arithmetic, precision, alpha, and pyramid levels are preserved. Intermediate pyramid display/dumps are unavailable because earlier values have been overwritten. The separate optimized low-level three-image blend API rejects compact contexts; panorama remap fusion uses the compatible blender.

A null output canvas requests managed output. Full-canvas soft blending with matching pipeline/compute types returns a non-owning view when compact mode is enabled or the trailing `process(..., borrow_workspace_output)` argument is true. The argument defaults to false, so standard-workspace callers retain owned output unless they opt in. For three cameras, it follows the existing `fused` argument and works with either remap route. Borrowing the standard workspace avoids a full-panorama allocation and device copy without changing the blend intermediates.

The view is valid until the next process call or context destruction. Consume it on the producing stream before reuse; retaining its `unique_ptr` does not retain the scratch storage. Hard seams, minimized blending, and different pipeline/compute types return owned output. An explicitly supplied output always takes precedence. `CudaMat::owns_memory()` distinguishes the two: retain and reuse owned output to avoid per-frame allocation, and discard borrowed wrappers before the stitcher is reused or destroyed.

`CudaMat` constructors accept a trailing `std::optional<cudaStream_t>`. Omitting it
or passing `std::nullopt` preserves synchronous `cudaMalloc`/`cudaFree` behavior.
Supplying a stream uses `cudaMallocAsync`/`cudaFreeAsync`; an explicitly supplied
default stream (`cudaStream_t{}` or `nullptr`) also selects the async APIs.
For example, `CudaMat<half4>(batch, width, height, 1, stream)` allocates on `stream`.
Keep that stream alive until the matrix is destroyed, and order any access from
other streams after allocation and before release. CPU-image constructors and
`download()` wait for their host copies to finish. `upload()` without an override
uses the matrix's stream; for an async matrix, keep the source image alive until
the upload completes. Borrowed matrices never free their pointers.

All three panorama paths pass their processing stream when allocating managed
output (and when wrapping borrowed output for download). Destroy owned managed
outputs before destroying that stream. Blend pyramids and pointer tables allocate
on their processing stream. Blend contexts record completion events and order
their async cleanup after the last call, so contexts can outlive caller streams.
Successive calls on a context must use the same stream or establish an explicit
GPU dependency before switching streams; a host mutex alone does not order work
already queued on different streams.
The low-level wrappers returning CPU images still wait for their output copies;
the functions returning device output remain asynchronous.

Low-level blend contexts expose `reuseInputs`. Their input images must be writable and refilled before each call. Two-/three-image contexts retain their established input-pointer lifetime contract; N-image contexts rebind inputs on each call. Context calls must be serialized. Outputs may alias the supported input scratch or use separate allocations.

## Validation

`rgb10Remap_test` compares packed and staged FP16 images byte-for-byte across pitched, batched inputs, all channel codes, packed alpha values, odd sizes, offsets, invalid coordinates, hard/soft seams, full/minimized blending, compact/standard storage, repeated frames, and 2/3/N cameras. Three-camera parity covers both remap routes. Compact blend tests also compare float and half precision and alternating output allocations. Run with Compute Sanitizer memcheck to check device accesses.

Panorama output tests compare standard borrowed output with owned output byte for byte across 2/3/N cameras, float4/half4, hard/soft seams, repeated frames and both three-camera remap routes. They also check minimized fallback ownership and preservation of explicitly supplied canvases.
