# Low-memory CUDA stitching

`CudaStitchPano`, `CudaStitchPano3`, and `CudaStitchPanoN` accept packed `CudaMat<Rgb10A2>` camera inputs with `half4` pipeline and compute types. Wrap existing pitched device surfaces with `SurfaceInfo` to avoid a full source-frame allocation. Batch images are contiguous at `pitch * height` bytes per image. Inputs must remain valid until work on the supplied stream completes; serialize calls on each context.

Remapping loads each packed source sample directly into FP16 compute scratch. R/G/B occupy bits 0–9, 10–19, and 20–29. Conversion preserves the staged path's float multiply by `255.0f / 1023.0f`, followed by round-to-nearest half conversion. Packed alpha is ignored: valid source samples receive alpha 255. Invalid mappings retain the selected remapper's existing behavior, including the distinct reference and fused three-camera routes. Existing half4 input calls remain available for unpack-then-remap callers.

For two 7680 × 4320 sources, eliminating the two half4 staging images saves exactly 530,841,600 bytes (506.25 MiB). Canvas resolution, remap coordinates, blend levels, and downstream grading are unchanged. No host readback is introduced.

## Compact workspace

The trailing constructor option `compact_workspace` defaults to false. When enabled, writable remap scratch is reused for Gaussian, Laplacian, blended, and reconstruction values after their last use. Arithmetic, precision, alpha, and pyramid levels are preserved. Intermediate pyramid display/dumps are unavailable because earlier values have been overwritten. The separate optimized low-level three-image blend API rejects compact contexts; panorama remap fusion uses the compatible blender.

A null output canvas requests managed output. Full-canvas soft blending with matching pipeline/compute types can return a non-owning view of compact scratch. It is valid until the next process call or context destruction. Other cases allocate owned output. `CudaMat::owns_memory()` distinguishes the two: retain and reuse owned output to avoid per-frame allocation, and consume borrowed output on the producing stream before reuse. Existing callers can continue supplying owned output.

Low-level blend contexts expose `reuseInputs`. Their input images must be writable and refilled before each call. Two-/three-image contexts retain their established input-pointer lifetime contract; N-image contexts rebind inputs on each call. Context calls must be serialized. Outputs may alias the supported input scratch or use separate allocations.

## Validation

`rgb10Remap_test` compares packed and staged FP16 images byte-for-byte across pitched, batched inputs, all channel codes, packed alpha values, odd sizes, offsets, invalid coordinates, hard/soft seams, full/minimized blending, compact/standard storage, repeated frames, and 2/3/N cameras. Three-camera parity covers both remap routes. Compact blend tests also compare float and half precision and alternating output allocations. Run with Compute Sanitizer memcheck to check device accesses.
