# Packed RGB10 CUDA remapping

`CudaStitchPano`, `CudaStitchPano3`, and `CudaStitchPanoN` accept packed `CudaMat<Rgb10A2>` camera inputs with `half4` pipeline and compute types. Wrap existing pitched device surfaces with `SurfaceInfo` to avoid a full source-frame allocation. Batch images are contiguous at `pitch * height` bytes per image. Inputs must remain valid until work on the supplied stream completes; serialize calls on each context.

Remapping loads each packed source sample directly into FP16 compute scratch. R/G/B occupy bits 0–9, 10–19, and 20–29. Conversion preserves the staged path's float multiply by `255.0f / 1023.0f`, followed by round-to-nearest half conversion. Packed alpha is ignored: valid source samples receive alpha 255. Invalid mappings retain the selected remapper's existing behavior, including the distinct reference and fused three-camera routes. Existing half4 input calls remain available for unpack-then-remap callers.

For two 7680 × 4320 sources, eliminating the two half4 staging images saves exactly 530,841,600 bytes (506.25 MiB). Canvas resolution, remap coordinates, blend levels, and downstream grading are unchanged. No host readback is introduced.

## Validation

`rgb10Remap_test` compares packed and staged FP16 images byte-for-byte across pitched, batched inputs, odd sizes, offsets, invalid coordinates, hard/soft seams, full/minimized blending, repeated frames, and 2/3/N cameras. Three-camera parity covers both remap routes. Run with Compute Sanitizer memcheck to check device accesses.
