Fast panorama stitching of two images.

Used for stitching two high-resolution (5K) GoPro videos into a single (roughly) 8500x3300 panorama video.
This repo simply delivers the high-performance stitching code.  Running through each frame of videos and stitching them is not currently part of this repo, since this repo is focused mainly on the stitching code wrt a single frame.

```
Speed on RTX4090 with 6 laplacian levels:     340 fps
Speed on Jetson with 6 levels:                  8 fps
Speed on Jetson with 0 levels:                 20 fps
```

For the Jetson, scaling down the pano config to around 6000 pixels width brings it up to 50 fps

Procedure:

Install Hugin and Enblend:
```
sudo apt-get install hugin hugin-tools enblend
```

Maybe install bazelisk
```
./scripts/install_bazelisk.sh
```

Build the code
```
bazelisk build //...
```

Put the left and right videos into some directory
Run the configuration to configure the stitching. 
It will write some project and mapping files (along with keypoint matching illustrations to) the directory where the videos reside.
```
python scripts/create_control_points.py <path to left video> <path to right video>
```
 
Run the stitching test
```
./bazel-bin/tests/test_cuda_blend --show --perf --output=myframe.png --directory=<directory where the video was/project files were saved>
```

Left frame:
![alt text](./assets/left.png)

Right frame:
![alt text](./assets/right.png)

Key points (SuperPoint)
![alt text](./assets/keypoints.png)

Matches (LightGlue)
![alt text](./assets/matches.png)

Stitched Panorama (CUDA Kernels, hard seam or laplacian blending + color correction)
![alt text](./assets/s.png)

## N-Image Stitching (Arbitrary N)

You can stitch 2–8 images using the N-image path.

Build:
```
bazelisk build //tests:test_cuda_blend_n
```

Run (soft seam example with 4 inputs):
```
./bazel-bin/tests/test_cuda_blend_n \
  --num-images=4 --levels=6 \
  --directory=<data_dir> \
  --output=out.png --show
```

Run (hard seam, no pyramid):
```
./bazel-bin/tests/test_cuda_blend_n \
  --num-images=3 --levels=0 \
  --directory=<data_dir> \
  --output=out.png
```

Expected files under `<data_dir>` for N images (0..N-1):
- Input frames: `image0.png`, `image1.png`, ..., `image{N-1}.png`
- Remaps (CV_16U): `mapping_000i_x.tif`, `mapping_000i_y.tif` for each i
- Positions (TIFF tags): `mapping_000i.tif` (used to derive canvas placement)
- Seam mask (indexed 8-bit paletted): `seam_file.png` with classes `[0..N-1]` (one class per image)
