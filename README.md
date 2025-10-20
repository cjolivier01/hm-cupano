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

## Working Examples

The commands below are tested on the provided assets using the Python from the `ubuntu` conda env and the built Bazel binaries.

### 1) Two-Image Stitch (assets/left.png + assets/right.png)

- Generate control points and Hugin mappings (writes into `assets/`):
```
python3 scripts/create_control_points.py --left assets/left.png --right assets/right.png --max-control-points 200
```
- Run the CUDA stitcher (0 levels = hard seam, fastest):
```
./bazel-bin/tests/test_cuda_blend --levels=6 --adjust=1 --directory=assets --output=assets/pano_left_right.png
```
- View the result:
```
viewnior assets/pano_left_right.png &
```
- Result preview (clickable path): `assets/pano_left_right.png`

Example outputs and diagnostics written to `assets/`:
- Matches: `assets/matches.png`
- Keypoints: `assets/keypoints.png`
- Seam file (from enblend): `assets/seam_file.png`
- Mappings: `assets/mapping_000{0,1}_[xy].tif`

### 2) Three-Image Stitch (assets/weir_1,2,3 → assets/three/)

Prepare a small working folder and convert the inputs to the naming the demos expect (`image0.png`, `image1.png`, `image2.png`):
```
mkdir -p assets/three
ffmpeg -y -loglevel error -i assets/weir_1.jpg assets/three/image0.png
ffmpeg -y -loglevel error -i assets/weir_2.jpg assets/three/image1.png
ffmpeg -y -loglevel error -i assets/weir_3.jpg assets/three/image2.png
```

Create the Hugin project, find control points, optimize, and export remap files (`mapping_000i_[xy].tif`) and warped layers (`mapping_000i.tif`):
```
cd assets/three
pto_gen -p 0 -o three.pto -f 65 image0.png image1.png image2.png
cpfind --multirow -o three_cp.pto three.pto
autooptimiser -a -m -l -s -o three_opt.pto three_cp.pto
nona -m TIFF_m -z NONE --bigtiff -c -o mapping_ three_opt.pto
```

Generate the seam mask with multiblend (recommended for 3+). This uses the Bazel external `@multiblend`:
```
bazelisk run @multiblend//:multiblend -- --save-seams=seam_file.png -o panorama.tif mapping_????.tif
```
Alternatively, you can use enblend (also available as Bazel external `@enblend`) and convert to a 3‑class paletted mask automatically:
```
python3 scripts/generate_seam.py --directory=$(pwd) --num-images=3 --seam enblend
```

Stitch using the 3-image path (6 levels + exposure adjust) and view it:
```
cd -
./bazel-bin/tests/test_cuda_blend3 --levels=6 --adjust=1 --directory=assets/three --output=assets/three/pano_three_3way.png
viewnior assets/three/pano_three_3way.png &
```

Alternatively, the generic N-image path works with the same folder (num-images=3):
```
./bazel-bin/tests/test_cuda_blend_n --levels=6 --adjust=1 --num-images=3 --directory=assets/three --output=assets/three/pano_three_n.png
```

Input previews:
- `assets/three/image0.png`
- `assets/three/image1.png`
- `assets/three/image2.png`

Output previews:
- `assets/three/pano_three_3way.png`
- `assets/three/pano_three_n.png`

## N-Image Stitching (Arbitrary N)

You can stitch 2–8 images using the N-image path.

Build:
```
bazelisk build //tests:test_cuda_blend_n
```

Run (soft seam example with 4 inputs):
```
./bazel-bin/tests/test_cuda_blend_n \
  --num-images=4 --levels=6 --adjust=1 \
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
