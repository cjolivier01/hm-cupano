# Bazel is only available for amd64 and arm64.
# load("//:buildfiles/third_party/opencv.bzl", "get_opencv_version")

config_setting(
    name = "aarch64-linux-gnu",
    constraint_values = ["@platforms//cpu:aarch64"],
)

config_setting(
    name = "x86_64-linux-gnu",
    constraint_values = ["@platforms//cpu:x86_64"],
)

# OPENCV_VERSION = "opencv5"
OPENCV_VERSION = "opencv4"

cc_library(
    name = "opencv",
    hdrs = glob([
        # "opencv5/opencv2/**/*.h*",
    ]) + select({
        ":aarch64-linux-gnu": [
            # "aarch64-linux-gnu/opencv5/opencv2/cvconfig.h"
        ] + glob([
            "opencv4/opencv2/**/*.h*",
        ]),
        ":x86_64-linux-gnu": [
            # "x86_64-linux-gnu/opencv5/opencv2/cvconfig.h",
        ] + glob([
            OPENCV_VERSION + "/opencv2/**/*.h*",
        ]),
        "//conditions:default": [],
    }),
    includes = [
    ] + select({
        ":aarch64-linux-gnu": [
            "opencv4",
            "x86_64-linux-gnu/opencv4/opencv2",
            # "aarch64-linux-gnu/opencv5",
        ],
        ":x86_64-linux-gnu": [
            OPENCV_VERSION,
            "x86_64-linux-gnu/" + OPENCV_VERSION,
            "x86_64-linux-gnu/" + OPENCV_VERSION + "/opencv2",
        ],
        "//conditions:default": [],
    }),
    linkopts = [
        #"-L/usr/local/lib",
        "-l:libopencv_core.so",
        #"-l:libopencv_calib3d.so",
        #"-l:libopencv_features2d.so",
        "-l:libopencv_imgproc.so",
        "-l:libopencv_imgcodecs.so",
        "-l:libopencv_highgui.so",
        "-l:libopencv_video.so",
        "-l:libopencv_videoio.so",
        # Add as neeeded
        # '-lopencv_stitching', '-lopencv_alphamat', '-lopencv_aruco', '-lopencv_barcode', '-lopencv_bgsegm',
        # '-lopencv_bioinspired', '-lopencv_ccalib', '-lopencv_cvv', '-lopencv_dnn_objdetect',
        # '-lopencv_dnn_superres', '-lopencv_dpm', '-lopencv_face', '-lopencv_freetype',
        # '-lopencv_fuzzy', '-lopencv_hdf', '-lopencv_hfs', '-lopencv_img_hash',
        # '-lopencv_intensity_transform', '-lopencv_line_descriptor', '-lopencv_mcc',
        # '-lopencv_quality', '-lopencv_rapid', '-lopencv_reg', '-lopencv_rgbd', '-lopencv_saliency',
        # '-lopencv_shape', '-lopencv_stereo', '-lopencv_structured_light',
        # '-lopencv_phase_unwrapping', '-lopencv_superres', '-lopencv_optflow',
        # '-lopencv_surface_matching', '-lopencv_tracking', '-lopencv_highgui',
        # '-lopencv_datasets', '-lopencv_text', '-lopencv_plot', '-lopencv_ml',
        # '-lopencv_videostab', '-lopencv_videoio', '-lopencv_viz', '-lopencv_wechat_qrcode',
        # '-lopencv_ximgproc', '-lopencv_video', '-lopencv_xobjdetect', '-lopencv_objdetect',
        # '-lopencv_calib3d', '-lopencv_imgcodecs',
        # '-lopencv_features2d', '-lopencv_dnn', '-lopencv_flann', '-lopencv_xphoto',
        # '-lopencv_photo', '-lopencv_imgproc', '-lopencv_core', '-lopencv_cudawarping',
        # '-lopencv_cudafilters', '-lopencv_cudaimgproc', '-lopencv_cudafeatures2d', '-lopencv_cudaoptflow', '-lopencv_cudabgsegm', '-lopencv_cudastereo', '-lopencv_cudalegacy', '-lopencv_cudaobjdetect', '-lopencv_cudacodec', '-lopencv_cudafeatures2d', '-lopencv_cudacodec', '-lopencv_cudalegacy', '-lopencv_cudaobjdetect', '-lopencv_cudastereo', '-lopencv_cudabgsegm', '-lopencv_cudaoptflow', '-lopencv_cudafilters', '-lopencv_cudaimgproc', '-lopencv_cudawarping ', '-lopencv_cudev', '-lopencv_cudnn', '-lopencv_cudnn_ops', '-lopencv_cudnn_solvers
    ],
    visibility = ["//visibility:public"],
)
