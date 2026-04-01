def _candidate_prefixes(version):
    # Search order: prefer /usr/local includes, then /usr includes, then multiarch variants.
    return [
        "local/include/{}".format(version),
        "include/{}".format(version),
        "local/include/x86_64-linux-gnu/{}".format(version),
        "include/x86_64-linux-gnu/{}".format(version),
        "local/include/aarch64-linux-gnu/{}".format(version),
        "include/aarch64-linux-gnu/{}".format(version),
    ]


def _candidate_lib_dirs(prefix_is_local):
    # Prefer matching /usr vs /usr/local based on header selection.
    if prefix_is_local:
        return [
            "local/lib",
            "local/lib64",
            "lib",
            "lib64",
            "lib/x86_64-linux-gnu",
            "lib/aarch64-linux-gnu",
            "local/lib/x86_64-linux-gnu",
            "local/lib/aarch64-linux-gnu",
        ]
    else:
        return [
            "lib/x86_64-linux-gnu",
            "lib/aarch64-linux-gnu",
            "lib",
            "lib64",
            "local/lib",
            "local/lib64",
        ]


def _detect_opencv():
    """Detects available OpenCV headers and returns a struct with version, headers, includes."""
    repo_root = "/usr"  # new_local_repository path in WORKSPACE
    # Prefer OpenCV 4 over 5 unless the machine only has 5 installed.
    # The repo and local soname symlinks are typically aligned to 4.x today.
    for version in ["opencv4", "opencv5"]:
        for prefix in _candidate_prefixes(version):
            headers = native.glob([prefix + "/opencv2/**/*.h*"], allow_empty = True)
            if headers:
                includes = [prefix, prefix + "/opencv2"]
                prefix_is_local = prefix.startswith("local/")
                lib_dir = None
                lib_dir_abs = None
                for lib_candidate in _candidate_lib_dirs(prefix_is_local):
                    if native.glob([lib_candidate + "/libopencv_core.so*"], allow_empty = True):
                        lib_dir = lib_candidate
                        lib_dir_abs = repo_root + "/" + lib_candidate
                        break

                return struct(
                    version = version,
                    headers = headers,
                    includes = includes,
                    lib_dir = lib_dir,
                    lib_dir_abs = lib_dir_abs,
                )

    fail("OpenCV headers not found (looked for opencv5/opencv2 or opencv4/opencv2)")


_REQUIRED_OPENCV_LIBS = [
    "core",
    "imgproc",
    "imgcodecs",
    "highgui",
    "video",
    "videoio",
    "cudacodec",
    "cudaimgproc",
    "cudawarping",
]


def _ensure_required_libs(lib_dir):
    missing = []
    for lib in _REQUIRED_OPENCV_LIBS:
        if not native.glob(["{}/libopencv_{}.so*".format(lib_dir, lib)], allow_empty = True):
            missing.append(lib)

    if missing:
        fail("OpenCV is missing required libs in '{}': {}".format(lib_dir, ", ".join(missing)))


def opencv_library(name, visibility = None):
    detected = _detect_opencv()
    if not detected.lib_dir_abs or not detected.lib_dir:
        fail("OpenCV headers were found, but no corresponding OpenCV lib directory was detected")

    # Enforce that CUDA-enabled OpenCV libs are available at analysis time.
    _ensure_required_libs(detected.lib_dir)

    linkopts = [
        "-L{}".format(detected.lib_dir_abs),
        "-Wl,-rpath,{}".format(detected.lib_dir_abs),
        "-l:libopencv_core.so",
        "-l:libopencv_imgproc.so",
        "-l:libopencv_imgcodecs.so",
        "-l:libopencv_highgui.so",
        "-l:libopencv_video.so",
        "-l:libopencv_videoio.so",
    ]

    native.cc_library(
        name = name,
        hdrs = detected.headers,
        includes = detected.includes,
        linkopts = linkopts,
        visibility = visibility if visibility else ["//visibility:public"],
    )
