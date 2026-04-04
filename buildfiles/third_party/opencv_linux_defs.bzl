def _candidate_prefixes(version):
    # Search order: prefer /usr includes first to avoid stale local installs, then /usr/local.
    return [
        "include/{}".format(version),
        "include/x86_64-linux-gnu/{}".format(version),
        "include/aarch64-linux-gnu/{}".format(version),
        "local/include/{}".format(version),
        "local/include/x86_64-linux-gnu/{}".format(version),
        "local/include/aarch64-linux-gnu/{}".format(version),
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

_CPU_OPENCV_LIBS = ["core", "imgproc", "imgcodecs", "highgui", "video", "videoio"]
_CUDA_OPENCV_LIBS = ["cudacodec", "cudaimgproc", "cudawarping"]


def _dedupe(values):
    seen = {}
    out = []
    for value in values:
        if value in seen:
            continue
        seen[value] = True
        out.append(value)
    return out


def _collect_soname_suffixes(lib_dir, lib):
    prefix = "{}/libopencv_{}.so.".format(lib_dir, lib)
    matches = native.glob(["{}/libopencv_{}.so.*".format(lib_dir, lib)], allow_empty = True)
    suffixes = []
    for path in matches:
        if path.startswith(prefix):
            suffixes.append(path[len(prefix):])
    return _dedupe(suffixes)


def _intersection(lhs, rhs):
    rhs_set = {}
    for item in rhs:
        rhs_set[item] = True
    out = []
    for item in lhs:
        if item in rhs_set:
            out.append(item)
    return out


def _find_common_suffixes(lib_dir, libs):
    common = None
    for lib in libs:
        suffixes = _collect_soname_suffixes(lib_dir, lib)
        if not suffixes:
            return []
        if common == None:
            common = suffixes
        else:
            common = _intersection(common, suffixes)
        if not common:
            return []
    return common


def _ensure_required_libs(lib_dir):
    missing = []
    for lib in _REQUIRED_OPENCV_LIBS:
        if not native.glob(["{}/libopencv_{}.so*".format(lib_dir, lib)], allow_empty = True):
            missing.append(lib)

    if missing:
        fail("OpenCV is missing required libs in '{}': {}".format(lib_dir, ", ".join(missing)))

    # Keep CPU-side OpenCV libraries ABI-consistent.
    cpu_common = _find_common_suffixes(lib_dir, _CPU_OPENCV_LIBS)
    if not cpu_common:
        fail("OpenCV CPU libs are version-inconsistent in '{}': {}".format(lib_dir, ", ".join(_CPU_OPENCV_LIBS)))

    # Keep CUDA-side OpenCV libraries ABI-consistent.
    cuda_common = _find_common_suffixes(lib_dir, _CUDA_OPENCV_LIBS)
    if not cuda_common:
        fail("OpenCV CUDA libs are version-inconsistent in '{}': {}".format(lib_dir, ", ".join(_CUDA_OPENCV_LIBS)))


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
        "-l:libopencv_cudacodec.so",
        "-l:libopencv_cudaimgproc.so",
        "-l:libopencv_cudawarping.so",
    ]

    native.cc_library(
        name = name,
        hdrs = detected.headers,
        includes = detected.includes,
        linkopts = linkopts,
        visibility = visibility if visibility else ["//visibility:public"],
    )
