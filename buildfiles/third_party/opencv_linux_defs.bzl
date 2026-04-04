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
    repo_root = "/usr"  # mirrored into @opencv_linux by opencv_linux_repository

    # Prefer OpenCV 4 over 5 unless the machine only has 5 installed.
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
                    prefix = prefix,
                )

    fail("OpenCV headers not found (looked for opencv4/opencv2 or opencv5/opencv2)")


_REQUIRED_OPENCV_CPU_LIBS = ["core", "imgproc", "imgcodecs", "highgui", "video", "videoio"]
_REQUIRED_OPENCV_CUDA_LIBS = ["cudacodec", "cudaimgproc", "cudawarping"]
_REQUIRED_CUDA_HEADERS = [
    "cudaarithm.hpp",
    "cudacodec.hpp",
    "cudaimgproc.hpp",
    "cudawarping.hpp",
]

_CUDA_NVCC_GLOBS = [
    "bin/nvcc",
    "local/cuda/bin/nvcc",
    "local/cuda-*/bin/nvcc",
    "lib/cuda/bin/nvcc",
]

_CUDA_CUDART_GLOBS = [
    "lib/x86_64-linux-gnu/libcudart.so*",
    "lib/aarch64-linux-gnu/libcudart.so*",
    "lib/libcudart.so*",
    "lib64/libcudart.so*",
    "local/cuda/lib64/libcudart.so*",
    "local/cuda-*/lib64/libcudart.so*",
]

_ROCM_TOOL_GLOBS = [
    "bin/rocm-smi",
    "bin/hipcc",
    "local/rocm/bin/rocm-smi",
    "local/rocm/bin/hipcc",
    "local/rocm-*/bin/rocm-smi",
    "local/rocm-*/bin/hipcc",
    "opt/rocm/bin/rocm-smi",
    "opt/rocm/bin/hipcc",
    "opt/rocm-*/bin/rocm-smi",
    "opt/rocm-*/bin/hipcc",
]


def _dedupe(values):
    seen = {}
    out = []
    for value in values:
        if value in seen:
            continue
        seen[value] = True
        out.append(value)
    return out


def _has_any_glob(patterns):
    for pattern in patterns:
        if native.glob([pattern], allow_empty = True):
            return True
    return False


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


def _normalize_backend(raw_backend):
    backend = (raw_backend or "auto").strip().lower()
    if backend == "":
        backend = "auto"
    if backend not in ["auto", "cuda", "rocm"]:
        fail("Invalid OpenCV backend '{}'; expected one of: auto, cuda, rocm".format(backend))
    return backend


def _detect_backend_availability():
    return struct(
        cuda_available = _has_any_glob(_CUDA_NVCC_GLOBS) and _has_any_glob(_CUDA_CUDART_GLOBS),
        rocm_available = _has_any_glob(_ROCM_TOOL_GLOBS),
    )


def _resolve_backend(requested_backend, availability):
    backend = _normalize_backend(requested_backend)

    if backend == "cuda":
        if not availability.cuda_available:
            fail("OpenCV backend forced to CUDA, but CUDA toolkit markers (nvcc/libcudart) were not detected under /usr")
        return "cuda"

    if backend == "rocm":
        if not availability.rocm_available:
            fail("OpenCV backend forced to ROCm, but ROCm toolkit markers (hipcc/rocm-smi) were not detected under /usr")
        return "rocm"

    if availability.cuda_available:
        return "cuda"
    if availability.rocm_available:
        return "rocm"
    fail("OpenCV build requires either CUDA or ROCm toolkit markers under /usr, but neither was detected")


def _ensure_required_headers(prefix):
    missing = []
    for header in _REQUIRED_CUDA_HEADERS:
        header_path = "{}/opencv2/{}".format(prefix, header)
        if not native.glob([header_path], allow_empty = True):
            missing.append(header)

    if missing:
        fail("OpenCV CUDA headers are missing in '{}': {}".format(prefix, ", ".join(missing)))


def _ensure_required_libs(lib_dir, libs, kind):
    missing = []
    for lib in libs:
        if not native.glob(["{}/libopencv_{}.so*".format(lib_dir, lib)], allow_empty = True):
            missing.append(lib)

    if missing:
        fail("OpenCV {} libs are missing in '{}': {}".format(kind, lib_dir, ", ".join(missing)))


def opencv_library(name, backend = "auto", visibility = None):
    availability = _detect_backend_availability()
    selected_backend = _resolve_backend(backend, availability)

    detected = _detect_opencv()
    if not detected.lib_dir_abs or not detected.lib_dir:
        fail("OpenCV headers were found, but no corresponding OpenCV lib directory was detected")

    # Always require CPU-side OpenCV.
    _ensure_required_libs(detected.lib_dir, _REQUIRED_OPENCV_CPU_LIBS, "CPU")

    # Keep CPU-side OpenCV libraries ABI-consistent.
    cpu_common = _find_common_suffixes(detected.lib_dir, _REQUIRED_OPENCV_CPU_LIBS)
    if not cpu_common:
        fail("OpenCV CPU libs are version-inconsistent in '{}': {}".format(detected.lib_dir, ", ".join(_REQUIRED_OPENCV_CPU_LIBS)))

    cuda_linkopts = []
    if selected_backend == "cuda":
        _ensure_required_headers(detected.prefix)
        _ensure_required_libs(detected.lib_dir, _REQUIRED_OPENCV_CUDA_LIBS, "CUDA")

        cuda_common = _find_common_suffixes(detected.lib_dir, _REQUIRED_OPENCV_CUDA_LIBS)
        if not cuda_common:
            fail("OpenCV CUDA libs are version-inconsistent in '{}': {}".format(detected.lib_dir, ", ".join(_REQUIRED_OPENCV_CUDA_LIBS)))

        cuda_linkopts = [
            "-l:libopencv_cudacodec.so",
            "-l:libopencv_cudaimgproc.so",
            "-l:libopencv_cudawarping.so",
        ]

    linkopts = [
        "-L{}".format(detected.lib_dir_abs),
        "-Wl,-rpath,{}".format(detected.lib_dir_abs),
        "-l:libopencv_core.so",
        "-l:libopencv_imgproc.so",
        "-l:libopencv_imgcodecs.so",
        "-l:libopencv_highgui.so",
        "-l:libopencv_video.so",
        "-l:libopencv_videoio.so",
    ] + cuda_linkopts

    native.cc_library(
        name = name,
        hdrs = detected.headers,
        includes = detected.includes,
        linkopts = linkopts,
        visibility = visibility if visibility else ["//visibility:public"],
    )
