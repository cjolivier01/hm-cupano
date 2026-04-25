"""Repository rule that builds FFmpeg with a required GPU backend (CUDA or ROCm)."""

_FFMPEG_BASE_CONFIGURE_FLAGS = [
    "--disable-doc",
    "--enable-decoder=aac",
    "--enable-decoder=h264",
    "--enable-decoder=rawvideo",
    "--enable-indev=lavfi",
    "--enable-encoder=libx264",
    "--enable-demuxer=mov",
    "--enable-muxer=mp4",
    "--enable-filter=scale",
    "--enable-filter=testsrc2",
    "--enable-protocol=file",
    "--enable-protocol=https",
    "--enable-gnutls",
    "--disable-shared",
    "--enable-static",
    "--enable-gpl",
    "--enable-nonfree",
    "--enable-libx264",
    "--enable-pic",
    "--enable-rpath",
    "--enable-libfontconfig",
    "--enable-libfreetype",
    "--enable-libfribidi",
    "--enable-libharfbuzz",
    "--enable-libvidstab",
    "--enable-vaapi",
]

_FFMPEG_LIBS = [
    "avdevice",
    "avfilter",
    "avformat",
    "avcodec",
    "swscale",
    "swresample",
    "postproc",
    "avutil",
]

_FFMPEG_CUDA_CONFIGURE_FLAGS = [
    "--enable-decoder=h264_cuvid",
    "--enable-decoder=hevc_cuvid",
    "--enable-decoder=av1_cuvid",
    "--enable-encoder=h264_nvenc",
    "--enable-encoder=hevc_nvenc",
    "--enable-encoder=av1_nvenc",
    "--enable-cuda-nvcc",
    "--disable-libnpp",
    "--enable-nvenc",
    "--enable-cuvid",
    "--enable-nvdec",
]

_FFMPEG_ROCM_CONFIGURE_FLAGS = [
    "--enable-amf",
    "--enable-encoder=hevc_amf",
    "--enable-decoder=hevc_amf",
]


def _shell_quote(value):
    value_str = str(value)
    return "'{}'".format(value_str.replace("'", "'\\''"))


def _join_shell_args(args):
    return " ".join([_shell_quote(a) for a in args])


def _starlark_string_list(values, indent = "        "):
    if not values:
        return "[]"
    return "[\n" + "".join([
        '{}"{}",\n'.format(indent, value.replace("\\", "\\\\").replace('"', '\\"'))
        for value in values
    ]) + "    ]"


def _static_ffmpeg_linkopts(ctx, install_lib, env):
    ffmpeg_pkg_config_path = "{}/pkgconfig".format(install_lib)
    existing_pkg_config_path = env.get("PKG_CONFIG_PATH", "")
    pkg_config_path = (
        "{}:{}".format(ffmpeg_pkg_config_path, existing_pkg_config_path)
        if existing_pkg_config_path
        else ffmpeg_pkg_config_path
    )
    command = "PKG_CONFIG_PATH={} pkg-config --static --libs {}".format(
        _shell_quote(pkg_config_path),
        " ".join(["lib{}".format(lib) for lib in _FFMPEG_LIBS]),
    )
    result = ctx.execute(["bash", "-lc", command], environment = env, quiet = True)
    if result.return_code:
        fail("\n".join([
            "Could not compute static FFmpeg link options.",
            "command: {}".format(command),
            "stdout:\n{}".format(result.stdout),
            "stderr:\n{}".format(result.stderr),
        ]))

    ffmpeg_lib_flags = {("-l" + lib): True for lib in _FFMPEG_LIBS}
    external_linkopts = []
    install_lib_flag = "-L{}".format(install_lib)
    install_lib_rpath_flag = "-Wl,-rpath,{}".format(install_lib)
    for opt in result.stdout.strip().split(" "):
        if not opt:
            continue
        if opt == install_lib_flag or opt == install_lib_rpath_flag or opt in ffmpeg_lib_flags:
            continue
        external_linkopts.append(opt)

    # The FFmpeg archives have internal cycles; grouping keeps GNU ld from
    # depending on a brittle one-pass archive order.
    return [
        install_lib_flag,
        "-Wl,--start-group",
    ] + [
        "-l:lib{}.a".format(lib)
        for lib in _FFMPEG_LIBS
    ] + [
        "-Wl,--end-group",
    ] + external_linkopts


def _detect_jobs(ctx):
    result = ctx.execute(["bash", "-lc", "nproc"], quiet = True)
    if result.return_code == 0:
        jobs = result.stdout.strip()
        if jobs:
            return jobs
    return "8"


def _build_env(ctx, cuda_home):
    env = {}
    for key, value in ctx.os.environ.items():
        env[key] = value

    cuda_bin = "{}/bin".format(cuda_home)
    existing_path = env.get("PATH", "")
    env["PATH"] = "{}:{}".format(cuda_bin, existing_path) if existing_path else cuda_bin
    env["CUDA_HOME"] = env.get("CUDA_HOME", cuda_home)
    env["CUDA_PATH"] = env.get("CUDA_PATH", cuda_home)
    env["CUDA_ROOT"] = env.get("CUDA_ROOT", cuda_home)
    return env


def _normalize_backend(raw_backend):
    backend = (raw_backend or "auto").strip().lower()
    if backend == "":
        backend = "auto"
    if backend not in ["auto", "cuda", "rocm"]:
        fail("Invalid GPU_BACKEND '{}'; expected one of: auto, cuda, rocm".format(backend))
    return backend


def _has_nvcc(ctx, env):
    return ctx.execute(["bash", "-lc", "command -v nvcc"], environment = env, quiet = True).return_code == 0


def _has_rocm_tool(ctx, env):
    return (
        ctx.execute(["bash", "-lc", "command -v hipcc"], environment = env, quiet = True).return_code == 0 or
        ctx.execute(["bash", "-lc", "command -v rocm-smi"], environment = env, quiet = True).return_code == 0
    )


def _detect_nvcc_arch(ctx, env):
    # CUDA 12+ may drop older default arches (e.g. compute_60). Pick the
    # lowest nvcc-supported arch so configure checks remain portable.
    result = ctx.execute(
        ["bash", "-lc", "nvcc --list-gpu-arch | head -n1"],
        environment = env,
        quiet = True,
    )
    if result.return_code == 0:
        arch = result.stdout.strip()
        if arch.startswith("compute_"):
            return arch[len("compute_"):]
    return "75"


def _select_backend(requested_backend, nvcc_available, rocm_available):
    if requested_backend == "cuda":
        if not nvcc_available:
            fail("GPU_BACKEND=cuda was requested, but CUDA toolkit (nvcc) was not detected while preparing FFmpeg build.")
        return "cuda"

    if requested_backend == "rocm":
        if not rocm_available:
            fail("GPU_BACKEND=rocm was requested, but ROCm toolkit (hipcc/rocm-smi) was not detected while preparing FFmpeg build.")
        return "rocm"

    if nvcc_available:
        return "cuda"
    if rocm_available:
        return "rocm"
    fail("Could not find either CUDA (nvcc) or ROCm (hipcc/rocm-smi) while preparing FFmpeg build.")


def _ffmpeg_linux_repo_impl(ctx):
    version = ctx.attr.version
    src_dir = "ffmpeg-src"
    install_dir = "install"
    cuda_home = ctx.os.environ.get("CUDA_HOME", "") or ctx.os.environ.get("CUDA_PATH", "") or ctx.os.environ.get("CUDA_ROOT", "") or "/usr/local/cuda"
    env = _build_env(ctx, cuda_home)

    requested_backend = _normalize_backend(ctx.os.environ.get("GPU_BACKEND", "auto"))
    nvcc_available = _has_nvcc(ctx, env)
    rocm_available = _has_rocm_tool(ctx, env)
    selected_backend = _select_backend(requested_backend, nvcc_available, rocm_available)

    ctx.download_and_extract(
        url = "https://ffmpeg.org/releases/ffmpeg-{}.tar.xz".format(version),
        sha256 = ctx.attr.sha256,
        stripPrefix = "ffmpeg-{}".format(version),
        output = src_dir,
    )

    configure_args = [
        "./configure",
        "--prefix={}".format(ctx.path(install_dir)),
    ] + _FFMPEG_BASE_CONFIGURE_FLAGS

    if selected_backend == "cuda":
        nvcc_arch = _detect_nvcc_arch(ctx, env)
        configure_args.extend([
            "--extra-cflags=-I{}/include".format(cuda_home),
            "--extra-ldflags=-L{}/lib64".format(cuda_home),
            "--nvcc={}/bin/nvcc".format(cuda_home),
            "--nvccflags=-gencode arch=compute_{0},code=sm_{0} -gencode arch=compute_{0},code=compute_{0}".format(nvcc_arch),
        ] + _FFMPEG_CUDA_CONFIGURE_FLAGS)
    else:
        configure_args.extend(_FFMPEG_ROCM_CONFIGURE_FLAGS)

    jobs = _detect_jobs(ctx)

    build_script = [
        "set -euo pipefail",
        "echo 'FFmpeg selected backend: {}'".format(selected_backend),
        "cd {}".format(_shell_quote(ctx.path(src_dir))),
        _join_shell_args(configure_args),
        "make -j{}".format(jobs),
        "make install",
    ]

    result = ctx.execute(["bash", "-lc", "\n".join(build_script)], environment = env, quiet = False)
    if result.return_code:
        fail("\n".join([
            "FFmpeg build failed (return code {}).".format(result.return_code),
            "stdout:\n{}".format(result.stdout),
            "stderr:\n{}".format(result.stderr),
        ]))

    install_lib = str(ctx.path(install_dir + "/lib"))
    ffmpeg_linkopts = _static_ffmpeg_linkopts(ctx, install_lib, env)
    ctx.file(
        "BUILD.bazel",
        """
package(default_visibility = ["//visibility:public"])

licenses(["restricted"])

cc_import(
    name = "avcodec",
    static_library = "install/lib/libavcodec.a",
)

cc_import(
    name = "avdevice",
    static_library = "install/lib/libavdevice.a",
)

cc_import(
    name = "avfilter",
    static_library = "install/lib/libavfilter.a",
)

cc_import(
    name = "avformat",
    static_library = "install/lib/libavformat.a",
)

cc_import(
    name = "avutil",
    static_library = "install/lib/libavutil.a",
)

cc_import(
    name = "postproc",
    static_library = "install/lib/libpostproc.a",
)

cc_import(
    name = "swresample",
    static_library = "install/lib/libswresample.a",
)

cc_import(
    name = "swscale",
    static_library = "install/lib/libswscale.a",
)

cc_library(
    name = "ffmpeg_libs",
    hdrs = glob(["install/include/**/*.h"]),
    includes = ["install/include"],
    linkopts = {ffmpeg_linkopts},
    deps = [
        ":avcodec",
        ":avdevice",
        ":avfilter",
        ":avformat",
        ":avutil",
        ":postproc",
        ":swresample",
        ":swscale",
    ],
)

filegroup(
    name = "ffmpeg",
    srcs = ["install/bin/ffmpeg"],
)

filegroup(
    name = "ffprobe",
    srcs = ["install/bin/ffprobe"],
)

filegroup(
    name = "runtime_libs",
    srcs = glob(["install/lib/*.so*"]),
)
""".format(ffmpeg_linkopts = _starlark_string_list(ffmpeg_linkopts)),
    )


ffmpeg_linux_repository = repository_rule(
    implementation = _ffmpeg_linux_repo_impl,
    attrs = {
        "version": attr.string(default = "6.1.3"),
        "sha256": attr.string(default = "bc5f1e4a4d283a6492354684ee1124129c52293bcfc6a9169193539fbece3487"),
    },
    environ = [
        "AR",
        "CC",
        "CXX",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_ROOT",
        "GPU_BACKEND",
        "HIP_PATH",
        "ROCM_PATH",
        "LD",
        "NM",
        "PATH",
        "PKG_CONFIG",
        "PKG_CONFIG_PATH",
        "RANLIB",
    ],
)
