"""Repository rule that builds FFmpeg with required CUDA/NVDEC/NVENC settings."""

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
    "--enable-shared",
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


def _shell_quote(value):
    value_str = str(value)
    return "'{}'".format(value_str.replace("'", "'\\''"))


def _join_shell_args(args):
    return " ".join([_shell_quote(a) for a in args])


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


def _ffmpeg_linux_repo_impl(ctx):
    version = ctx.attr.version
    src_dir = "ffmpeg-src"
    install_dir = "install"
    cuda_home = ctx.os.environ.get("CUDA_HOME", "") or ctx.os.environ.get("CUDA_PATH", "") or ctx.os.environ.get("CUDA_ROOT", "") or "/usr/local/cuda"
    env = _build_env(ctx, cuda_home)

    nvcc_check = ctx.execute(["bash", "-lc", "command -v nvcc"], environment = env, quiet = True)
    if nvcc_check.return_code:
        fail("Could not find nvcc while preparing FFmpeg build (searched PATH with CUDA home {}).".format(cuda_home))

    nvcc_arch = _detect_nvcc_arch(ctx, env)

    ctx.download_and_extract(
        url = "https://ffmpeg.org/releases/ffmpeg-{}.tar.xz".format(version),
        sha256 = ctx.attr.sha256,
        stripPrefix = "ffmpeg-{}".format(version),
        output = src_dir,
    )

    configure_args = [
        "./configure",
        "--prefix={}".format(ctx.path(install_dir)),
        "--extra-cflags=-I{}/include".format(cuda_home),
        "--extra-ldflags=-L{}/lib64".format(cuda_home),
        "--nvcc={}/bin/nvcc".format(cuda_home),
        "--nvccflags=-gencode arch=compute_{0},code=sm_{0} -gencode arch=compute_{0},code=compute_{0}".format(nvcc_arch),
    ] + _FFMPEG_BASE_CONFIGURE_FLAGS + _FFMPEG_CUDA_CONFIGURE_FLAGS

    if ctx.which("rocm-smi"):
        configure_args.extend([
            "--enable-amf",
            "--enable-encoder=hevc_amf",
            "--enable-decoder=hevc_amf",
        ])

    jobs = _detect_jobs(ctx)

    build_script = [
        "set -euo pipefail",
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
    ctx.file(
        "BUILD.bazel",
        """
package(default_visibility = ["//visibility:public"])

licenses(["restricted"])

cc_import(
    name = "avcodec",
    shared_library = "install/lib/libavcodec.so",
)

cc_import(
    name = "avdevice",
    shared_library = "install/lib/libavdevice.so",
)

cc_import(
    name = "avfilter",
    shared_library = "install/lib/libavfilter.so",
)

cc_import(
    name = "avformat",
    shared_library = "install/lib/libavformat.so",
)

cc_import(
    name = "avutil",
    shared_library = "install/lib/libavutil.so",
)

cc_import(
    name = "postproc",
    shared_library = "install/lib/libpostproc.so",
)

cc_import(
    name = "swresample",
    shared_library = "install/lib/libswresample.so",
)

cc_import(
    name = "swscale",
    shared_library = "install/lib/libswscale.so",
)

cc_library(
    name = "ffmpeg_libs",
    hdrs = glob(["install/include/**/*.h"]),
    includes = ["install/include"],
    linkopts = [
        "-L{install_lib}",
        "-Wl,-rpath,{install_lib}",
        "-Wl,--no-as-needed",
        "-l:libavcodec.so",
        "-l:libavdevice.so",
        "-l:libavfilter.so",
        "-l:libavformat.so",
        "-l:libavutil.so",
        "-l:libpostproc.so",
        "-l:libswresample.so",
        "-l:libswscale.so",
        "-Wl,--as-needed",
    ],
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
""".format(install_lib = install_lib),
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
        "LD",
        "NM",
        "PATH",
        "PKG_CONFIG",
        "PKG_CONFIG_PATH",
        "RANLIB",
    ],
)
