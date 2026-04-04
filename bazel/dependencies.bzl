def _discover_root(ctx, env_vars, probe_script):
    for env_var in env_vars:
        value = ctx.os.environ.get(env_var)
        if value and ctx.path(value).exists:
            return value

    result = ctx.execute(
        ["/bin/bash", "-lc", probe_script],
        quiet = True,
    )

    if result.return_code == 0:
        root = result.stdout.strip()
        if root:
            return root

    return ""

def _symlink_entries(ctx, root, entries):
    for entry in entries:
        path = root + "/" + entry
        if ctx.path(path).exists:
            ctx.symlink(ctx.path(path), entry)

def _normalize_backend(raw_backend):
    backend = (raw_backend or "auto").strip().lower()
    if backend == "":
        backend = "auto"
    if backend not in ["auto", "cuda", "rocm"]:
        fail("Invalid GPU_BACKEND '{}'; expected one of: auto, cuda, rocm".format(backend))
    return backend

def _has_tool(ctx, tool):
    return ctx.which(tool) != None

def _cuda_available(ctx):
    if _has_tool(ctx, "nvcc"):
        return True

    for env_var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"]:
        root = ctx.os.environ.get(env_var, "")
        if root and ctx.path(root + "/bin/nvcc").exists:
            return True
    return False

def _rocm_available(ctx):
    if _has_tool(ctx, "hipcc") or _has_tool(ctx, "rocm-smi"):
        return True

    for env_var in ["ROCM_PATH", "HIP_PATH"]:
        root = ctx.os.environ.get(env_var, "")
        if not root:
            continue
        if ctx.path(root + "/bin/hipcc").exists or ctx.path(root + "/bin/rocm-smi").exists:
            return True
    return False

def _select_backend(ctx):
    requested = _normalize_backend(ctx.os.environ.get("GPU_BACKEND", "auto"))
    cuda_ok = _cuda_available(ctx)
    rocm_ok = _rocm_available(ctx)

    if requested == "cuda":
        if not cuda_ok:
            fail("GPU_BACKEND=cuda was requested, but CUDA toolkit (nvcc) was not detected.")
        return "cuda"

    if requested == "rocm":
        if not rocm_ok:
            fail("GPU_BACKEND=rocm was requested, but ROCm toolkit (hipcc/rocm-smi) was not detected.")
        return "rocm"

    if cuda_ok:
        return "cuda"
    if rocm_ok:
        return "rocm"
    fail("No GPU backend detected. Install CUDA or ROCm, or set GPU_BACKEND=cuda|rocm with the matching toolkit available.")

def _opencv_linux_repo_impl(ctx):
    backend = _select_backend(ctx)

    # Mirror enough of /usr and /opt so buildfiles/third_party/opencv_linux_defs.bzl
    # can probe headers/libs and toolkit markers with stable relative paths.
    symlinks = {
        "include": "/usr/include",
        "lib": "/usr/lib",
        "lib64": "/usr/lib64",
        "local": "/usr/local",
        "bin": "/usr/bin",
        "opt": "/opt",
    }
    for name, target in symlinks.items():
        if ctx.path(target).exists:
            ctx.symlink(ctx.path(target), name)

    ctx.file("WORKSPACE", 'workspace(name = "%s")\n' % ctx.name)
    ctx.file(
        "BUILD.bazel",
        """
load("@//buildfiles/third_party:opencv_linux_defs.bzl", "opencv_library")

opencv_library(
    name = "opencv",
    backend = "{backend}",
)
""".format(backend = backend),
    )

def _local_rocm_repo_impl(ctx):
    root = _discover_root(
        ctx,
        ["ROCM_PATH", "HIP_PATH"],
        """
set -eu
for d in /opt/rocm /opt/rocm-*; do
    [ -x "$d/bin/hipcc" ] || continue
    [ -f "$d/include/hip/hip_runtime.h" ] || continue
    [ -f "$d/lib/libamdhip64.so" ] || continue
    printf '%s' "$d"
    exit 0
done
exit 1
""",
    )

    build = [
        'load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")',
        'package(default_visibility = ["//visibility:public"])',
    ]

    if root:
        _symlink_entries(ctx, root, ["bin", "include", "lib"])

        build.extend([
            'filegroup(name = "hipcc", srcs = ["bin/hipcc"])',
            'cc_import(',
            '    name = "amdhip64",',
            '    shared_library = "lib/libamdhip64.so",',
            ')',
            'cc_library(',
            '    name = "rocm_sdk_core",',
            '    srcs = [],',
            '    hdrs = [],',
            '    includes = ["include"],',
            ')',
            'cc_library(',
            '    name = "hip_runtime",',
            '    srcs = [],',
            '    hdrs = [],',
            '    includes = ["include"],',
            '    deps = [":amdhip64"],',
            ')',
        ])
    else:
        build.extend([
            'filegroup(name = "hipcc", srcs = [])',
            'cc_library(name = "amdhip64", srcs = [], hdrs = [])',
            'cc_library(name = "rocm_sdk_core", srcs = [], hdrs = [], includes = [])',
            'cc_library(name = "hip_runtime", srcs = [], hdrs = [], includes = [])',
        ])

    ctx.file("WORKSPACE", 'workspace(name = "%s")\n' % ctx.name)
    ctx.file("BUILD.bazel", "\n".join(build) + "\n")

local_rocm_repository = repository_rule(
    implementation = _local_rocm_repo_impl,
    environ = ["HIP_PATH", "ROCM_PATH"],
)

opencv_linux_repository = repository_rule(
    implementation = _opencv_linux_repo_impl,
    environ = [
        "GPU_BACKEND",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_ROOT",
        "ROCM_PATH",
        "HIP_PATH",
        "PATH",
    ],
)
