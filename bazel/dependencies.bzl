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
    if backend not in ["auto", "cuda", "rocm", "vulkan"]:
        fail("Invalid GPU_BACKEND '{}'; expected one of: auto, cuda, rocm, vulkan".format(backend))
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

def _vulkan_available(ctx):
    sdk_root = ctx.os.environ.get("VULKAN_SDK", "")
    if sdk_root and (
        ctx.path(sdk_root + "/include/vulkan/vulkan.h").exists or
        ctx.path(sdk_root + "/Include/vulkan/vulkan.h").exists
    ):
        return True

    if _has_tool(ctx, "vulkaninfo"):
        return True

    if ctx.path("/usr/include/vulkan/vulkan.h").exists:
        return True
    if ctx.path("/usr/local/include/vulkan/vulkan.h").exists:
        return True
    if ctx.path("/opt/vulkan/include/vulkan/vulkan.h").exists:
        return True

    if ctx.path("/usr/lib/x86_64-linux-gnu/libvulkan.so").exists:
        return True
    if ctx.path("/usr/lib/aarch64-linux-gnu/libvulkan.so").exists:
        return True
    if ctx.path("/usr/lib64/libvulkan.so").exists:
        return True
    if ctx.path("/usr/lib/libvulkan.so").exists:
        return True

    return False

def _parent_dir(path):
    parts = path.split("/")
    if len(parts) <= 1:
        return ""
    return "/".join(parts[:-1])

def _dedupe_non_empty(values):
    seen = {}
    out = []
    for value in values:
        if not value or value in seen:
            continue
        seen[value] = True
        out.append(value)
    return out

def _resolve_rocm_layout(ctx, root):
    if not ctx.path(root + "/bin/hipcc").exists:
        return None
    if not ctx.path(root + "/include/hip/hip_runtime.h").exists:
        return None

    lib_rel = ""
    for rel in [
        "lib",
        "lib64",
        "lib/x86_64-linux-gnu",
        "lib/aarch64-linux-gnu",
    ]:
        if ctx.path(root + "/" + rel + "/libamdhip64.so").exists:
            lib_rel = rel
            break

    if not lib_rel:
        return None

    return struct(
        root = root,
        include_rel = "include",
        lib_rel = lib_rel,
    )

def _find_rocm_layout(ctx):
    roots = []
    for env_var in ["ROCM_PATH", "HIP_PATH"]:
        value = ctx.os.environ.get(env_var, "")
        if value:
            roots.append(value)

    # Prefer canonical ROCm installs before generic /usr layouts.
    roots.extend([
        "/opt/rocm",
        "/usr/local/rocm",
    ])

    hipcc = ctx.which("hipcc")
    if hipcc != None:
        hipcc_path = str(hipcc)
        resolved = ctx.execute(
            ["/bin/bash", "-lc", "readlink -f \"{}\"".format(hipcc_path)],
            quiet = True,
        )
        if resolved.return_code == 0:
            resolved_path = resolved.stdout.strip()
            if resolved_path:
                roots.append(_parent_dir(_parent_dir(resolved_path)))
        roots.append(_parent_dir(_parent_dir(hipcc_path)))

    roots.append("/usr")

    for root in _dedupe_non_empty(roots):
        layout = _resolve_rocm_layout(ctx, root)
        if layout != None:
            return layout

    return None

def _select_backend(ctx):
    requested = _normalize_backend(ctx.os.environ.get("GPU_BACKEND", "auto"))
    cuda_ok = _cuda_available(ctx)
    rocm_ok = _rocm_available(ctx)
    vulkan_ok = _vulkan_available(ctx)

    if requested == "cuda":
        if not cuda_ok:
            fail("GPU_BACKEND=cuda was requested, but CUDA toolkit (nvcc) was not detected.")
        return "cuda"

    if requested == "rocm":
        if not rocm_ok:
            fail("GPU_BACKEND=rocm was requested, but ROCm toolkit (hipcc/rocm-smi) was not detected.")
        return "rocm"

    if requested == "vulkan":
        if not vulkan_ok:
            fail("GPU_BACKEND=vulkan was requested, but Vulkan markers (headers/loader) were not detected.")
        return "vulkan"

    if cuda_ok:
        return "cuda"
    if rocm_ok:
        return "rocm"
    if vulkan_ok:
        return "vulkan"
    fail("No GPU backend detected. Install CUDA/ROCm/Vulkan, or set GPU_BACKEND=cuda|rocm|vulkan with the matching toolkit available.")

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
    layout = _find_rocm_layout(ctx)

    build = [
        'load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")',
        'package(default_visibility = ["//visibility:public"])',
    ]

    if layout:
        root = layout.root
        ctx.file(
            "hipcc_wrapper.sh",
            "#!/usr/bin/env bash\nexec \"{}/bin/hipcc\" \"$@\"\n".format(root),
            executable = True,
        )
        ctx.symlink(ctx.path(root + "/bin"), "bin")
        ctx.symlink(ctx.path(root + "/" + layout.include_rel), "include")
        ctx.symlink(ctx.path(root + "/" + layout.lib_rel), "lib")

        build.extend([
            'filegroup(name = "hipcc", srcs = ["hipcc_wrapper.sh"])',
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
        ctx.file(
            "hipcc_wrapper.sh",
            "#!/usr/bin/env bash\necho \"hipcc not found\" >&2\nexit 127\n",
            executable = True,
        )
        build.extend([
            'filegroup(name = "hipcc", srcs = ["hipcc_wrapper.sh"])',
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
