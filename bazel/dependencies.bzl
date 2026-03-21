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
