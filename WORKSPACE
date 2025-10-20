_workspace_name = "hm-cupano"

workspace(name = _workspace_name)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

git_repository(
    name = "rules_cuda",
    # v0.2.3 breaks some lubcupti for our version of bazel
    commit = "3f2429254ec956220557e79ea9d5f5e8871c2907",
    remote = "https://github.com/bazel-contrib/rules_cuda",
)

load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")

rules_cuda_dependencies()

register_detected_cuda_toolchains()

# External seam blender used to compute indexed seam masks for N-image/3-image paths
git_repository(
    name = "multiblend",
    remote = "https://github.com/cjolivier01/multiblend",
    branch = "colivier/hm",
)

# System libraries required by multiblend
new_local_repository(
    name = "libtiff",
    path = "/usr",
    build_file_content = """
cc_library(
    name = "tiff",
    hdrs = glob(["include/tiff*.h", "include/x86_64-linux-gnu/tiff*.h"], allow_empty = True),
    includes = ["include", "include/x86_64-linux-gnu"],
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu",
        "-L/usr/lib",
        "-Wl,-rpath,/usr/lib/x86_64-linux-gnu",
        "-ltiff",
        "-lwebp",
        "-lsharpyuv",
        "-lLerc",
        "-ljbig",
        "-ldeflate",
        "-lz",
    ],
    visibility = ["//visibility:public"],
)
""",
)

new_local_repository(
    name = "libjpeg",
    path = "/usr",
    build_file_content = """
cc_library(
    name = "jpeg",
    hdrs = glob(["include/j*.h", "include/x86_64-linux-gnu/j*.h"], allow_empty = True),
    includes = ["include", "include/x86_64-linux-gnu"],
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu",
        "-L/usr/lib",
        "-Wl,-rpath,/usr/lib/x86_64-linux-gnu",
        "-ljpeg",
    ],
    visibility = ["//visibility:public"],
)
""",
)

new_local_repository(
    name = "libpng",
    path = "/usr",
    build_file_content = """
cc_library(
    name = "png",
    hdrs = glob(["include/png*.h", "include/libpng*/*.h"], allow_empty = True),
    includes = ["include"],
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu",
        "-L/usr/lib",
        "-Wl,-rpath,/usr/lib/x86_64-linux-gnu",
        "-lpng",
        "-lz",
    ],
    visibility = ["//visibility:public"],
)
""",
)

new_local_repository(
    name = "opencv_linux",
    build_file = "@//buildfiles:third_party/opencv_linux.BUILD",
    path = "/usr/include",
    # path = "/usr/local/include",
)

new_git_repository(
    name = "yaml-cpp",
    build_file = "@//buildfiles:third_party/yaml-cpp.BUILD",
    tag="0.8.0",
    remote = "https://github.com/jbeder/yaml-cpp.git",
)

# Choose the most recent version available at
# https://registry.bazel.build/modules/googletest
new_git_repository(
    name = "googletest",
    tag = "v1.17.0",
    remote = "https://github.com/google/googletest.git",
)

# Optional seam blender alternative (two-image default; can also be used for three)
# Fork with Bazel build rules for enblend/enfuse
git_repository(
    name = "enblend",
    remote = "https://github.com/cjolivier01/enblend-enfuse",
    branch = "colivier/hockeymom",
)
