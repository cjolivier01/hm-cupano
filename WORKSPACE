_workspace_name = "hm-cupano"

workspace(name = _workspace_name)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("//bazel:dependencies.bzl", "local_rocm_repository", "opencv_linux_repository")

git_repository(
    name = "rules_cuda",
    # Keep a release tag with Blackwell arch support (sm_120+) and WORKSPACE compatibility.
    commit = "32414222a7997ee66035d51404fc5427df94c699",
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
    name = "lcms2",
    path = "/usr",
    build_file_content = """
cc_library(
    name = "lcms2",
    hdrs = glob(["include/lcms2*.h", "include/x86_64-linux-gnu/lcms2*.h"], allow_empty = True),
    includes = ["include", "include/x86_64-linux-gnu"],
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu",
        "-L/usr/lib",
        "-Wl,-rpath,/usr/lib/x86_64-linux-gnu",
        "-llcms2",
    ],
    visibility = ["//visibility:public"],
)
""",
)

new_local_repository(
    name = "boost",
    path = "/usr",
    build_file_content = """
cc_library(
    name = "filesystem",
    hdrs = glob(["include/boost/**/*.hpp"], allow_empty = True),
    includes = ["include"],
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu",
        "-L/usr/lib",
        "-Wl,-rpath,/usr/lib/x86_64-linux-gnu",
        "-lboost_filesystem",
        "-lboost_system",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "system",
    hdrs = glob(["include/boost/**/*.hpp"], allow_empty = True),
    includes = ["include"],
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu",
        "-L/usr/lib",
        "-Wl,-rpath,/usr/lib/x86_64-linux-gnu",
        "-lboost_system",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "algorithm",
    hdrs = glob(["include/boost/**/*.hpp"], allow_empty = True),
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "math",
    hdrs = glob(["include/boost/**/*.hpp"], allow_empty = True),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""",
)

new_local_repository(
    name = "gsl",
    path = "/usr",
    build_file_content = """
cc_library(
    name = "gsl",
    hdrs = glob(["include/gsl/*.h"], allow_empty = True),
    includes = ["include"],
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu",
        "-L/usr/lib",
        "-Wl,-rpath,/usr/lib/x86_64-linux-gnu",
        "-lgsl",
        "-lgslcblas",
    ],
    visibility = ["//visibility:public"],
)
""",
)

new_local_repository(
    name = "vigra",
    path = "/usr",
    build_file_content = """
cc_library(
    name = "vigra",
    hdrs = glob(["include/vigra/**/*.hxx", "include/vigra/**/*.h"], allow_empty = True),
    includes = ["include"],
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu",
        "-L/usr/lib",
        "-Wl,-rpath,/usr/lib/x86_64-linux-gnu",
        "-lvigraimpex",
    ],
    visibility = ["//visibility:public"],
)
""",
)

opencv_linux_repository(
    name = "opencv_linux",
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

# Local ROCm (HIP) headers and runtime
local_rocm_repository(
    name = "local_rocm",
)
