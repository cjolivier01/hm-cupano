_workspace_name = "jetson-utils"

workspace(name = _workspace_name)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
    name = "rules_cuda",
    # v0.2.3 breaks some lubcupti for our version of bazel
    commit = "3f2429254ec956220557e79ea9d5f5e8871c2907",
    remote = "https://github.com/bazel-contrib/rules_cuda",
)

load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")

rules_cuda_dependencies()

register_detected_cuda_toolchains()

load("//:conda.bzl", "detect_conda_python")

# detect_conda_python(name = "conda_python")

new_local_repository(
    name = "conda_python",
    path = "/home/colivier/.conda/envs/ubuntu",
    build_file = "@//buildfiles:third_party/python.BUILD",
)

# git_repository(
#     name = "rules_python",
#     remote = "https://github.com/bazelbuild/rules_python.git",
#     tag = "0.1.0",
# )

http_archive(
    name = "rules_python",
    sha256 = "9c6e26911a79fbf510a8f06d8eedb40f412023cf7fa6d1461def27116bff022c",
    strip_prefix = "rules_python-1.1.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/1.1.0/rules_python-1.1.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

http_archive(
    name = "pybind11",
    urls = ["https://github.com/pybind/pybind11/archive/refs/tags/v2.10.4.tar.gz"],
    strip_prefix = "pybind11-2.10.4",
    build_file = "//buildfiles/third_party:pybind11.BUILD",
)

new_local_repository(
    name = "glibconfig_x86",
    build_file = "//buildfiles:third_party/glibconfig.BUILD",
    path = "/usr/lib/x86_64-linux-gnu/glib-2.0/include",
)

new_local_repository(
    name = "glibconfig_aarch64",
    build_file = "//buildfiles:third_party/glibconfig.BUILD",
    path = "/usr/lib/aarch64-linux-gnu/glib-2.0/include",
)

new_local_repository(
    name = "glib",
    build_file = "@//buildfiles:third_party/glib_nobuild.BUILD",
    # path = "/usr/local",
    path = "/usr",
)

new_local_repository(
    name = "json_glib",
    build_file = "@//buildfiles:third_party/json_glib.BUILD",
    path = "/usr",
)

new_local_repository(
    name = "gstreamer",
    build_file = "@//buildfiles:third_party/gstreamer_nobuild.BUILD",
    # path = "/usr/local",
    path = "/usr",
)

new_local_repository(
    name = "libsoup",
    build_file = "@//buildfiles:third_party/libsoup.BUILD",
    path = "/usr",
)

new_local_repository(
    name = "opencv_linux",
    build_file = "@//buildfiles:third_party/opencv_linux.BUILD",
    path = "/usr/include",
)
