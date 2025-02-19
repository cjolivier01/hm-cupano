_workspace_name = "hm-cupano"

workspace(name = _workspace_name)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "rules_cuda",
    # v0.2.3 breaks some lubcupti for our version of bazel
    commit = "3f2429254ec956220557e79ea9d5f5e8871c2907",
    remote = "https://github.com/bazel-contrib/rules_cuda",
)

load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")

rules_cuda_dependencies()

register_detected_cuda_toolchains()

new_local_repository(
    name = "opencv_linux",
    build_file = "@//buildfiles:third_party/opencv_linux.BUILD",
    #path = "/usr/include",
    path = "/usr/local/include",
)
