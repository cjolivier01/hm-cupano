def _detect_conda_python(repository_ctx):
    home_dir = repository_ctx.os.environ.get("HOME", "")
    conda_prefix = repository_ctx.os.environ.get("CONDA_PREFIX", home_dir + "/.conda/envs/ubuntu")
    print("Conda prefix: ", conda_prefix)
    if not conda_prefix:
        fail("Environment variable CONDA_PREFIX is not set.")

    repository_ctx.symlink(conda_prefix, "conda_python")

    # Run a command inside Conda to get the Python version
    result = repository_ctx.execute([conda_prefix + "/bin/python", "-c", "import sys; print(sys.version[:4])"])

    if result.return_code != 0:
        fail("Failed to get Python version: " + result.stderr)

    python_version = result.stdout.strip()

    # Debugging output
    print("Detected Conda Python version:", python_version)

    repository_ctx.file("BUILD.bazel", """
cc_library(
    name = "python",
    hdrs = glob(["include/**/*.h"]),
    includes = [
        "include",
        "include/python3.11",
    ],
    visibility = ["//visibility:public"],
)
""")

detect_conda_python = repository_rule(
    implementation = _detect_conda_python,
    environ = ["CONDA_PREFIX"],
)
