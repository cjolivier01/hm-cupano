
cc_library(
    name = "python",
    hdrs = glob(["include/**/*.h"]),
    includes = [
      "include",
      "include/python3.11"
    ],
    linkopts = [
      "-Llib",
      "-lpython3",
    ],
    visibility = ["//visibility:public"],
)
