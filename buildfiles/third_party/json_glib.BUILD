config_setting(
    name = "aarch64-linux-gnu",
    constraint_values = ["@platforms//cpu:aarch64"],
)

config_setting(
    name = "x86_64-linux-gnu",
    constraint_values = ["@platforms//cpu:x86_64"],
)

cc_library(
    name = "json_glib",
    hdrs = glob([
        "include/json-glib-1.0/**/*.h",
    ], allow_empty = True),
    includes = [
        "include/json-glib-1.0",
    ],
    linkopts = select({
        ":aarch64-linux-gnu": ["-L/usr/lib/aarch64-linux-gnu"],
        ":x86_64-linux-gnu": ["-L/usr/lib/x86_64-linux-gnu"],
        "//conditions:default": [],
    }) + [
        "-ljson-glib-1.0",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@glib",
    ],
)
