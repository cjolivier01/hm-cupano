config_setting(
    name = "aarch64-linux-gnu",
    constraint_values = ["@platforms//cpu:aarch64"],
)

config_setting(
    name = "x86_64-linux-gnu",
    constraint_values = ["@platforms//cpu:x86_64"],
)

cc_library(
    name = "glib",
    hdrs = glob([
        "include/glib-2.0/**/*.h",
        "lib/aarch64-linux-gnu/glib-2.0/include/**/*.h",
        "lib/x86_64-linux-gnu/glib-2.0/include/**/*.h",
    ], allow_empty = True),
    includes = [
        "include/glib-2.0",
    ] + select({
        ":aarch64-linux-gnu": ["lib/aarch64-linux-gnu/glib-2.0/include"],
        ":x86_64-linux-gnu": ["lib/x86_64-linux-gnu/glib-2.0/include"],
        "//conditions:default": [],
    }),
    linkopts = select({
        ":aarch64-linux-gnu": ["-L/usr/lib/aarch64-linux-gnu"],
        ":x86_64-linux-gnu": ["-L/usr/lib/x86_64-linux-gnu"],
        "//conditions:default": [],
    }) + [
        "-lglib-2.0",
        "-lgobject-2.0",
    ],
    visibility = ["//visibility:public"],
)
