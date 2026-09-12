config_setting(
    name = "aarch64-linux-gnu",
    constraint_values = ["@platforms//cpu:aarch64"],
)

config_setting(
    name = "x86_64-linux-gnu",
    constraint_values = ["@platforms//cpu:x86_64"],
)

cc_library(
    name = "gstreamer",
    hdrs = glob([
        "include/gstreamer-1.0/**/*.h",
    ], allow_empty = True),
    includes = [
        "include/gstreamer-1.0",
    ],
    linkopts = select({
        ":aarch64-linux-gnu": ["-L/usr/lib/aarch64-linux-gnu"],
        ":x86_64-linux-gnu": ["-L/usr/lib/x86_64-linux-gnu"],
        "//conditions:default": [],
    }) + [
        "-lgstreamer-1.0",
        "-lgstbase-1.0",
        "-lgstvideo-1.0",
        "-lgstaudio-1.0",
        "-lgstpbutils-1.0",
        "-lgstrtsp-1.0",
        "-lgstapp-1.0",
        "-lgstsdp-1.0",
        "-lgstrtp-1.0",
        "-lgstwebrtc-1.0",
        "-lgstrtspserver-1.0",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@glib",
    ],
)
