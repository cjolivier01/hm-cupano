"""Bzlmod extension to expose a Bazel-built FFmpeg repository."""

load("//bazel:ffmpeg_repo.bzl", "ffmpeg_linux_repository")


def _ffmpeg_ext_impl(_module_ctx):
    ffmpeg_linux_repository(name = "ffmpeg_linux")


ffmpeg_ext = module_extension(
    implementation = _ffmpeg_ext_impl,
)
