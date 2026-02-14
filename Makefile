BAZELISK ?= bazelisk

CPU := $(shell uname -p)
# Some distros return "unknown" for uname -p; fall back to uname -m so Bazel
# still gets a valid --cpu value.
ifeq ($(CPU),unknown)
CPU := $(shell uname -m)
endif
ifeq ($(CPU),x86_64)
CPU := k8
endif

all: print_targets

.PHONY: all print_targets perf debug jetson \
	cuda_pano cuda_pano_debug cuda_pano_n \
	test_cuda_blend test_cuda_blend3 test_cuda_blend_n \
	stitch_two_videos stitch_three_videos \
	unit_tests test clean distclean expunge

perf:
	./perf

debug:
	./bld

jetson:
	$(BAZELISK) build --config=jetson //...

cuda_pano:
	$(BAZELISK) build --config=opt --cpu=$(CPU) //src/pano:cuda_pano

cuda_pano_debug:
	$(BAZELISK) build --config=debug --cpu=$(CPU) //src/pano:cuda_pano

cuda_pano_n:
	$(BAZELISK) build --config=opt --cpu=$(CPU) //src/pano:cuda_pano_n

test_cuda_blend:
	$(BAZELISK) build --config=opt --cpu=$(CPU) //tests:test_cuda_blend

test_cuda_blend3:
	$(BAZELISK) build --config=opt --cpu=$(CPU) //tests:test_cuda_blend3

test_cuda_blend_n:
	$(BAZELISK) build --config=opt --cpu=$(CPU) //tests:test_cuda_blend_n

stitch_two_videos:
	$(BAZELISK) build --config=opt --cpu=$(CPU) //tests:stitch_two_videos

stitch_three_videos:
	$(BAZELISK) build --config=opt --cpu=$(CPU) //tests:stitch_three_videos

unit_tests:
	$(BAZELISK) test --config=opt --cpu=$(CPU) \
		//src/cuda:cudaBlend3_test \
		//src/pano:cudaPano3_test \
		//src/pano:matchSeam3_test

test:
	$(BAZELISK) test --config=opt --cpu=$(CPU) //...

clean:
	$(BAZELISK) clean

distclean expunge:
	$(BAZELISK) clean --expunge

print_targets:
	@printf '%s\n' \
		"Available make targets (run 'make <target>'):" \
		'' \
		'Build Outputs' \
		'-------------' \
		'perf               Builds every Bazel target using ./perf (opt config, CPU auto-detected for x86_64 as k8).' \
		'debug              Builds every Bazel target using ./bld (debug config).' \
		'jetson             Builds every Bazel target with --config=jetson for JetPack/Jetson environments.' \
		'cuda_pano          Builds the primary CUDA panorama library target.' \
		'cuda_pano_debug    Builds //src/pano:cuda_pano in debug mode.' \
		'cuda_pano_n        Builds the generic N-image panorama library target.' \
		'test_cuda_blend    Builds the 2-image stitch demo binary.' \
		'test_cuda_blend3   Builds the 3-image stitch demo binary.' \
		'test_cuda_blend_n  Builds the N-image stitch demo binary.' \
		'stitch_two_videos  Builds the two-video end-to-end stitching app.' \
		'stitch_three_videos Builds the three-video end-to-end stitching app.' \
		'' \
		'Verification' \
		'-------------' \
		'unit_tests         Runs the core GTest targets under src/cuda and src/pano (opt mode).' \
		'test               Runs the full Bazel test suite in opt mode.' \
		'' \
		'Maintenance & Cleanup' \
		'---------------------' \
		'clean              bazel clean to drop cached outputs.' \
		'distclean          bazel clean --expunge (also aliased as expunge).' \
		'expunge            Same as distclean; provided for convenience.' \
		'' \
		'Meta' \
		'----' \
		'print_targets      Shows this help text.'
