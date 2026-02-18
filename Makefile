TOPDIR := $(shell pwd)

UNAME_M := $(shell uname -m)
BAZEL_CPU_DEFAULT := $(if $(filter x86_64,$(UNAME_M)),k8,$(UNAME_M))
BAZEL_CPU ?= $(BAZEL_CPU_DEFAULT)

BAZELISK ?= bazelisk
BAZEL_TARGETS ?= //...
BAZEL_ARGS ?=

all: print_targets

.PHONY: all print_targets perf debug bld test clean distclean expunge

perf:
	$(BAZELISK) build --config=opt --cpu=$(BAZEL_CPU) $(BAZEL_ARGS) $(BAZEL_TARGETS)

debug bld:
	$(BAZELISK) build --config=debug --cpu=$(BAZEL_CPU) $(BAZEL_ARGS) $(BAZEL_TARGETS)

test:
	$(BAZELISK) test --config=opt --cpu=$(BAZEL_CPU) $(BAZEL_ARGS) $(BAZEL_TARGETS)

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
		'perf         Build every Bazel target with --config=opt (optimized).' \
		'debug        Build every Bazel target with --config=debug (symbols, no opt).' \
		'bld          Alias for debug (matches legacy workflow).' \
		'' \
		'Developer Workflow' \
		'------------------' \
		'test         Runs the optimized Bazel test suite.' \
		'' \
		'Maintenance & Cleanup' \
		'---------------------' \
		'clean        bazel clean to drop cached outputs.' \
		'distclean    bazel clean --expunge (also aliased as expunge) for a fully fresh Bazel state.' \
		'expunge      Same as distclean; provided for convenience.' \
		'' \
		'Variables' \
		'---------' \
		'BAZEL_CPU     Override auto-detected CPU ("k8" on x86_64; otherwise uname -m).' \
		'BAZEL_TARGETS Override default build/test target set (default: //...).' \
		'BAZEL_ARGS    Extra flags passed through to bazelisk (e.g. BAZEL_ARGS=--test_output=errors).'

