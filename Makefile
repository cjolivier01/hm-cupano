TOPDIR := $(shell pwd)

UNAME_M := $(shell uname -m)
BAZEL_CPU_DEFAULT := $(if $(filter x86_64,$(UNAME_M)),k8,$(UNAME_M))
BAZEL_CPU ?= $(BAZEL_CPU_DEFAULT)

BAZELISK ?= bazelisk
BAZEL_TARGETS ?= //...
BAZEL_ARGS ?=

DEFAULT_BACKEND := $(shell bash -lc 'source "$(TOPDIR)/toolchain_env.sh"; if ensure_cuda_env >/dev/null 2>&1; then printf cuda; elif ensure_rocm_env >/dev/null 2>&1; then printf rocm; fi')

define run_default_build
source "$(TOPDIR)/toolchain_env.sh"; \
if ensure_cuda_env >/dev/null 2>&1; then \
	ensure_rocm_env >/dev/null 2>&1 || true; \
	ensure_cuda_bazel_archs; \
	exec $(BAZELISK) build --config=$(1) --cpu=$(BAZEL_CPU) --cuda_archs=$${CUDA_BAZEL_ARCHS} $(BAZEL_ARGS) $(BAZEL_TARGETS); \
elif ensure_rocm_env >/dev/null 2>&1; then \
	ensure_cuda_env >/dev/null 2>&1 || true; \
	exec $(BAZELISK) build --config=$(1) --config=rocm --cpu=$(BAZEL_CPU) $(BAZEL_ARGS) $(BAZEL_TARGETS); \
else \
	printf 'No CUDA or ROCm toolkit detected.\n' >&2; \
	exit 1; \
fi
endef

define run_default_test
source "$(TOPDIR)/toolchain_env.sh"; \
if ensure_cuda_env >/dev/null 2>&1; then \
	ensure_rocm_env >/dev/null 2>&1 || true; \
	ensure_cuda_bazel_archs; \
	exec $(BAZELISK) test --config=$(1) --cpu=$(BAZEL_CPU) --cuda_archs=$${CUDA_BAZEL_ARCHS} $(BAZEL_ARGS) $(BAZEL_TARGETS); \
elif ensure_rocm_env >/dev/null 2>&1; then \
	ensure_cuda_env >/dev/null 2>&1 || true; \
	exec $(BAZELISK) test --config=$(1) --config=rocm --cpu=$(BAZEL_CPU) $(BAZEL_ARGS) $(BAZEL_TARGETS); \
else \
	printf 'No CUDA or ROCm toolkit detected.\n' >&2; \
	exit 1; \
fi
endef

all: $(if $(DEFAULT_BACKEND),perf,print_targets)

.PHONY: all print_targets perf debug bld cuda rocm test test-cuda test-rocm clean distclean expunge

perf:
	@bash -lc '$(call run_default_build,opt)'

debug:
	@bash -lc '$(call run_default_build,debug)'

bld: debug

cuda:
	@bash -lc 'source "$(TOPDIR)/toolchain_env.sh"; ensure_cuda_env; ensure_rocm_env || true; ensure_cuda_bazel_archs; exec $(BAZELISK) build --config=opt --cpu=$(BAZEL_CPU) --cuda_archs=$${CUDA_BAZEL_ARCHS} $(BAZEL_ARGS) $(BAZEL_TARGETS)'

rocm:
	@bash -lc 'source "$(TOPDIR)/toolchain_env.sh"; ensure_cuda_env || true; ensure_rocm_env; exec $(BAZELISK) build --config=opt --config=rocm --cpu=$(BAZEL_CPU) $(BAZEL_ARGS) $(BAZEL_TARGETS)'

test:
	@bash -lc '$(call run_default_test,opt)'

test-cuda:
	@bash -lc 'source "$(TOPDIR)/toolchain_env.sh"; ensure_cuda_env; ensure_rocm_env || true; ensure_cuda_bazel_archs; exec $(BAZELISK) test --config=opt --cpu=$(BAZEL_CPU) --cuda_archs=$${CUDA_BAZEL_ARCHS} $(BAZEL_ARGS) $(BAZEL_TARGETS)'

test-rocm:
	@bash -lc 'source "$(TOPDIR)/toolchain_env.sh"; ensure_cuda_env || true; ensure_rocm_env; exec $(BAZELISK) test --config=opt --config=rocm --cpu=$(BAZEL_CPU) $(BAZEL_ARGS) $(BAZEL_TARGETS)'

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
		'perf         Build with the preferred installed backend (CUDA first, then ROCm).' \
		'cuda         Build with the CUDA backend.' \
		'rocm         Build with the HIP/ROCm backend.' \
		'debug        Debug build with the preferred installed backend.' \
		'bld          Alias for debug (matches legacy workflow).' \
		'' \
		'Developer Workflow' \
		'------------------' \
		'test         Run tests with the preferred installed backend.' \
		'test-cuda    Run tests with the CUDA backend.' \
		'test-rocm    Run tests with the HIP/ROCm backend.' \
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
		'BAZEL_ARGS    Extra flags passed through to bazelisk (e.g. BAZEL_ARGS=--test_output=errors).' \
		'CUDA_BAZEL_ARCHS Override detected rules_cuda arch list (e.g. sm_70;sm_80).'
