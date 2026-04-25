#!/bin/bash

append_path_once() {
	local entry="$1"

	case ":$PATH:" in
	*":$entry:"*) ;;
	*)
		export PATH="$entry:$PATH"
		;;
	esac
}

valid_cuda_root() {
	local root="$1"
	local rel=""

	[ -n "$root" ] || return 1
	[ -x "$root/bin/nvcc" ] || return 1

	for rel in \
		targets/x86_64-linux/lib/libculibos.a \
		targets/aarch64-linux/lib/libculibos.a \
		targets/sbsa-linux/lib/libculibos.a
	do
		[ -f "$root/$rel" ] && return 0
	done

	return 1
}

valid_rocm_root() {
	local root="$1"

	[ -n "$root" ] || return 1
	[ -x "$root/bin/hipcc" ] || return 1
	[ -f "$root/include/hip/hip_runtime.h" ] || return 1
	[ -f "$root/lib/libamdhip64.so" ] || return 1

	return 0
}

valid_vulkan_root() {
	local root="$1"

	[ -n "$root" ] || return 1
	[ -f "$root/include/vulkan/vulkan.h" ] || return 1
	[ -f "$root/lib/libvulkan.so" ] || [ -f "$root/lib64/libvulkan.so" ] || return 1

	return 0
}

detect_cuda_path() {
	local candidate=""

	for candidate in "${CUDA_PATH:-}" "${CUDA_HOME:-}" "${CUDA_ROOT:-}"; do
		[ -n "$candidate" ] || continue
		valid_cuda_root "$candidate" && { printf '%s\n' "$candidate"; return 0; }
	done

	if command -v nvcc >/dev/null 2>&1; then
		candidate="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
		valid_cuda_root "$candidate" && { printf '%s\n' "$candidate"; return 0; }
	fi

	for candidate in /usr/local/cuda /usr/local/cuda-*; do
		valid_cuda_root "$candidate" && { printf '%s\n' "$candidate"; return 0; }
	done

	return 1
}

detect_rocm_path() {
	local candidate=""

	for candidate in "${ROCM_PATH:-}" "${HIP_PATH:-}"; do
		[ -n "$candidate" ] || continue
		valid_rocm_root "$candidate" && { printf '%s\n' "$candidate"; return 0; }
	done

	if command -v hipcc >/dev/null 2>&1; then
		candidate="$(dirname "$(dirname "$(readlink -f "$(command -v hipcc)")")")"
		valid_rocm_root "$candidate" && { printf '%s\n' "$candidate"; return 0; }
	fi

	for candidate in /opt/rocm /opt/rocm-*; do
		valid_rocm_root "$candidate" && { printf '%s\n' "$candidate"; return 0; }
	done

	return 1
}

detect_vulkan_path() {
	local candidate=""

	for candidate in "${VULKAN_SDK:-}"; do
		[ -n "$candidate" ] || continue
		valid_vulkan_root "$candidate" && { printf '%s\n' "$candidate"; return 0; }
	done

	for candidate in /usr /usr/local /opt/vulkan-sdk /opt/vulkan; do
		valid_vulkan_root "$candidate" && { printf '%s\n' "$candidate"; return 0; }
	done

	if [ -f /usr/include/vulkan/vulkan.h ]; then
		printf '%s\n' '/usr'
		return 0
	fi

	return 1
}

normalize_cuda_arch_for_rules_cuda() {
	local arch="$1"

	case "$arch" in
	30|32|35|37|50|52|53|60|61|62|70|72|75|80|86|87|89|90)
		printf 'sm_%s\n' "$arch"
		return 0
		;;
	esac

	# Older rules_cuda releases in this repo currently top out at sm_90.
	if [ "$arch" -gt 90 ] 2>/dev/null; then
		printf '%s\n' 'sm_90'
		return 0
	fi

	return 1
}

detect_cuda_bazel_archs() {
	local archs=""
	local raw_archs=""

	if [ -n "${CUDA_BAZEL_ARCHS:-}" ]; then
		printf '%s\n' "$CUDA_BAZEL_ARCHS"
		return 0
	fi

	if command -v nvidia-smi >/dev/null 2>&1; then
		raw_archs="$(
			nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
				| awk '
					NF {
						gsub(/[^0-9]/, "", $1)
						if (length($1) == 1) {
							$1 = $1 "0"
						}
						print $1
					}
					' \
					| awk '!seen[$0]++'
			)"

		archs="$(
			while IFS= read -r arch; do
				normalize_cuda_arch_for_rules_cuda "$arch" || true
			done <<<"$raw_archs" \
				| awk '!seen[$0]++' \
				| paste -sd ';' -
		)"
	fi

	if [ -n "$archs" ]; then
		printf '%s\n' "$archs"
		return 0
	fi

	printf '%s\n' 'sm_53;sm_62;sm_70;sm_72;sm_75;sm_80;sm_86;sm_87;sm_89;sm_90'
}

ensure_cuda_env() {
	local cuda_root=""

	cuda_root="$(detect_cuda_path)" || return 1
	export CUDA_PATH="$cuda_root"
	append_path_once "$cuda_root/bin"
}

ensure_rocm_env() {
	local rocm_root=""

	rocm_root="$(detect_rocm_path)" || return 1
	export ROCM_PATH="$rocm_root"
	export HIP_PATH="$rocm_root"
	append_path_once "$rocm_root/bin"
}

ensure_vulkan_env() {
	local vk_root=""

	vk_root="$(detect_vulkan_path)" || return 1
	export VULKAN_SDK="$vk_root"
	append_path_once "$vk_root/bin"
}

ensure_cuda_bazel_archs() {
	local archs=""

	archs="$(detect_cuda_bazel_archs)" || return 1
	export CUDA_BAZEL_ARCHS="$archs"
}
