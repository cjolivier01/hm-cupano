#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_scoped_ids(vcd_path: Path, scope_path: str) -> tuple[str, str]:
    scope = []
    host_rd_valid_id = None
    host_rd_data_id = None
    wanted = tuple(part for part in scope_path.split("/") if part)

    with vcd_path.open("r", encoding="ascii", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("$scope"):
                parts = line.split()
                if len(parts) >= 3:
                    scope.append(parts[2])
            elif line.startswith("$upscope"):
                if scope:
                    scope.pop()
            elif line.startswith("$var") and tuple(scope) == wanted:
                parts = line.split()
                if len(parts) >= 5:
                    var_id = parts[3]
                    name = parts[4]
                    if name == "host_rd_valid":
                        host_rd_valid_id = var_id
                    elif name == "host_rd_data":
                        host_rd_data_id = var_id
            elif line.startswith("$enddefinitions"):
                break

    if host_rd_valid_id is None or host_rd_data_id is None:
        raise SystemExit(f"Unable to find host_rd_valid/host_rd_data under scope {scope_path} in {vcd_path}")
    return host_rd_valid_id, host_rd_data_id


def parse_vector(value: str) -> int:
    lowered = value.lower()
    if any(ch in lowered for ch in "xz"):
        raise SystemExit(f"Unsupported X/Z value in VCD vector: {value}")
    return int(lowered, 2)


def extract_readback_words(vcd_path: Path, scope_path: str) -> list[int]:
    valid_id, data_id = parse_scoped_ids(vcd_path, scope_path)
    words: list[int] = []
    current_valid = 0
    current_data = 0
    pending_valid: int | None = None
    pending_data: int | None = None
    seen_time = False

    def flush_time() -> None:
        nonlocal current_valid, current_data, pending_valid, pending_data
        prev_valid = current_valid
        if pending_valid is not None:
            current_valid = pending_valid
        if pending_data is not None:
            current_data = pending_data
        if prev_valid == 0 and current_valid == 1:
            words.append(current_data & 0xFFFFFFFF)
        pending_valid = None
        pending_data = None

    with vcd_path.open("r", encoding="ascii", errors="replace") as f:
        in_dump = False
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("$dumpvars"):
                in_dump = True
                continue
            if line.startswith("$end"):
                if in_dump:
                    in_dump = False
                continue
            if line.startswith("$"):
                continue
            if line.startswith("#"):
                if seen_time:
                    flush_time()
                seen_time = True
                continue
            if line[0] in "01xz":
                value = line[0]
                var_id = line[1:]
                if var_id == valid_id:
                    if value in "xz":
                        raise SystemExit("Unsupported X/Z value on host_rd_valid")
                    pending_valid = int(value)
            elif line[0] in "bBrR":
                parts = line.split()
                if len(parts) != 2:
                    continue
                value = parts[0][1:]
                var_id = parts[1]
                if var_id == data_id:
                    pending_data = parse_vector(value)

    if seen_time:
        flush_time()
    return words


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract GEM readback words from output.vcd")
    parser.add_argument("--vcd", required=True)
    parser.add_argument("--out-hex", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--scope", default="pano_two_image_assets_tb/uut")
    args = parser.parse_args()

    words = extract_readback_words(Path(args.vcd), args.scope)
    if len(words) != args.expected_count:
        raise SystemExit(f"Expected {args.expected_count} readback words, found {len(words)} in {args.vcd}")
    Path(args.out_hex).write_text("\n".join(f"{word:08x}" for word in words) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
