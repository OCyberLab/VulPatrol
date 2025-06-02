#!/usr/bin/env python3
"""
compile_tool.py

– Strips out all C/C++ comments from the source file.
– Accepts include directories for Clang in case you have project‐local headers.
– Wraps `clang` (or `clang++`) to emit LLVM IR (`.ll`).

Usage:
    python3 scripts/compile_tool.py \
      --input path/to/foo.c \
      --output path/to/foo.ll \
      [--opt-level 0|1|2|3] \
      [--include_dirs dir1 dir2 ...] \
      [--cxx]

Examples:
    # Simple C → IR, no extra include dirs, optimization = 0:
    python3 scripts/compile_tool.py \
      --input examples/toy/foo.c \
      --output work/llvm_ir/foo.ll

    # C++ → IR with -O2 and two include paths:
    python3 scripts/compile_tool.py \
      --input examples/toy/foo.cpp \
      --output work/llvm_ir/foo.ll \
      --opt-level 2 \
      --include_dirs /usr/local/include /home/me/project/includes \
      --cxx
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

def strip_comments(code: str) -> str:
    """
    Remove all C‐style comments (/* … */) and C++‐style comments (// …) from code.
    Returns the “cleaned” string.
    """
    # Pattern for C++‐style comments (// … to end‐of‐line)
    cpp_comment_pattern = r"//.*?$"
    # Pattern for C‐style comments (/* … */)
    c_comment_pattern = r"/\*.*?\*/"
    # First remove /* … */ (multiline), then remove // … (line comments)
    no_c_comments   = re.sub(c_comment_pattern, "", code, flags=re.DOTALL)
    no_cpp_comments = re.sub(cpp_comment_pattern, "", no_c_comments, flags=re.MULTILINE)
    return no_cpp_comments

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile a C/C++ source file into stripped, comment‐free LLVM IR (.ll)."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input .c or .cpp file."
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Desired path for the output .ll file."
    )
    parser.add_argument(
        "--opt-level", "-O",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Clang optimization level (0–3). Default = 0."
    )
    parser.add_argument(
        "--include_dirs", "-I",
        nargs="*",
        default=[],
        help="Zero or more include directories for Clang to search (equivalent to -I)."
    )
    parser.add_argument(
        "--cxx",
        action="store_true",
        help="If set, use clang++ instead of clang (for C++ code)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    src_path = args.input
    ll_path  = args.output
    opt_lvl  = args.opt_level
    inc_dirs = args.include_dirs
    use_cxx  = args.cxx

    # 1) Validate input exists
    if not os.path.isfile(src_path):
        print(f"✖ Error: Input file '{src_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # 2) Read original source, strip comments, write to a temporary file
    with open(src_path, "r", encoding="utf8", errors="ignore") as f_in:
        original_code = f_in.read()

    code_no_comments = strip_comments(original_code)

    tmp_dir = tempfile.mkdtemp(prefix="compile_tool_")
    tmp_source = os.path.join(tmp_dir, os.path.basename(src_path))
    # Write stripped‐comment code to tmp_source
    with open(tmp_source, "w", encoding="utf8") as f_tmp:
        f_tmp.write(code_no_comments)

    # 3) Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(ll_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 4) Build the clang command
    compiler = "clang++" if use_cxx else "clang"

    # Collect -I flags; always at least -I.
    if len(inc_dirs) == 0:
        # If the user did not supply any include dirs, add the current directory by default
        inc_dirs = ["."]
    clang_includes = []
    for d in inc_dirs:
        clang_includes.extend(["-I", d])

    cmd = [
        compiler,
        "-S",                    # Emit human‐readable output
        "-emit-llvm",            # Emit LLVM IR
        f"-O{opt_lvl}",          # Optimization level
        tmp_source,              # Temporarily stripped source
        "-o",
        ll_path
    ] + clang_includes

    # 5) Run clang
    print("▶ Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("✖ clang failed with exit code", e.returncode, file=sys.stderr)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(e.returncode)

    print(f"✔ Successfully wrote LLVM IR to: {ll_path}")

    # 6) Clean up the temporary directory
    shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
