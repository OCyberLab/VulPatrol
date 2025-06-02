#!/usr/bin/env python3
"""
cwe_map.py

Scan an LLVM‐IR (.ll) file (compiled with -g) for calls to known-unsafe
functions, map each occurrence to a CWE ID, and—when debug info is present—
print out the original C/C++ source file and line number.

Usage:
  python3 cwe_map.py --input work/llvm_ir/foo.ll [--output results.json]

If compiled with `-g`, the LLVM IR will contain `!DILocation` metadata that
indicates the original filename and line.  We do a two‐pass parse:
  1) Build a map fileID → (filename, directory)
  2) Scan for call instructions to unsafe functions, grab their `!dbg !N`
     tags, look up `!N` in the `!DILocation(...)` tables, and then
     look up the fileID to reconstruct “file:line”.

If no debug info is present, we still report the function name and the raw
IR line number (within the .ll file itself).  Output is printed to STDOUT
and optionally written to `--output <path>` in JSON form.
"""

import re
import json
import argparse
import os
import sys

# -----------------------------------------------------------------------------
# 1) A minimal “unsafe function → CWE ID” mapping
#
#    We include just a few examples; feel free to add more as needed.
#    In practice, you might augment this dict with any library calls you
#    care about (e.g. gets → CWE-120, strcpy → CWE-119, sprintf → CWE-119, etc.)
#
UNSAFE_FN_TO_CWE = {
    "gets":         "CWE-120",  # Buffer copy without checking size
    "strcpy":       "CWE-119",  # Buffer overrun
    "strncpy":      "CWE-119",  # If misused (no NUL-terminator)
    "sprintf":      "CWE-119",  # Buffer overrun
    "vsprintf":     "CWE-119",
    "snprintf":     "CWE-119",  # still risky if length is wrong
    "strcat":       "CWE-119",
    "strncat":      "CWE-119",
    "memcpy":       "CWE-119",  # If length unchecked
    "memmove":      "CWE-119",
    "fgets":        "CWE-119",  # can still be dangerous if used incorrectly
    "scanf":        "CWE-119",  # without width specifiers
    "sscanf":       "CWE-119",
    "strtok":       "CWE-125",  # out‐of bounds if misused
    "atoi":         "CWE-190",  # Integer overflow/underflow
    "atol":         "CWE-190",
    "atoll":        "CWE-190",
    "strtol":       "CWE-190",
    "strtoul":      "CWE-190",
    # …and so on.  Add any others you need.
}

# -----------------------------------------------------------------------------
# 2) Regex patterns to find “call” instructions and debug metadata
#
#    We look for lines like:
#       %3 = call i8* @strcpy(i8* %dst, i8* %src), !dbg !42
#
#    So we match “call … @<fn> … !dbg !<dbgID>”.
#
#    Then, elsewhere in the same .ll we will see:
#      !42 = !DILocation(line: 123, column: 5, scope: !15)
#
#    And a “!15 = !DISubprogram(… file: !8, line: … )”
#    and “!8 = !DIFile(filename: "code/foo.cpp", directory: "/home/user/proj")”
#
CALL_PATTERN = re.compile(
    r"""
    ^\s*                             # optional leading whitespace
    %?[\w\d\.]+ \s* = \s*            # some SSA result (or ignore “%?”)
    call\b[^@]*@                     # the “call” keyword then “@”
    (?P<fnname>[\w\d_]+)             # capture function name (e.g. strcpy)
    [^(]*\([^)]*\)                   # arguments in parentheses
    (?:,[^!]*?)?                     # possibly more commas before debug info
    \s*!dbg\s*! (?P<dbgid>\d+)       # “!dbg !42” → dbgid=42
    """,
    re.VERBOSE,
)

# Match a line like:   !42 = !DILocation(line: 123, column: 5, scope: !15)
DILOC_PATTERN = re.compile(
    r"""
    ^\s*!(?P<locid>\d+)\s*=\s*!DILocation
      \(\s*line:\s*(?P<line>\d+),\s*column:\s*\d+,\s*scope:\s*!(?P<scopeid>\d+)\s*\)
    """,
    re.VERBOSE,
)

# Match a line like:   !8 = !DIFile(filename: "foo.cpp", directory: "/home/user/proj")
DIFILE_PATTERN = re.compile(
    r"""
    ^\s*!(?P<fileid>\d+)\s*=\s*!DIFile
      \(\s*filename:\s*"(?P<fname>[^"]+)"\s*,\s*directory:\s*"(?P<dir>[^"]+)"\s*\)
    """,
    re.VERBOSE,
)

# Match a line like:  !15 = !DISubprogram(name: "...", scope: !<someID>, file: !8, line: 42, …)
# We want to capture “!15” and which file‐ID it references (“file: !8”).
DISUB_PATTERN = re.compile(
    r"""
    ^\s*!(?P<scopeid>\d+)\s*=\s*!DISubprogram
      \([^)]*?file:\s*!(?P<fileid>\d+)\s*,\s*line:\s*(?P<decl_line>\d+)
    """,
    re.VERBOSE,
)

def parse_args():
    p = argparse.ArgumentParser(description="Map unsafe function calls in LLVM IR → CWE + source line")
    p.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the LLVM‐IR (.ll) file (must be compiled with -g)."
    )
    p.add_argument(
        "--output", "-o",
        help="Optional JSON output file.  If omitted, prints only to STDOUT."
    )
    return p.parse_args()

def main():
    args = parse_args()
    ll_path = args.input

    if not os.path.isfile(ll_path):
        print(f"ERROR: Cannot read '{ll_path}'", file=sys.stderr)
        sys.exit(1)

    # Step 1: Read all lines of the .ll file
    with open(ll_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    # Step 2: Two‐pass parse → build metadata maps
    #
    #   → fileid_map:   { fileID (int)  →  (filename, directory) }
    #   → scope_map:    { scopeID (int) → fileID (int) }
    #   → loc_map:      { locID   (int) →  (line_number (int), scopeID (int)) }
    #
    fileid_map = {}
    scope_map  = {}
    loc_map    = {}

    for ln in lines:
        # 2a) Look for !DIFile entries
        m_file = DIFILE_PATTERN.match(ln)
        if m_file:
            fid = int(m_file.group("fileid"))
            fname = m_file.group("fname")
            dname = m_file.group("dir")
            fileid_map[fid] = (fname, dname)
            continue

        # 2b) Look for !DISubprogram entries (maps scopeID → fileID)
        m_sub = DISUB_PATTERN.match(ln)
        if m_sub:
            scopeid = int(m_sub.group("scopeid"))
            fileid  = int(m_sub.group("fileid"))
            scope_map[scopeid] = fileid
            continue

        # 2c) Look for !DILocation entries (maps locID → (line, scopeID))
        m_loc = DILOC_PATTERN.match(ln)
        if m_loc:
            lid = int(m_loc.group("locid"))
            line_no = int(m_loc.group("line"))
            scopeid = int(m_loc.group("scopeid"))
            loc_map[lid] = (line_no, scopeid)
            continue

    # Step 3: Scan again for “call” instructions that reference UNSAFE_FN_TO_CWE
    results = []
    for idx, ln in enumerate(lines):
        m_call = CALL_PATTERN.match(ln)
        if not m_call:
            continue

        fnname = m_call.group("fnname")
        dbgid  = int(m_call.group("dbgid"))

        # Is this function in our “unsafe” dictionary?
        if fnname not in UNSAFE_FN_TO_CWE:
            continue

        # We found an unsafe call.
        cwe_id = UNSAFE_FN_TO_CWE[fnname]

        # By default, record the IR‐line number if we cannot find debug → C++ location
        source_file = None
        source_line = None

        # If we have debug info "dbgid" → look up loc_map to get (lineNo, scopeID)
        if dbgid in loc_map:
            line_no, scopeid = loc_map[dbgid]
            if scopeid in scope_map:
                fileid = scope_map[scopeid]
                if fileid in fileid_map:
                    fname, dname = fileid_map[fileid]
                    source_file = os.path.join(dname, fname)
                    source_line = line_no

        result = {
            "ir_line_index": idx + 1,      # 1-based index into the .ll file
            "called_fn": fnname,           # e.g. “strcpy”
            "cwe_id": cwe_id,              # e.g. “CWE-119”
            "source_file": source_file,    # full path or None
            "source_line": source_line     # integer or None
        }
        results.append(result)

    # Step 4: Print to STDOUT + optionally write JSON
    print("Found {} unsafe‐function call site(s):\n".format(len(results)))
    for r in results:
        if r["source_file"] and r["source_line"]:
            print(f"  - {r['called_fn']}  →  {r['cwe_id']}  @  "
                  f"{r['source_file']}:{r['source_line']}")
        else:
            print(f"  - {r['called_fn']}  →  {r['cwe_id']}  (no debug→C++ info; IR line {r['ir_line_index']})")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, indent=2)
        print(f"\nWrote JSON results to: {args.output}")

if __name__ == "__main__":
    main()
