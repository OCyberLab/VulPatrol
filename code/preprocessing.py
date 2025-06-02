
import re
from typing import List, Dict

# --------------------------
# Token abstraction patterns
# --------------------------
LOCAL_ID = r"%[a-zA-Z_][a-zA-Z0-9_]*"
GLOBAL_ID = r"@[a-zA-Z_][a-zA-Z0-9_]*"
LABEL_LINE = r"; <label>:\d+:"
FLOAT_HEX = r"0x[0-9a-fA-F]+\.[0-9a-fA-F]+p[+-]?[0-9]+"
FLOAT_SCI = r"[+-]?[0-9]+\.[0-9]+e[+-]?[0-9]+"
INT = r"(?<!align)(?<!\\[) \\d+"
STRING = r'\"[^\"]*\"'

# --------------------------
# Preprocess a single line
# --------------------------
def preprocess_statement_joern(stmt: str) -> str:
    """Preprocess a single LLVM IR instruction line using inst2vec-style abstraction."""
    stmt = re.sub(LOCAL_ID, "<%ID>", stmt)
    stmt = re.sub(GLOBAL_ID, "<@ID>", stmt)

    if re.match(LABEL_LINE, stmt):
        stmt = re.sub(r":\d+", ":<LABEL>", stmt)
        stmt = re.sub("<%ID>", "<LABEL>", stmt)

    stmt = re.sub(FLOAT_HEX, "<FLOAT>", stmt)
    stmt = re.sub(FLOAT_SCI, "<FLOAT>", stmt)
    stmt = re.sub(INT, " <INT>", stmt)
    stmt = re.sub(STRING, "<STRING>", stmt)
    stmt = re.sub(r'\\b(i[0-9]+|float|double|void|char|char\\*|bool)\\b', '<TYPE>', stmt)

    stmt = stmt.strip()
    stmt = re.sub(r'\\s+', ' ', stmt)

    return stmt

# --------------------------
# Struct type extraction
# --------------------------
def GetStructTypes(ir: str) -> Dict[str, str]:
    """Extract a dictionary of struct definitions from the given IR."""
    try:
        lines = ir.split("\n")
        struct_defs = {}
        for line in lines:
            if "= type" in line and "{" in line:
                parts = line.split("=", 1)
                name = parts[0].strip()
                definition = parts[1].split("type", 1)[-1].strip()
                struct_defs[name] = definition
        return struct_defs
    except Exception as e:
        raise ValueError(f"Error extracting struct types: {e}")

# --------------------------
# Full preprocessing
# --------------------------
def preprocess_ir_lines(ir_lines: List[str], inline_structs: bool = True) -> List[str]:
    """Preprocess LLVM IR lines to inst2vec-ready format."""
    if inline_structs:
        try:
            structs = GetStructTypes("\n".join(ir_lines))
            for i, line in enumerate(ir_lines):
                for struct, definition in structs.items():
                    ir_lines[i] = line.replace(struct, definition)
        except ValueError:
            pass

    return [preprocess_statement_joern(line) for line in ir_lines if line.strip()]

def remove_leading_spaces(data: List[str]) -> List[str]:
    \"\"\"Remove leading whitespace from each line.\"\"\"
    return [line.strip() for line in data]


def remove_trailing_comments_and_metadata(data: List[str]) -> List[str]:
    \"\"\"Remove metadata/comments at the end of LLVM IR lines.\"\"\"
    cleaned = []
    for line in data:
        line = re.sub(r",? metadata !\\d+", "", line)
        line = re.sub(r"!dbg !\\d+", "", line)
        line = re.sub(r"#\\d+", "", line)  # Remove attribute references like #1
        line = re.sub(r"!tbaa !\\d+", "", line)
        cleaned.append(line.strip())
    return cleaned


def remove_structure_definitions(data: List[str]) -> List[str]:
    \"\"\"Remove struct definitions like %name = type {...}\"\"\"
    return [line for line in data if not re.match(r"%.* = type .*", line)]


def preprocess_ir_lines_extended(ir_lines: List[str], inline_structs: bool = True) -> List[str]:
    \"\"\"Full preprocessing pipeline: clean, inline structs, tokenize.\"\"\"
    ir_lines = remove_leading_spaces(ir_lines)
    ir_lines = remove_trailing_comments_and_metadata(ir_lines)
    if inline_structs:
        try:
            structs = GetStructTypes("\\n".join(ir_lines))
            for i, line in enumerate(ir_lines):
                for struct, definition in structs.items():
                    ir_lines[i] = ir_lines[i].replace(struct, definition)
        except ValueError:
            pass
    ir_lines = remove_structure_definitions(ir_lines)
    return [preprocess_statement_joern(line) for line in ir_lines if line.strip()]
