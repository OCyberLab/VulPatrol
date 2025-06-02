# VulPatrol: Graph-Based Vulnerability Detection Pipeline

This repository contains a complete, Bazel-driven pipeline for:
1. **Slicing** C/C++ source files (optional)  
2. **Compiling** each slice to LLVM IR  
3. **Converting** LLVM IR → Code Property Graph (CPG) using Joern  
4. **Extracting** per-method JSON subgraphs  
5. **Preprocessing** / encoding node features (e.g. Inst2Vec)  
6. **Training** a Message-Passing Neural Network (MPNN) on those JSON graphs  
7. **Evaluating** classification or regression metrics (F₁, Precision/Recall, MAE/MSE, etc.)

---

## 1. Project Overview

Many vulnerability-detection efforts rely on graph representations of programs (CPGs, ASTs, DFGs, etc.) and Graph Neural Networks (GNNs). This repository provides a complete pipeline that starts from raw C/C++ source, slices it (if desired), compiles to LLVM IR, uses **Joern** to generate a Code Property Graph, extracts each method’s subgraph into a JSON file, preprocesses/encodes node features (e.g. Inst2Vec-like embeddings), and finally trains an MPNN to detect vulnerabilities (or perform regression).  

- **Slice → Compile**. If you only care about one function (e.g. `vuln_func()`), our `slice_tool` will extract it into its own `.c` file. Then we call Clang to produce `.ll` (LLVM IR).  
- **LLVM IR → CPG**. We wrap ProGraML’s `llvm2cpg` (via Bazel) to convert `foo.ll` → `foo.pb` (a CPG protobuf).  
- **CPG → JSON**. We use a Joern-CLI Scala script to walk through each method node in the CPG and emit one `method_<ID>.json` file containing nodes + edges + attributes.  
- **Preprocessing / Encoding**. Our Python scripts (`preprocessing.py`, `inst2vec_encoder.py`, `rgx_utils.py`) parse those JSON files, extract “Inst2Vec”‐style vectors for each LLVM statement, add structural features (in/out-degrees, edge types), and produce a final serialized dataset ready for training.  
- **Train MPNN**. `mpnn_train.py` is a PyTorch‐based training driver that builds a Dataset/DataLoader from the preprocessed JSONs, constructs an MPNN with configurable layers, and runs a full training loop.  

All of these steps are orchestrated in Bazel so that you can cache intermediate artifacts, parallelize builds, and ensure reproducibility. A single “umbrella” `run_pipeline.sh` script (wrapped as a Bazel `sh_binary`) shows how to invoke each stage in the proper order.

---

## 2. Prerequisites

The following packages need to be installed and available on your `\$PATH`:

1. **Bazel** (v3.7.0 or higher)  
2. **Clang/LLVM** (tested with Clang 10.0+)  
   - Make sure `clang` (for compiling .c → .ll) is available.  
3. **Joern** (v1.1.150 or higher)  
   - You need the `joern-cli` executable.  
   - If `joern-cli` is not on your system PATH, set an environment variable:  

      ```bash
     export JOERN_CLI=/absolute/path/to/joern-cli
     ```
      
4. **Python 3.7+** (with pip)  
   - We assume you’ll install Python dependencies (PyTorch, NumPy, NetworkX, etc.) inside a virtual environment or via Bazel’s `pip_import`.  
5. **ProGraML’s `llvm2cpg`** (either cloned locally or vendored into Bazel)  
   - We expect a Bazel target `//src/llvm2cpg:llvm2cpg_sh` that wraps a compiled binary called `llvm2cpg_bin`.

---


## 3. Run
# 1) (Optional) Slice a function from a large file:
python3 scripts/slice_tool.py \
  --input=examples/toyproj/foo.c \
  --function_name="vuln_func" \
  --output=work/slices/foo_vuln.c

# 2) Compile to LLVM IR:
python3 scripts/compile_tool.py \
  --input=work/slices/foo_vuln.c \
  --output=work/llvm_ir/foo_vuln.ll \
  --opt-level=0

# 3) Build & run llvm2cpg (via Bazel):
mkdir -p work/cpg_proto
bazel run //src/llvm2cpg:llvm2cpg_sh -- \
  --input_ll=work/llvm_ir/foo_vuln.ll \
  --output_pb=work/cpg_proto/foo_vuln.cpg.pb

# 4) Import CPG into Joern:
mkdir -p work/joern_db
joern-cli --import-cpg work/cpg_proto/foo_vuln.cpg.pb --out work/joern_db

# 5) Dump Call Graph JSONs:
mkdir -p work/json_call_graphs
joern-cli --run-script scripts/extractCallGraph.sc \
  --input=work/joern_db \
  --output=work/json_call_graphs

# 6) Dump CPG JSONs (nodes + edges):
mkdir -p work/json_cpgs
joern-cli --run-script code/extractCPG.sc \
  --input=work/joern_db \
  --output=work/json_cpgs

# 7) (Optional) Annotate CWE IDs:
mkdir -p work/preprocessed_graphs_cwe
python3 code/cwe_map.py \
  --input_graphs=work/preprocessed_graphs \
  --output_graphs=work/preprocessed_graphs_cwe \
  --cwe_db=cwe_list.json

# 8) Preprocess JSON → PyTorch graphs:
mkdir -p work/preprocessed_graphs
python3 code/preprocessing.py \
  --input_json_dir=work/json_cpgs \
  --inst2vec_vocab=pretrained_inst2vec_vocab.pt \
  --output_dir=work/preprocessed_graphs

# 9) Train & evaluate MPNN (reports F1 on test set):
mkdir -p work/checkpoints
python3 code/mpnn_train.py \
  --data_dir=work/preprocessed_graphs \
  --epochs=300 \
  --batch_size=128 \
  --lr=0.0001 \
  --num_passes=6 \
  --hidden_dim=200 \
  --read_hidden_dim=512 \
  --dropout=0.2 \
  --early_stopping_patience=10 \
  --task=classification \
  --checkpoint_dir=work/checkpoints \
  --device=cuda
