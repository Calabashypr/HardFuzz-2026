#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "Usage: $0 JSON_PATH [OUTDIR]"
  exit 1
fi
JSON_PATH="$1"
OUTDIR="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$JSON_PATH" ]; then
  echo "JSON not found: $JSON_PATH"
  exit 1
fi
BASENAME="$(basename "$JSON_PATH")"
NAME="${BASENAME%.*}"
PROGRAM_CPP="$SCRIPT_DIR/program_${NAME}.cpp"
echo $SCRIPT_DIR

python3 "$SCRIPT_DIR/tools/prepare_graph.py" "$JSON_PATH" "$PROGRAM_CPP.graph.json"
JSON_PATH="$PROGRAM_CPP.graph.json"

python3 "$SCRIPT_DIR/execute_graph.py" --json "$JSON_PATH" --shape-manifest "$PROGRAM_CPP.tensor_lengths.json" --dtype-manifest "$PROGRAM_CPP.dtypes.json" | tee "$PROGRAM_CPP"
COPY_DIR="${OUTDIR:-$SCRIPT_DIR/baremix_${NAME}}"
rm -rf "$COPY_DIR"
cp -r "$SCRIPT_DIR/baremix" "$COPY_DIR"
cp "$PROGRAM_CPP.tensor_lengths.json" "$COPY_DIR/tensor_lengths.json"
cp "$PROGRAM_CPP.dtypes.json" "$COPY_DIR/dtypes.json"
cp "$JSON_PATH" "$COPY_DIR/graph.json"

read MATMUL_COUNT MMAD_COUNT <<EOF
$(python3 - "$JSON_PATH" << 'PY'
import sys, json
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
ops = [str(n.get("op","")) for n in data.get("nodes",[])]
def cnt(name):
    return sum(1 for o in ops if o.lower()==name.lower())
print(cnt("MatMul"), cnt("Mmad"))
PY
)
EOF
if [ "${MATMUL_COUNT:-0}" = "0" ] && [ "${MMAD_COUNT:-0}" = "1" ]; then

  REF_CPP="${SCRIPT_DIR}/baremix/baremix_custom.cpp"
  if [ -f "$REF_CPP" ]; then
  PY_SCRIPT=$(cat << 'PY'
import sys,io,os
dst = sys.argv[1]
src = sys.argv[2]
L1 = int(sys.argv[3]); L2 = int(sys.argv[4])
with open(src, "r", encoding="utf-8") as f:
    s_lines = f.readlines()
snippet = "".join(s_lines[L1-1:L2])
with open(dst, "r", encoding="utf-8") as f:
    d_lines = f.readlines()

start = None
for i,ln in enumerate(d_lines):
    if "if ASCEND_IS_AIC" in ln and "{" in ln:
        start = i
        break
if start is None:
    sys.exit(0)
depth = 0
end = None
for j in range(start, len(d_lines)):
    ln = d_lines[j]
    depth += ln.count("{")
    depth -= ln.count("}")
    if depth==0 and j>start:
        end = j
        break
if end is None or end <= start+1:
    sys.exit(0)
inner_first = start+1
inner_last = end-1

head = d_lines[:inner_first+1]
tail = d_lines[inner_last:]
out = head + [snippet if snippet.endswith("\n") else snippet+"\n"][0:1] + tail
with open(dst, "w", encoding="utf-8") as f:
    f.writelines(out)
PY
)
  python3 - <<PY "$COPY_DIR/baremix_custom.cpp" "$REF_CPP" 532 534
$PY_SCRIPT
PY
  fi
fi
if [ "${MATMUL_COUNT:-0}" = "0" ] && [ "${MMAD_COUNT:-0}" = "0" ]; then
  python3 - "$COPY_DIR/baremix_custom.cpp" <<'PY_REMOVE_AIC'
from pathlib import Path
import sys
p = Path(sys.argv[1]); text = p.read_text()
a = text.index("if ASCEND_IS_AIC")
b = text.index("{", a); depth = 1; end = b + 1
while depth:
    depth += (text[end] == "{") - (text[end] == "}")
    end += 1
p.write_text(text[:b+1] + "\n        // No matrix invocation in this graph.\n    " + text[end-1:])
PY_REMOVE_AIC
fi
if [ "${MATMUL_COUNT:-0}" -ge 1 ]; then
  PY_AIC=$(cat << 'PY'
import sys
import re
dst = sys.argv[1]
K = int(sys.argv[2])
matmul_src = sys.argv[3]
with open(matmul_src, "r", encoding="utf-8") as f:
    matmul_lines = f.readlines()
with open(dst, "r", encoding="utf-8") as f:
    d_lines = f.readlines()
start = None
for i,ln in enumerate(d_lines):
    if "if ASCEND_IS_AIC" in ln and "{" in ln:
        start = i
        break
if start is None:
    sys.exit(0)
depth = 0
end = None
for j in range(start, len(d_lines)):
    ln = d_lines[j]
    depth += ln.count("{")
    depth -= ln.count("}")
    if depth==0 and j>start:
        end = j
        break
if end is None or end <= start+1:
    sys.exit(0)
search_region_start = start + 1
search_region_end = end

matmul_decl_start = None
for j in range(search_region_start, search_region_end):
    if "Matmul<" in d_lines[j]:
        matmul_decl_start = j
        break

matmul_end = None
for j in range(search_region_start, search_region_end):
    if "matmulObj.End()" in d_lines[j]:
        matmul_end = j
        break

if matmul_decl_start is None:

    matmul_decl_start = search_region_start
if matmul_end is None:
    matmul_end = search_region_end - 1

for j in range(matmul_end, search_region_end):
    if "CrossCoreSetFlag" in d_lines[j]:
        matmul_end = j
        break

indent = re.match(r"^(\s*)", d_lines[matmul_decl_start]).group(1) if d_lines[matmul_decl_start] else "        "
snippet = []
for i in range(K):
    sid_wait = 1 + 2*i + 1
    sid_set = 1 + 2*i + 2
    snippet.append(f"{indent}// MatMul #{i+1}")
    snippet.append(f"{indent}AscendC::CrossCoreWaitFlag({sid_wait});")
    matmul_obj_var = "matmulObj_%d" % (i + 1)
    for ln in matmul_lines:
        snippet.append(ln.rstrip("\n").replace("matmulObj", matmul_obj_var))
    snippet.append(f"{indent}AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>({sid_set});")

head = []
for ln in d_lines[:matmul_decl_start]:
    if ("CrossCoreWaitFlag" in ln) or ("CrossCoreSetFlag" in ln):
        continue
    head.append(ln)
tail = d_lines[matmul_end+1:]
out = head + [ln + ("\n" if not ln.endswith("\n") else "") for ln in snippet] + tail
with open(dst, "w", encoding="utf-8") as f:
    f.writelines(out)
PY
)
  python3 - <<PY "$COPY_DIR/baremix_custom.cpp" "$MATMUL_COUNT" "$SCRIPT_DIR/ops/MatMul.h"
$PY_AIC
PY
fi
python3 "$SCRIPT_DIR/split_cpp_sections.py" "$PROGRAM_CPP" "$SCRIPT_DIR" "program_${NAME}" "$COPY_DIR/baremix_custom.cpp" "$JSON_PATH"
GEN_CPP="$COPY_DIR/baremix_custom_program_out.cpp"
if [ ! -s "$GEN_CPP" ]; then
  echo "Failed to assemble AscendC program: $GEN_CPP" >&2
  exit 1
fi
mv "$COPY_DIR/baremix_custom.cpp" "$COPY_DIR/baremix_custom.cpp.bak"
mv "$GEN_CPP" "$COPY_DIR/baremix_custom.cpp"
echo "$COPY_DIR"
