import argparse
import hashlib
import json
import os
import re
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


_WRAPPER_PRIORITY_FIRST = frozenset({"GenerateRandomMask.cpp"})

def _extract_aicore_signatures(cpp: str) -> List[str]:

    pat = re.compile(r"__aicore__\s+inline\s+[\s\S]*?\)\s*\{")
    keys: List[str] = []
    for m in pat.finditer(cpp):
        s = m.group(0)

        s = re.sub(r"\{\s*$", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        keys.append(s)
    return keys

def load_all_wrapper_codes(selected_wrapper_stems: Optional[set] = None) -> List[str]:

    base = Path(__file__).resolve().parent
    wdir = base / "wrappers"
    if not wdir.is_dir():
        raise FileNotFoundError(
            f"Wrapper directory not found: {wdir}. "
            "Restore the wrappers directory from the distribution."
        )
    paths_all = list(wdir.glob("*.cpp"))
    if selected_wrapper_stems is not None:
        selected_wrapper_stems = set(selected_wrapper_stems)
        paths_all = [p for p in paths_all if p.stem in selected_wrapper_stems]
    paths = sorted(paths_all, key=lambda p: (0 if p.name in _WRAPPER_PRIORITY_FIRST else 1, p.name))
    if not paths:
        if selected_wrapper_stems is None:
            raise FileNotFoundError(f"No .cpp wrappers in {wdir}")
        return []
    seen_sigs: set = set()
    chunks: List[str] = []
    skipped_files: List[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Cannot read wrapper file {path}: {exc}") from exc
        sigs = _extract_aicore_signatures(text)
        if sigs and any(s in seen_sigs for s in sigs):
            skipped_files.append(path.name)
            continue
        for s in sigs:
            seen_sigs.add(s)
        if not text.endswith("\n"):
            text += "\n"
        chunks.append(text)

    if skipped_files and os.environ.get("WRAPPER_DEDUP_VERBOSE"):
        import sys

        for name in skipped_files:
            print(f"[wrappers] Skipping duplicate signature: {name}", file=sys.stderr)
    return chunks

def _env_on(name: str) -> bool:
    v = os.environ.get(name, "")
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

def sanitize_op(name: Optional[str]) -> str:
    if not name:
        return "nop"
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "nop"

_NATIVE_DTYPE_ARITY = {"Add": 2, "Sub": 2, "Mul": 2, "Max": 2, "Min": 2,
                       "Abs": 1, "Relu": 1, "Exp": 1}
_NATIVE_INT32_OPS = frozenset({"Add", "Sub", "Max", "Min", "Relu"})

def interface_dtypes(op: str) -> List[str]:
    if op not in _NATIVE_DTYPE_ARITY:
        return ["float32"]
    return ["float32", "float16"] + (["int32"] if op in _NATIVE_INT32_OPS else [])

def typed_interface_wrapper(op: str, dtype: str, capacity: int) -> List[str]:
    cpp_type = {"float16": "half", "int32": "int32_t"}[dtype]
    arity = _NATIVE_DTYPE_ARITY[op]
    name = f"HardFuzzTyped_{op}_{dtype}"
    args = ["const AscendC::LocalTensor<float> &output"]
    args += [f"const AscendC::LocalTensor<float> &input{i}" for i in range(arity)]
    args += ["AscendC::LocalTensor<uint8_t> &tmp", "int &offset_ptr", "const int32_t data_length"]
    lines = ["__aicore__ inline void " + name + "(" + ", ".join(args) + ") {"]
    for var in [f"typed_input{i}" for i in range(arity)] + ["typed_output"]:
        lines += ["    offset_ptr = (offset_ptr + 31) & ~31;",
                  f"    AscendC::LocalTensor<{cpp_type}> {var} = tmp.ReinterpretCast<{cpp_type}>()[offset_ptr / sizeof({cpp_type})];",
                  f"    {var}.SetSize({capacity} * sizeof({cpp_type}));",
                  f"    offset_ptr += {capacity} * sizeof({cpp_type});"]
    round_mode = "CAST_RINT" if dtype == "int32" else "CAST_NONE"
    for i in range(arity):
        lines.append(f"    AscendC::Cast<{cpp_type}, float>(typed_input{i}, input{i}, AscendC::RoundMode::{round_mode}, data_length);")
    lines.append("    AscendC::PipeBarrier<PIPE_ALL>();")
    native_args = ", ".join(["typed_output"] + [f"typed_input{i}" for i in range(arity)] + ["data_length"])
    lines.append(f"    AscendC::{op}<{cpp_type}>({native_args});")
    lines.append("    AscendC::PipeBarrier<PIPE_ALL>();")
    lines.append(f"    AscendC::Cast<float, {cpp_type}>(output, typed_output, AscendC::RoundMode::CAST_NONE, data_length);")
    lines.append("}")
    return lines

# Convert graph dependencies into typed AscendC calls.
class CFG:
    def __init__(self, data: Dict):
        self.num_nodes: int = int(data.get("num_nodes", 0))
        self.shape_seed = int.from_bytes(hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).digest()[:8], "big")
        raw_adj: Dict[str, List[int]] = data.get("adjacency", {})
        self.adj: Dict[int, List[int]] = {int(k): list(map(int, v)) for k, v in raw_adj.items()}
        self.edge_tags: Dict[Tuple[int, int], List[str]] = {}
        for e in data.get("edges", []):
            try:
                u = int(e.get("from"))
                v = int(e.get("to"))
            except Exception:
                continue
            tags = e.get("tags", []) or []
            self.edge_tags[(u, v)] = list(tags)
        self.nodes: Dict[int, Dict] = {}
        for nd in data.get("nodes", []):
            self.nodes[int(nd.get("id"))] = nd
        self.incoming: Dict[int, List[int]] = {i: [] for i in range(self.num_nodes)}
        for u, vs in self.adj.items():
            for v in vs:
                if 0 <= v < self.num_nodes:
                    self.incoming[v].append(u)
        for v in range(self.num_nodes):
            self.incoming[v] = sorted(set(self.incoming[v]))
        self.incoming_non_struct: Dict[int, List[int]] = {i: [] for i in range(self.num_nodes)}
        for (u, v), tags in self.edge_tags.items():
            if 0 <= v < self.num_nodes and 0 <= u < self.num_nodes:
                if ("cycle" in tags) or ("loop_entry" in tags):
                    continue
                self.incoming_non_struct[v].append(u)
        for v in range(self.num_nodes):
            unlabeled_preds = [u for u in self.incoming.get(v, []) if (u, v) not in self.edge_tags]
            merged = set(self.incoming_non_struct[v]) | set(unlabeled_preds)
            self.incoming_non_struct[v] = sorted(merged)
        self.cond_map: Dict[int, Tuple[int, int, int]] = {}
        cond_list = data.get("conditions", [])
        for c in cond_list:
            s = int(c["split"]) if isinstance(c, dict) else int(c[0])
            m = int(c["merge"]) if isinstance(c, dict) else int(c[1])
            nd = self.nodes.get(s, {})
            if_succ = nd.get("cond_if")
            else_succ = nd.get("cond_else")
            if if_succ is None or else_succ is None:
                succs = sorted(self.adj.get(s, []))
                if len(succs) >= 2:
                    if_succ, else_succ = succs[0], succs[1]
                elif len(succs) == 1:
                    if_succ, else_succ = succs[0], succs[0]
                else:
                    if_succ, else_succ = s + 1, s + 1
            self.cond_map[s] = (int(if_succ), int(else_succ), m)
        self.while_map: Dict[int, Tuple[int, int]] = {}
        for cyc in data.get("cycles", []):
            a = int(cyc["start"]) if isinstance(cyc, dict) else int(cyc[0])
            b = int(cyc["end"]) if isinstance(cyc, dict) else int(cyc[1])
            pre = a - 1
            if pre >= 0 and self.op(pre) == "while_loop":
                self.while_map[pre] = (a, b)
        self.target_nodes: set = set()
        for while_head, (a, b) in self.while_map.items():
            if 0 <= a < self.num_nodes:
                self.target_nodes.add(a)
        for s, (if_succ, else_succ, _) in self.cond_map.items():
            if 0 <= if_succ < self.num_nodes:
                self.target_nodes.add(if_succ)
            if 0 <= else_succ < self.num_nodes:
                self.target_nodes.add(else_succ)

    def op(self, n: int) -> str:
        return sanitize_op(self.nodes.get(n, {}).get("op"))

    def generate_code(self, enable_dump: bool = False) -> List[str]:
        lines: List[str] = []
        indent_unit = "    "
        trace_node_exec = _env_on("TRACE_NODE_EXEC")

        capacity = int(os.environ.get("BENCHMARK_TENSOR_LEN", "64"))
        template = Path(__file__).resolve().parent / "baremix" / "baremix_custom.cpp"
        match = re.search(r"\bint\s+tensor_len\s*=\s*(\d+)\s*;", template.read_text(encoding="utf-8"))
        if match is None or capacity != int(match.group(1)):
            raise ValueError("BENCHMARK_TENSOR_LEN must match the buffer capacity in baremix/baremix_custom.cpp")
        lower = int(os.environ.get("TENSOR_LEN_MIN", "16"))
        upper = int(os.environ.get("TENSOR_LEN_MAX", str(capacity)))
        if capacity <= 0 or not (0 < lower <= upper <= capacity):
            raise ValueError("Require 0 < TENSOR_LEN_MIN <= TENSOR_LEN_MAX <= BENCHMARK_TENSOR_LEN")
        if lower & (lower - 1) or upper & (upper - 1):
            raise ValueError("TENSOR_LEN_MIN and TENSOR_LEN_MAX must be powers of two")
        choices = [1 << exponent for exponent in range(lower.bit_length() - 1, upper.bit_length())]
        shape_seed = int(os.environ.get("TENSOR_LEN_SEED", str(self.shape_seed)))
        shape_rng = random.Random(shape_seed)
        self.tensor_lengths = {}
        for node_id in sorted(self.nodes):
            if self.op(node_id) in ("while_loop", "if_else_condition", "nop", "Input_node"):
                continue
            explicit = self.nodes[node_id].get("tensor_len")
            length = int(explicit) if explicit is not None else shape_rng.choice(choices)
            if self.op(node_id) == "MatMul":
                length = 16
            if length <= 0 or length & (length - 1) or length > capacity:
                raise ValueError(f"Invalid power-of-two tensor_len at node {node_id}: {length}")
            self.tensor_lengths[node_id] = length
        self.shape_manifest = {"seed": shape_seed, "buffer_capacity": capacity,
                               "choices": choices, "node_tensor_lengths": self.tensor_lengths}
        dtype_seed = int(os.environ.get("DTYPE_SEED", str(self.shape_seed ^ 0xD7A5E)))
        dtype_rng = random.Random(dtype_seed)
        self.node_dtypes = {}
        for node_id in sorted(self.tensor_lengths):
            options = interface_dtypes(self.op(node_id))
            explicit = self.nodes[node_id].get("dtype")
            selected = str(explicit) if explicit is not None else dtype_rng.choice(options)
            if selected not in options:
                raise ValueError(f"Unsupported dtype {selected} for node {node_id} ({self.op(node_id)}); allowed: {options}")
            self.node_dtypes[node_id] = selected
        self.dtype_manifest = {"seed": dtype_seed, "storage_dtype": "float32",
                               "node_interface_dtypes": self.node_dtypes,
                               "supported_native_dtypes": {op: interface_dtypes(op) for op in _NATIVE_DTYPE_ARITY}}
        out_var_map: Dict[int, str] = {}
        next_output_id = 1
        first_instruction_done = False

        function_ops_config: Dict[str, Dict] = {
            "Not": {"fn_name": "Not", "num_inputs": 1},
            "Xor": {"fn_name": "Xor", "num_inputs": 2},
            "And": {"fn_name": "And", "num_inputs": 2},
            "Or": {"fn_name": "Or", "num_inputs": 2},
        }

        def ensure_output_var(n: int) -> str:
            nonlocal next_output_id
            if n not in out_var_map:
                out_var_map[n] = f"output{next_output_id}"
                next_output_id += 1
            return out_var_map[n]

        def get_input_vars(n: int) -> List[str]:
            preds = self.incoming_non_struct.get(n, [])
            inputs: List[str] = []
            for p in preds:
                if self.op(p) in ("while_loop", "if_else_condition"):
                    continue
                if p in out_var_map:
                    inputs.append(out_var_map[p])
                else:
                    inputs.append("input1")
            return inputs

        def emit_op_line(op_name: str, out_var: str, in_vars: List[str], curr: int, indent: int) -> str:
            ind = indent_unit * indent
            dtype = self.node_dtypes.get(curr, "float32")
            if dtype != "float32":
                arity = _NATIVE_DTYPE_ARITY[op_name]
                inputs = (in_vars + ["input1"] * arity)[:arity]
                args = ", ".join([out_var] + inputs + ["tmp", "offset_ptr", "tensor_len"])
                return f"{ind}HardFuzzTyped_{op_name}_{dtype}({args}); // node {curr}"

            if op_name == "Ln":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Ln({out_var}, {src}, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Sqrt":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Sqrt({out_var}, {src}, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Rsqrt":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Rsqrt({out_var}, {src}, offset_ptr, tensor_len); // node {curr}"

            if op_name == "Acos":
                in0 = in_vars[0] if in_vars else "input1"
                return f"{ind}Acos({out_var}, {in0}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Asin":
                in0 = in_vars[0] if in_vars else "input1"
                return f"{ind}Asin({out_var}, {in0}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Atanh":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Atanh({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"

            if op_name == "Acosh":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Acosh({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name in function_ops_config:
                cfg = function_ops_config[op_name]
                num_inputs = int(cfg.get("num_inputs", len(in_vars)))
                chosen_inputs = in_vars[:num_inputs]
                if op_name == "Not":

                    src = chosen_inputs[0] if chosen_inputs else "input1"
                    args = ", ".join([out_var, src, "tmp", "offset_ptr", "tensor_len"])
                    return f"{ind}{cfg.get('fn_name', op_name)}({args}); // node {curr}"
                if op_name in ("Xor", "And", "Or"):
                    if len(chosen_inputs) < 2:
                        chosen_inputs = (chosen_inputs + ["input1", "input1"])[:2]
                    args = ", ".join([out_var, chosen_inputs[0], chosen_inputs[1], "tmp", "offset_ptr", "tensor_len"])
                    return f"{ind}{cfg.get('fn_name', op_name)}({args}); // node {curr}"
                return f"{ind}{cfg.get('fn_name', op_name)}({args}); // node {curr}"
            if op_name == "Div":
                if len(in_vars) >= 2:
                    return f"{ind}Div({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                if len(in_vars) >= 1:
                    return f"{ind}Div({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"

                return f"{ind}Div({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Reciprocal":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Reciprocal({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Fmod":
                if len(in_vars) >= 2:
                    return f"{ind}Fmod({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                if len(in_vars) >= 1:
                    return f"{ind}Fmod({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"
                return f"{ind}Fmod({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "BilinearInterpolation":
                if len(in_vars) >= 2:
                    return f"{ind}BilinearInterpolation({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                elif len(in_vars) == 1:
                    return f"{ind}BilinearInterpolation({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"
                else:
                    return f"{ind}BilinearInterpolation({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "AddReluCast":
                if len(in_vars) >= 2:
                    return f"{ind}AddReluCast({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                elif len(in_vars) >= 1:
                    return f"{ind}AddReluCast({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"
                else:
                    return f"{ind}AddReluCast({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "AddDeqRelu":
                if len(in_vars) >= 2:
                    return f"{ind}AddDeqRelu({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                elif len(in_vars) >= 1:
                    return f"{ind}AddDeqRelu({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"
                else:
                    return f"{ind}AddDeqRelu({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "SubReluCast":
                if len(in_vars) >= 2:
                    return f"{ind}SubReluCast({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                elif len(in_vars) >= 1:
                    return f"{ind}SubReluCast({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"
                else:
                    return f"{ind}SubReluCast({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "MulCast":
                if len(in_vars) >= 2:
                    return f"{ind}MulCast({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                elif len(in_vars) >= 1:
                    return f"{ind}MulCast({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"
                else:
                    return f"{ind}MulCast({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "WholeReduceMax":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}WholeReduceMax({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "Sort32":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Sort32({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "MrgSort":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}MrgSort({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "GetMrgSortResult":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}GetMrgSortResult({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "GetReduceMaxMinCount":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}GetReduceMaxMinCount({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "WholeReduceMin":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}WholeReduceMin({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "WholeReduceSum":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}WholeReduceSum({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "BlockReduceMax":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}BlockReduceMax({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "BlockReduceMin":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}BlockReduceMin({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "BlockReduceSum":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}BlockReduceSum({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "PairReduceSum":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}PairReduceSum({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "RepeatReduceSum":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}RepeatReduceSum({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "CreateVecIndex":
                return f"{ind}CreateVecIndex({out_var}, tensor_len); // node {curr}"
            if op_name == "VectorPadding":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}VectorPadding({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "Compare":
                if len(in_vars) >= 2:
                    return f"{ind}Compare({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                elif len(in_vars) >= 1:
                    return f"{ind}Compare({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"
                else:
                    return f"{ind}Compare({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "CompareCmpMask":
                if len(in_vars) >= 2:
                    return f"{ind}CompareCmpMask({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                elif len(in_vars) >= 1:
                    return f"{ind}CompareCmpMask({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"
                else:
                    return f"{ind}CompareCmpMask({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Select":
                if len(in_vars) >= 2:
                    return f"{ind}Select({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                elif len(in_vars) >= 1:
                    return f"{ind}Select({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"
                else:
                    return f"{ind}Select({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "CompareScalar":
                if len(in_vars) >= 2:
                    return f"{ind}CompareScalar({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                elif len(in_vars) >= 1:
                    return f"{ind}CompareScalar({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"
                else:
                    return f"{ind}CompareScalar({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "GatherMask":
                if len(in_vars) >= 2:
                    return f"{ind}GatherMask({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                elif len(in_vars) >= 1:
                    return f"{ind}GatherMask({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"
                else:
                    return f"{ind}GatherMask({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "ProposalConcat":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}ProposalConcat({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "GeGLU":
                if len(in_vars) >= 2:
                    return f"{ind}GeGLU({out_var}, {in_vars[0]}, {in_vars[1]}, tensor_len); // node {curr}"
                elif len(in_vars) >= 1:
                    return f"{ind}GeGLU({out_var}, {in_vars[0]}, input1, tensor_len); // node {curr}"
                else:
                    return f"{ind}GeGLU({out_var}, input1, input1, tensor_len); // node {curr}"
            if op_name == "LogSoftMax":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}LogSoftMax({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "SoftmaxFlash":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}SoftmaxFlash({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "SoftmaxFlashV2":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}SoftmaxFlashV2({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "SoftmaxFlashV3":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}SoftmaxFlashV3({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "SoftmaxGrad":
                if len(in_vars) >= 2:
                    return f"{ind}SoftmaxGrad({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                elif len(in_vars) >= 1:
                    return f"{ind}SoftmaxGrad({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"
                else:
                    return f"{ind}SoftmaxGrad({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "SoftmaxGradFront":
                if len(in_vars) >= 2:
                    return f"{ind}SoftmaxGradFront({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                elif len(in_vars) >= 1:
                    return f"{ind}SoftmaxGradFront({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"
                else:
                    return f"{ind}SoftmaxGradFront({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Swish":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Swish({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "SwiGLU":
                in0 = in_vars[0] if len(in_vars) >= 1 else "input1"
                in1 = in_vars[1] if len(in_vars) >= 2 else "input1"

                return f"{ind}::SwiGLU({out_var}, {in0}, {in1}, tensor_len); // node {curr}"
            if op_name == "DataCopyPad":

                return f"{ind}DataCopyPad({out_var}, input1Global, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "SetPadValue":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}SetPadValue({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "InitConstValue":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}InitConstValue({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "LoadData":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}LoadData({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "LoadUnzipIndex":
                src = in_vars[0] if in_vars else "input1"

                return f"{ind}LoadUnzipIndex({out_var}, {src}, tilingGm, tensor_len); // node {curr}"
            if op_name == "LoadDataUnzip":
                src = in_vars[0] if in_vars else "input1"

                return f"{ind}LoadDataUnzip({out_var}, {src}, tilingGm, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "LoadDataWithTranspose":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}LoadDataWithTranspose({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "LoadDataWithSparse":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}LoadDataWithSparse({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "SetFmatrix":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}SetFmatrix({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "SetLoadDataBoundary":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}SetLoadDataBoundary({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "SetLoadDataRepeat":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}SetLoadDataRepeat({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "SetLoadDataPaddingValue":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}SetLoadDataPaddingValue({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "Fixpipe":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Fixpipe({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "SetFixPipeConfig":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}SetFixPipeConfig({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "SetFixpipeNz2ndFlag":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}SetFixpipeNz2ndFlag({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "SetFixpipePreQuantFlag":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}SetFixpipePreQuantFlag({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "SetFixPipeClipRelu":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}SetFixPipeClipRelu({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "SetFixPipeAddr":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}SetFixPipeAddr({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "SetAippFunctions":
                src = in_vars[0] if in_vars else "input1"

                return f"{ind}SetAippFunctions({out_var}, {src}, tilingGm, tensor_len); // node {curr}"
            if op_name == "LoadImageToLocal":
                src = in_vars[0] if in_vars else "input1"

                return f"{ind}LoadImageToLocal({out_var}, {src}, tilingGm, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "BatchNorm":
                in0 = in_vars[0] if len(in_vars) >= 1 else "input1"
                in1 = in_vars[1] if len(in_vars) >= 2 else "input1"
                in2 = in_vars[2] if len(in_vars) >= 3 else "input1"
                return f"{ind}BatchNorm({out_var}, {in0}, {in1}, {in2}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "ReGlu":
                in0 = in_vars[0] if len(in_vars) >= 1 else "input1"
                in1 = in_vars[1] if len(in_vars) >= 2 else "input1"
                return f"{ind}ReGlu({out_var}, {in0}, {in1}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "FasterGelu":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}FasterGelu({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "FasterGeluV2":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}FasterGeluV2({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "GeLU":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}GeLU({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "Sigmoid":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Sigmoid({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "Silu":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Silu({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "DeepNorm":
                in0 = in_vars[0] if len(in_vars) >= 1 else "input1"
                in1 = in_vars[1] if len(in_vars) >= 2 else "input1"
                return f"{ind}DeepNorm({out_var}, {in0}, {in1}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "GroupNorm":
                in0 = in_vars[0] if len(in_vars) >= 1 else "input1"
                in1 = in_vars[1] if len(in_vars) >= 2 else "input1"
                return f"{ind}GroupNorm({out_var}, {in0}, {in1}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "LayerNorm":
                in0 = in_vars[0] if len(in_vars) >= 1 else "input1"
                in1 = in_vars[1] if len(in_vars) >= 2 else "input1"
                return f"{ind}LayerNorm({out_var}, {in0}, {in1}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "LayerNormGrad":
                in0 = in_vars[0] if len(in_vars) >= 1 else "input1"
                in1 = in_vars[1] if len(in_vars) >= 2 else "input1"
                return f"{ind}LayerNormGrad({out_var}, {in0}, {in1}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "LayerNormGradBeta":
                in0 = in_vars[0] if len(in_vars) >= 1 else "input1"
                in1 = in_vars[1] if len(in_vars) >= 2 else "input1"
                return f"{ind}LayerNormGradBeta({out_var}, {in0}, {in1}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Normalize":
                in0 = in_vars[0] if len(in_vars) >= 1 else "input1"
                in1 = in_vars[1] if len(in_vars) >= 2 else "input1"
                return f"{ind}Normalize({out_var}, {in0}, {in1}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "ReduceXorSum":
                in0 = in_vars[0] if len(in_vars) >= 1 else "input1"
                in1 = in_vars[1] if len(in_vars) >= 2 else "input1"
                return f"{ind}ReduceXorSum({out_var}, {in0}, {in1}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Mean":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Mean({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "RmsNorm":
                in0 = in_vars[0] if len(in_vars) >= 1 else "input1"
                in1 = in_vars[1] if len(in_vars) >= 2 else "input1"
                return f"{ind}RmsNorm({out_var}, {in0}, {in1}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "WelfordFinalize":
                in0 = in_vars[0] if len(in_vars) >= 1 else "input1"
                in1 = in_vars[1] if len(in_vars) >= 2 else "input1"
                return f"{ind}WelfordFinalize({out_var}, {in0}, {in1}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "WelfordUpdate":
                in0 = in_vars[0] if len(in_vars) >= 1 else "input1"
                in1 = in_vars[1] if len(in_vars) >= 2 else "input1"
                return f"{ind}WelfordUpdate({out_var}, {in0}, {in1}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "TopK":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}TopK({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Sort":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Sort({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Extract":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Extract({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Concat":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Concat({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Arange":
                return f"{ind}Arange({out_var}, tensor_len); // node {curr}"
            if op_name == "Broadcast":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Broadcast({out_var}, {src}, tensor_len); // node {curr}"
            if op_name == "Fill":
                return f"{ind}Fill({out_var}, tensor_len); // node {curr}"
            if op_name == "Transpose":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Transpose({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "TransData":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}TransData({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Pad":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Pad({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "UnPad":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}UnPad({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "DropOut":
                if len(in_vars) >= 2:
                    return f"{ind}DropOut({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                elif len(in_vars) >= 1:
                    return f"{ind}DropOut({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"
                else:
                    return f"{ind}DropOut({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Gatherb":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Gatherb({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Scatter":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Scatter({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "GetCmpMask":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}GetCmpMask({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "SimpleSoftMax":
                src = in_vars[0] if len(in_vars) >= 1 else "input1"
                in_sum = in_vars[1] if len(in_vars) >= 2 else "input1"
                in_max = in_vars[2] if len(in_vars) >= 3 else "input1"
                return f"{ind}SimpleSoftMax({out_var}, {src}, {in_sum}, {in_max}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "SoftMax":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}SoftMax({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "AdjustSoftMaxRes":
                if len(in_vars) >= 2:
                    return f"{ind}AdjustSoftMaxRes({out_var}, {in_vars[0]}, {in_vars[1]}, tensor_len); // node {curr}"
                elif len(in_vars) >= 1:
                    return f"{ind}AdjustSoftMaxRes({out_var}, {in_vars[0]}, input1, tensor_len); // node {curr}"
                else:
                    return f"{ind}AdjustSoftMaxRes({out_var}, input1, input1, tensor_len); // node {curr}"
            if op_name == "ScalarCountLeadingZero":
                return f"{ind}ScalarCountLeadingZero({out_var}, tensor_len); // node {curr}"
            if op_name == "ScalarCast":
                return f"{ind}ScalarCast({out_var}, {in_vars[0] if in_vars else 'input1'}, tensor_len); // node {curr}"
            if op_name == "Cast":
                return f"{ind}Cast({out_var}, {in_vars[0] if in_vars else 'input1'}, tensor_len); // node {curr}"
            if op_name == "ShiftLeft":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}ShiftLeft({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "ShiftRight":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}ShiftRight({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Mins":
                in0 = in_vars[0] if len(in_vars) >= 1 else "input1"
                in1 = in_vars[1] if len(in_vars) >= 2 else "input1"
                return f"{ind}Mins({out_var}, {in0}, {in1}, tensor_len); // node {curr}"
            if op_name == "Digamma":
                in0 = in_vars[0] if len(in_vars) >= 1 else "input1"
                in1 = in_vars[1] if len(in_vars) >= 2 else "input1"
                return f"{ind}Digamma({out_var}, {in0}, {in1}, tensor_len); // node {curr}"
            if op_name == "ReduceMax":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}ReduceMax({out_var}, {src}, tmp, offset_ptr, (int32_t)sharedTmpBufferSize, tensor_len); // node {curr}"
            if op_name == "ReduceMin":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}ReduceMin({out_var}, {src}, tmp, offset_ptr, (int32_t)sharedTmpBufferSize, tensor_len); // node {curr}"
            if op_name == "ReduceSum":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}ReduceSum({out_var}, {src}, tmp, offset_ptr, (int32_t)sharedTmpBufferSize, tensor_len); // node {curr}"
            if op_name == "ReduceSum2":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}ReduceSum2({out_var}, {src}, tmp, offset_ptr, (int32_t)sharedTmpBufferSize, tensor_len); // node {curr}"
            if op_name == "ReduceMax2":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}ReduceMax2({out_var}, {src}, tmp, offset_ptr, (int32_t)sharedTmpBufferSize, tensor_len); // node {curr}"
            if op_name == "ReduceMin2":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}ReduceMin2({out_var}, {src}, tmp, offset_ptr, (int32_t)sharedTmpBufferSize, tensor_len); // node {curr}"
            if op_name == "ReduceMean":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}ReduceMean({out_var}, {src}, tmp, offset_ptr, (int32_t)sharedTmpBufferSize, tensor_len); // node {curr}"
            if op_name == "ReduceAny":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}ReduceAny({out_var}, {src}, tmp, offset_ptr, (int32_t)sharedTmpBufferSize, tensor_len); // node {curr}"
            if op_name == "ReduceAll":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}ReduceAll({out_var}, {src}, tmp, offset_ptr, (int32_t)sharedTmpBufferSize, tensor_len); // node {curr}"
            if op_name == "ReduceProd":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}ReduceProd({out_var}, {src}, tmp, offset_ptr, (int32_t)sharedTmpBufferSize, tensor_len); // node {curr}"
            if op_name == "AscendAntiQuant":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}AscendAntiQuant({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "AscendDequant":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}AscendDequant({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "AscendQuant":
                if len(in_vars) >= 2:
                    return f"{ind}AscendQuant({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                elif len(in_vars) >= 1:
                    return f"{ind}AscendQuant({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"
                else:
                    return f"{ind}AscendQuant({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "GetAccVal":
                return f"{ind}GetAccVal({out_var}, tensor_len); // node {curr}"
            if op_name == "Adds":
                return f"{ind}Adds({out_var}, {in_vars[0] if in_vars else 'input1'}, tensor_len); // node {curr}"
            if op_name == "Maxs":
                return f"{ind}Maxs({out_var}, {in_vars[0] if in_vars else 'input1'}, tensor_len); // node {curr}"
            if op_name == "LeakyRelu":
                return f"{ind}LeakyRelu({out_var}, {in_vars[0] if in_vars else 'input1'}, tensor_len); // node {curr}"
            if op_name == "Axpy":
                in0 = in_vars[0] if len(in_vars) >= 1 else "input1"
                in1 = in_vars[1] if len(in_vars) >= 2 else "input1"
                return f"{ind}Axpy({out_var}, {in0}, {in1}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "ConstZero" or op_name == "Const_Zero":
                return f"{ind}ConstZero({out_var}, tensor_len); // node {curr}"
            if op_name == "ConstOne":
                return f"{ind}ConstOne({out_var}, tensor_len); // node {curr}"
            if op_name == "Duplicate":
                return f"{ind}Duplicate({out_var}, {in_vars[0] if in_vars else 'input1'}, tensor_len); // node {curr}"
            if op_name == "ScalarGetCountOfValue":
                return f"{ind}ScalarGetCountOfValue({out_var}, {in_vars[0] if in_vars else 'input1'}, tensor_len); // node {curr}"
            if op_name == "CountBitsCntSameAsSignBit":
                return f"{ind}CountBitsCntSameAsSignBit({out_var}, {in_vars[0] if in_vars else 'input1'}, tensor_len); // node {curr}"
            if op_name == "ScalarGetSFFValue":
                return f"{ind}ScalarGetSFFValue({out_var}, {in_vars[0] if in_vars else 'input1'}, tensor_len); // node {curr}"

            if op_name == "Lgamma":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Lgamma({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "DataCopyPad_Mask1":
                return f"{ind}DataCopyPad_Mask1({out_var}, input1Global, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "DataCopyPad_Mask2":
                return f"{ind}DataCopyPad_Mask2({out_var}, input1Global, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "LoadUnzipIndex_Mask1":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}LoadUnzipIndex_Mask1({out_var}, {src}, tilingGm, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "LoadUnzipIndex_Mask2":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}LoadUnzipIndex_Mask2({out_var}, {src}, tilingGm, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "LoadDataUnzip_Mask1":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}LoadDataUnzip_Mask1({out_var}, {src}, tilingGm, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "LoadDataUnzip_Mask2":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}LoadDataUnzip_Mask2({out_var}, {src}, tilingGm, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "SetAippFunctions_Mask1":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}SetAippFunctions_Mask1({out_var}, {src}, tilingGm, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "SetAippFunctions_Mask2":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}SetAippFunctions_Mask2({out_var}, {src}, tilingGm, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "LoadImageToLocal_Mask1":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}LoadImageToLocal_Mask1({out_var}, {src}, tilingGm, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "LoadImageToLocal_Mask2":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}LoadImageToLocal_Mask2({out_var}, {src}, tilingGm, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Exp_Mask1":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Exp_Mask1({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"
            if op_name == "Exp_Mask2":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Exp_Mask2({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"

            _MASK_REDUCE_VEC_TMP_OPS = (
                "ReduceMax_Mask1",
                "ReduceMax_Mask2",
                "ReduceMin_Mask1",
                "ReduceMin_Mask2",
                "ReduceSum_Mask1",
                "ReduceSum_Mask2",
            )
            if op_name in _MASK_REDUCE_VEC_TMP_OPS:
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}{op_name}({out_var}, {src}, tmp, offset_ptr, (int32_t)sharedTmpBufferSize, tensor_len); // node {curr}"

            _MASK_UNARY_TMP_OPS = (
                "Ln_Mask1",
                "Ln_Mask2",
                "Abs_Mask1",
                "Abs_Mask2",
                "Reciprocal_Mask1",
                "Reciprocal_Mask2",
                "Sqrt_Mask1",
                "Sqrt_Mask2",
                "Rsqrt_Mask1",
                "Rsqrt_Mask2",
                "Not_Mask1",
                "Not_Mask2",
                "Relu_Mask1",
                "Relu_Mask2",
                "Adds_Mask1",
                "Adds_Mask2",
                "Muls_Mask1",
                "Muls_Mask2",
                "Maxs_Mask1",
                "Maxs_Mask2",
                "ShiftLeft_Mask1",
                "ShiftLeft_Mask2",
                "ShiftRight_Mask1",
                "ShiftRight_Mask2",
                "LeakyRelu_Mask1",
                "LeakyRelu_Mask2",
                "ScalarGetCountOfValue_Mask1",
                "ScalarGetCountOfValue_Mask2",
                "ScalarCountLeadingZero_Mask1",
                "ScalarCountLeadingZero_Mask2",
                "ScalarCast_Mask1",
                "ScalarCast_Mask2",
                "CountBitsCntSameAsSignBit_Mask1",
                "CountBitsCntSameAsSignBit_Mask2",
                "ScalarGetSFFValue_Mask1",
                "ScalarGetSFFValue_Mask2",
                "ToBfloat16_Mask1",
                "ToBfloat16_Mask2",
                "ToFloat_Mask1",
                "ToFloat_Mask2",
                "Cast_Mask1",
                "Cast_Mask2",
                "CastDeq_Mask1",
                "CastDeq_Mask2",
                "WholeReduceMax_Mask1",
                "WholeReduceMax_Mask2",
                "WholeReduceMin_Mask1",
                "WholeReduceMin_Mask2",
                "WholeReduceSum_Mask1",
                "WholeReduceSum_Mask2",
                "BlockReduceMax_Mask1",
                "BlockReduceMax_Mask2",
                "BlockReduceMin_Mask1",
                "BlockReduceMin_Mask2",
                "BlockReduceSum_Mask1",
                "BlockReduceSum_Mask2",
                "PairReduceSum_Mask1",
                "PairReduceSum_Mask2",
                "RepeatReduceSum_Mask1",
                "RepeatReduceSum_Mask2",
                "DataCopyPad_Mask1",
                "DataCopyPad_Mask2",
                "SetPadValue_Mask1",
                "SetPadValue_Mask2",
                "InitConstValue_Mask1",
                "InitConstValue_Mask2",
                "LoadData_Mask1",
                "LoadData_Mask2",
                "LoadUnzipIndex_Mask1",
                "LoadUnzipIndex_Mask2",
                "LoadDataUnzip_Mask1",
                "LoadDataUnzip_Mask2",
                "LoadDataWithTranspose_Mask1",
                "LoadDataWithTranspose_Mask2",
                "LoadDataWithSparse_Mask1",
                "LoadDataWithSparse_Mask2",
                "SetFmatrix_Mask1",
                "SetFmatrix_Mask2",
                "SetLoadDataBoundary_Mask1",
                "SetLoadDataBoundary_Mask2",
                "SetLoadDataRepeat_Mask1",
                "SetLoadDataRepeat_Mask2",
                "SetLoadDataPaddingValue_Mask1",
                "SetLoadDataPaddingValue_Mask2",
                "Fixpipe_Mask1",
                "Fixpipe_Mask2",
                "SetFixPipeConfig_Mask1",
                "SetFixPipeConfig_Mask2",
                "SetFixpipeNz2ndFlag_Mask1",
                "SetFixpipeNz2ndFlag_Mask2",
                "SetFixpipePreQuantFlag_Mask1",
                "SetFixpipePreQuantFlag_Mask2",
                "SetFixPipeClipRelu_Mask1",
                "SetFixPipeClipRelu_Mask2",
                "SetFixPipeAddr_Mask1",
                "SetFixPipeAddr_Mask2",
                "SetAippFunctions_Mask1",
                "SetAippFunctions_Mask2",
                "LoadImageToLocal_Mask1",
                "LoadImageToLocal_Mask2",
                "Duplicate_Mask1",
                "Duplicate_Mask2",
                "CreateVecIndex_Mask1",
                "CreateVecIndex_Mask2",
                "VectorPadding_Mask1",
                "VectorPadding_Mask2",
                "SetMaskCount_Mask1",
                "SetMaskCount_Mask2",
                "SetMaskNorm_Mask1",
                "SetMaskNorm_Mask2",
                "SetVectorMask_Mask1",
                "SetVectorMask_Mask2",
                "ResetMask_Mask1",
                "ResetMask_Mask2",
                "DataCopy_Mask1",
                "DataCopy_Mask2",
                "GetCmpMask_Mask1",
                "GetCmpMask_Mask2",
                "GetAccVal_Mask1",
                "GetAccVal_Mask2",
                "GetReduceMaxMinCount_Mask1",
                "GetReduceMaxMinCount_Mask2",
                "ProposalConcat_Mask1",
                "ProposalConcat_Mask2",
                "Sort32_Mask1",
                "Sort32_Mask2",
                "MrgSort_Mask1",
                "MrgSort_Mask2",
                "GetMrgSortResult_Mask1",
                "GetMrgSortResult_Mask2",
                "Gatherb_Mask1",
                "Gatherb_Mask2",
                "Scatter_Mask1",
                "Scatter_Mask2",
            )
            if op_name in _MASK_UNARY_TMP_OPS:
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}{op_name}({out_var}, {src}, tmp, offset_ptr, tensor_len); // node {curr}"

            _MASK_BINARY_TMP_OPS = (
                "Add_Mask1",
                "Add_Mask2",
                "Sub_Mask1",
                "Sub_Mask2",
                "Mul_Mask1",
                "Mul_Mask2",
                "Div_Mask1",
                "Div_Mask2",
                "Max_Mask1",
                "Max_Mask2",
                "Min_Mask1",
                "Min_Mask2",
                "BilinearInterpolation_Mask1",
                "BilinearInterpolation_Mask2",
                "And_Mask1",
                "And_Mask2",
                "Or_Mask1",
                "Or_Mask2",
                "AddRelu_Mask1",
                "AddRelu_Mask2",
                "AddReluCast_Mask1",
                "AddReluCast_Mask2",
                "AddDeqRelu_Mask1",
                "AddDeqRelu_Mask2",
                "SubRelu_Mask1",
                "SubRelu_Mask2",
                "SubReluCast_Mask1",
                "SubReluCast_Mask2",
                "MulAddDst_Mask1",
                "MulAddDst_Mask2",
                "MulCast_Mask1",
                "MulCast_Mask2",
                "FusedMulAdd_Mask1",
                "FusedMulAdd_Mask2",
                "FusedMulAddRelu_Mask1",
                "FusedMulAddRelu_Mask2",
                "Mins_Mask1",
                "Mins_Mask2",
                "Axpy_Mask1",
                "Axpy_Mask2",
                "Compare_Mask1",
                "Compare_Mask2",
                "CompareCmpMask_Mask1",
                "CompareCmpMask_Mask2",
                "CompareScalar_Mask1",
                "CompareScalar_Mask2",
                "Select_Mask1",
                "Select_Mask2",
                "GatherMask_Mask1",
                "GatherMask_Mask2",
            )
            if op_name in _MASK_BINARY_TMP_OPS:
                if len(in_vars) >= 2:
                    return f"{ind}{op_name}({out_var}, {in_vars[0]}, {in_vars[1]}, tmp, offset_ptr, tensor_len); // node {curr}"
                if len(in_vars) >= 1:
                    return f"{ind}{op_name}({out_var}, {in_vars[0]}, input1, tmp, offset_ptr, tensor_len); // node {curr}"
                return f"{ind}{op_name}({out_var}, input1, input1, tmp, offset_ptr, tensor_len); // node {curr}"
            unary_ops = {"Exp", "Relu", "Abs", "Atan", "Ceil", "Cos", "Cosh", "Erf", "Erfc", "Floor", "Frac", "Log", "Round", "Sign", "Sin", "Sinh", "Tan", "Tanh", "Trunc", "Asinh"}
            if op_name in unary_ops:
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}{op_name}({out_var}, {src}, tensor_len); // node {curr}"

            binary_ops = {"Add", "Sub", "Mul", "Max", "Min", "AddRelu", "SubRelu", "Power", "ClampMax", "FusedMulAdd", "MulAddDst"}
            if op_name in binary_ops:
                if len(in_vars) >= 2:
                    return f"{ind}{op_name}({out_var}, {in_vars[0]}, {in_vars[1]}, tensor_len); // node {curr}"
                elif len(in_vars) >= 1:
                    return f"{ind}{op_name}({out_var}, {in_vars[0]}, input1, tensor_len); // node {curr}"
                else:
                    return f"{ind}{op_name}({out_var}, input1, input1, tensor_len); // node {curr}"

            if op_name == "ClampMin":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}ClampMin({out_var}, {src}, 0.0f, tensor_len); // node {curr}"

            if op_name == "Muls":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Muls({out_var}, {src}, tensor_len); // node {curr}"

            if op_name == "Brcb":
                src = in_vars[0] if in_vars else "input1"
                return f"{ind}Brcb({out_var}, {src}, tensor_len); // node {curr}"

            if in_vars:

                limited_inputs = in_vars[:2]
                args_joined = ", ".join([out_var] + limited_inputs + ["tensor_len"])
            else:
                args_joined = f"{out_var}, input1, tensor_len"
            return f"{ind}{op_name}({args_joined}); // node {curr}"

        def collect_required_wrapper_stems() -> set:

            required: set = set()

            skip_ops = {"while_loop", "if_else_condition", "nop", "Input_node"}
            for n in range(self.num_nodes):
                opn = self.op(n)
                if opn in skip_ops:
                    continue
                required.add(opn)
            if any(("_Mask" in name) or name.endswith("Mask") for name in required):
                required.add("GenerateRandomMask")
            return required

        def emit_wrapper_defs() -> List[str]:
            defs: List[str] = []

            required_wrappers = None if _env_on("INJECT_ALL_WRAPPERS") else collect_required_wrapper_stems()
            for wrapper_text in load_all_wrapper_codes(required_wrappers):
                body = wrapper_text.rstrip("\n")
                if body:
                    for line in body.split("\n"):
                        defs.append(line)
                defs.append("")
            return defs

        def last_output_before(n: int) -> Optional[str]:
            keys = sorted([k for k in out_var_map.keys() if k < n])
            return out_var_map[keys[-1]] if keys else None

        def emit_sequence(curr: int, end_exclusive: int, indent: int) -> int:
            nonlocal first_instruction_done
            while curr < end_exclusive:
                if self.op(curr) == "while_loop" and curr in self.while_map:
                    body_start, body_end = self.while_map[curr]
                    lo = last_output_before(curr) or "input1"
                    lines.append(f"{indent_unit*indent}int loop_iter_{curr} = 0;")
                    lines.append(f"{indent_unit*indent}int loop_limit_{curr} = {random.randint(2, 6)};")

                    use_for_loop = random.choice([True, False])
                    if use_for_loop:

                        if random.random() < 0.5:
                            rand_idx = random.randint(0, 7)
                            limit_expr = f"input2Local.GetValue({rand_idx})"
                        else:
                            limit_expr = f"loop_limit_{curr}"
                        lines.append(
                            f"{indent_unit*indent}for (loop_iter_{curr} = 0; "
                            f"loop_iter_{curr} < {limit_expr}; "
                            f"loop_iter_{curr}++) {{"
                        )
                        emit_sequence(body_start, body_end + 1, indent + 1)

                        lines.append(f"{indent_unit*indent}}}")
                    else:

                        thr_literal = f"1e-{random.randint(3, 5)}f"
                        lines.append(f"{indent_unit*indent}float loop_threshold_{curr} = {thr_literal};")
                        lines.append(f"{indent_unit*indent}while (({lo}.GetValue(0) < loop_threshold_{curr}) && (loop_iter_{curr} < loop_limit_{curr})) {{")
                        emit_sequence(body_start, body_end + 1, indent + 1)
                        lines.append(f"{indent_unit*(indent+1)}loop_iter_{curr}++;")

                        lines.append(f"{indent_unit*indent}}}")
                    curr = body_end + 1
                    continue
                if curr in self.cond_map and self.op(curr) == "if_else_condition":
                    if_succ, else_succ, merge_n = self.cond_map[curr]
                    lo = last_output_before(curr) or "input1"
                    if_empty = (if_succ == merge_n)
                    else_empty = (else_succ == merge_n)

                    passthrough_vars: List[str] = []
                    if if_empty != else_empty:
                        non_empty_start = else_succ if if_empty else if_succ
                        n = non_empty_start
                        while n < merge_n:
                            opn = self.op(n)
                            if opn == "while_loop" and n in self.while_map:
                                body_start, body_end = self.while_map[n]
                                for bn in range(body_start, body_end + 1):
                                    bop = self.op(bn)
                                    if bop not in ("while_loop", "if_else_condition", "nop", "Input_node"):
                                        passthrough_vars.append(ensure_output_var(bn))
                                n = body_end + 1
                                continue
                            if opn == "if_else_condition" and n in self.cond_map:
                                _, _, nested_merge = self.cond_map[n]
                                for nn in range(n + 1, nested_merge):
                                    nop = self.op(nn)
                                    if nop not in ("while_loop", "if_else_condition", "nop", "Input_node"):
                                        passthrough_vars.append(ensure_output_var(nn))
                                n = nested_merge
                                continue
                            if opn not in ("nop", "Input_node"):
                                passthrough_vars.append(ensure_output_var(n))
                            n += 1

                    thr_literal = f"1e-{random.randint(3, 5)}f"
                    lines.append(f"{indent_unit*indent}float cond_threshold_{curr} = {thr_literal};")
                    lines.append(f"{indent_unit*indent}if ({lo}.GetValue(0) < cond_threshold_{curr}) {{")

                    if if_empty and passthrough_vars:
                        for var in passthrough_vars:
                            lines.append(f"{indent_unit*(indent+1)}AscendC::DataCopy({var}, {lo}, tensor_len);")
                    else:
                        emit_sequence(if_succ, merge_n, indent + 1)

                    lines.append(f"{indent_unit*indent}}} else {{")

                    if else_empty and passthrough_vars:
                        for var in passthrough_vars:
                            lines.append(f"{indent_unit*(indent+1)}AscendC::DataCopy({var}, {lo}, tensor_len);")
                    else:
                        emit_sequence(else_succ, merge_n, indent + 1)

                    lines.append(f"{indent_unit*indent}}}")
                    curr = merge_n
                    continue
                out_var = ensure_output_var(curr)
                if not first_instruction_done:
                    in_vars = ["input1"]
                    first_instruction_done = True
                else:
                    in_vars = get_input_vars(curr)
                opn = self.op(curr)
                op_line = emit_op_line(opn, out_var, in_vars, curr, indent)
                call_length = self.tensor_lengths.get(curr, capacity)
                op_line = op_line.replace("tensor_len", str(call_length))

                lines.append(f"{indent_unit * indent}AscendC::Duplicate<float>({out_var}, 0.0f, tensor_len); // pad node {curr}")
                lines.append(f"{indent_unit * indent}AscendC::PipeBarrier<PIPE_ALL>();")
                if trace_node_exec:
                    lines.append(f'{indent_unit * indent}AscendC::printf("[TRACE] node {curr} {opn} begin\\n");')

                if "offset_ptr" in op_line:
                    lines.append(f"{indent_unit * indent}offset_ptr = scratch_base;")
                lines.append(op_line)

                lines.append(f"{indent_unit * indent}AscendC::PipeBarrier<PIPE_ALL>();")
                if trace_node_exec:
                    lines.append(f'{indent_unit * indent}AscendC::printf("[TRACE] node {curr} {opn} end\\n");')
                if enable_dump:
                    lines.append(f"{indent_unit * indent}AscendC::DumpTensor({out_var}, 5, tensor_len);")
                    lines.append(f"{indent_unit * indent}AscendC::printf(\"{out_var}:{in_vars}\\n\");")
                curr += 1
            return curr

        wrapper_defs = emit_wrapper_defs()
        typed_pairs = sorted({(self.op(n), dtype) for n, dtype in self.node_dtypes.items() if dtype != "float32"})
        for op, dtype in typed_pairs:
            wrapper_defs.extend(typed_interface_wrapper(op, dtype, capacity))
            wrapper_defs.append("")
        emit_sequence(0, self.num_nodes, 1)
        max_out_id = next_output_id - 1
        sorted_outputs = [f"output{i}" for i in range(1, max_out_id + 1)]
        prog_lines: List[str] = []
        prog_lines.append("__aicore__ inline void Compute(uint32_t rowNum) {")

        prog_lines.append(f"{indent_unit}AscendC::LocalTensor<uint8_t> tmp;")
        prog_lines.append(f"{indent_unit}bool ans = AscendC::PopStackBuffer<uint8_t, AscendC::TPosition::LCM>(tmp);")
        prog_lines.append(f"{indent_unit}int offset_ptr = 0;")
        prog_lines.append(f"{indent_unit}int sharedTmpBufferSize = 60 * 1024;")
        prog_lines.append(f"{indent_unit}constexpr int kOutNum = {max(1, max_out_id)};")

        tensor_len_env = os.environ.get("BENCHMARK_TENSOR_LEN", "64")
        try:
            tensor_len_cfg = int(tensor_len_env)
        except ValueError:
            tensor_len_cfg = 64
        if tensor_len_cfg <= 0:
            tensor_len_cfg = 64
        full_tensor_len_cfg = tensor_len_cfg
        active_len_cfg = tensor_len_cfg
        active_len_env = os.environ.get("BENCHMARK_ACTIVE_TENSOR_LEN", "")
        if active_len_env:
            try:
                active_len_cfg = int(active_len_env)
            except ValueError:
                active_len_cfg = full_tensor_len_cfg
            if active_len_cfg <= 0:
                active_len_cfg = full_tensor_len_cfg
            if active_len_cfg > full_tensor_len_cfg:
                active_len_cfg = full_tensor_len_cfg
        prog_lines.append(f"{indent_unit}int tensor_len = {full_tensor_len_cfg};")
        prog_lines.append(f"{indent_unit}int active_len = {active_len_cfg};")

        if sorted_outputs:
            prog_lines.append(f"{indent_unit}AscendC::LocalTensor<float> {', '.join(sorted_outputs)};")
            prog_lines.append(f"{indent_unit}{sorted_outputs[0]} = tmp.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];")
            prog_lines.append(f"{indent_unit}{sorted_outputs[0]}.SetSize(tensor_len * sizeof(float));")
            prog_lines.append(f"{indent_unit}offset_ptr += {sorted_outputs[0]}.GetSize();")
            for var in sorted_outputs[1:]:
                prog_lines.append(f"{indent_unit}{var} = tmp.ReinterpretCast<float>()[offset_ptr / (int32_t)sizeof(float)];")
                prog_lines.append(f"{indent_unit}{var}.SetSize(tensor_len * sizeof(float));")
                prog_lines.append(f"{indent_unit}offset_ptr += {var}.GetSize();")
        prog_lines.append(f"{indent_unit}const int scratch_base = offset_ptr;")

        for var in sorted_outputs:
            prog_lines.append(
                f"{indent_unit}AscendC::Duplicate<float>({var}, 0.0f, (uint32_t)tensor_len);"
            )
        prog_lines.extend(lines)
        if sorted_outputs:

            prog_lines.append(f"{indent_unit}if (active_len < tensor_len) {{")
            prog_lines.append(f"{indent_unit}    for (int _i = active_len; _i < tensor_len; ++_i) {{")
            prog_lines.append(f"{indent_unit}        {sorted_outputs[-1]}.SetValue(_i, -1000.0f - (float)_i);")
            prog_lines.append(f"{indent_unit}    }}")
            prog_lines.append(f"{indent_unit}}}")

        try:
            from generate_graph import operation_set as ALL_NODE_OPS
        except Exception:
            ALL_NODE_OPS = []

        def _normalize_emit_op(op_raw: str) -> str:

            return sanitize_op(op_raw)

        structural_ops = {"nop", "Input_node", "while_loop", "if_else_condition", "if-else-condition"}
        total_ops_set = {_normalize_emit_op(op) for op in ALL_NODE_OPS if sanitize_op(op) not in structural_ops}
        total_ops = sorted(total_ops_set)

        code_text = "\n".join(wrapper_defs + prog_lines)
        covered_ops_set = {op for op in total_ops if f"{op}(" in code_text}
        covered_cnt = len(covered_ops_set)
        total_cnt = max(1, len(total_ops))
        ratio = covered_cnt / total_cnt

        prog_lines.append("}")
        out_lines = wrapper_defs + prog_lines
        return [(indent_unit + ln) if ln else ln for ln in out_lines]

def main():
    parser = argparse.ArgumentParser(description="Convert graph JSON into AscendC source")
    parser.add_argument("--json", type=str, default="graphx.json", help="Graph JSON path")
    parser.add_argument("--enable_dump", action="store_true", help="Enable output tensor dumps (disabled by default)")
    parser.add_argument("--shape-manifest", help="Write per-node power-of-two lengths as JSON")
    parser.add_argument("--dtype-manifest", help="Write per-node native interface dtypes as JSON")
    args = parser.parse_args()
    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = CFG(data)
    lines = cfg.generate_code(enable_dump=getattr(args, "enable_dump", False))
    if args.shape_manifest:
        Path(args.shape_manifest).write_text(json.dumps(cfg.shape_manifest, indent=2) + "\n", encoding="utf-8")
    if args.dtype_manifest:
        Path(args.dtype_manifest).write_text(json.dumps(cfg.dtype_manifest, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
