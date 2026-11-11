import os
import re
import sys
import json

_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from tools.log_utils import setup_logger
log = setup_logger("split_cpp")

def normalize_path(p):
    return os.path.abspath(p.replace("\\", "/"))

def split_cpp_file(path):
    p = normalize_path(path)
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"(?m)^[^\n]*\bvoid\s+Compute\s*\([^)]*\)\s*\{", text)
    if not m:
        return text, "", ""
    header_line_start = text.rfind("\n", 0, m.start())
    header_line_start = 0 if header_line_start == -1 else header_line_start + 1
    open_brace_index = text.find("{", m.start())
    depth = 0
    end_index = None
    for i, ch in enumerate(text[open_brace_index:], start=open_brace_index):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_index = i
                break
    if end_index is None:
        return text[:header_line_start], text[open_brace_index + 1 :], ""
    body = text[open_brace_index + 1 : end_index]

    node = re.search(r"(?m)//\s*(?:pad\s+)?node\b", body)
    if not node:
        return text[:header_line_start], body, ""

    pre_node_line_start = body.rfind("\n", 0, node.start())
    pre_node_line_start = 0 if pre_node_line_start == -1 else pre_node_line_start + 1
    part1 = text[:header_line_start]
    part2 = body[:pre_node_line_start]
    part3 = body[pre_node_line_start:]
    return part1, part2, part3

def find_all_calls(text, keywords=("MatMul",)):
    calls = []
    for m in re.finditer(r"(?mi)^[^\n]*\b(" + "|".join(keywords) + r")\s*\(", text):
        le = text.find("\n", m.end())
        ls = text.rfind("\n", 0, m.start())
        ls = 0 if ls == -1 else ls + 1
        line = text[ls:] if le == -1 else text[ls:le]
        args_all = []
        args_first3 = []
        mm = re.search(r"\(([^)]*)\)", line)
        if mm:
            inside = mm.group(1)
            parts = [p.strip() for p in inside.split(",")]
            args_all = parts
            args_first3 = parts[:3]
        calls.append({
            "start_line": ls,
            "end_line": le if le != -1 else len(text),
            "line": line,
            "args_all": args_all,
            "args_first3": args_first3
        })
    return calls

def remove_duplicate_decls(text):

    patterns = [
        (re.compile(r'^\s*AscendC::LocalTensor<uint8_t>\s+tmp\s*;\s*$'), 'LocalTensor tmp'),
        (re.compile(r'^\s*bool\s+ans\s*=\s*AscendC::PopStackBuffer\s*<[^>]*>\s*\(\s*tmp\s*\)\s*;\s*$'), 'PopStackBuffer'),
        (re.compile(r'^\s*int\s+offset_ptr\s*=\s*0\s*;\s*$'), 'offset_ptr = 0'),
        (re.compile(r'^\s*int\s+tensor_len\s*=\s*\d+\s*;\s*$'), 'int tensor_len'),
        (re.compile(r'^\s*int\s+active_len\s*=\s*\d+\s*;\s*$'), 'int active_len'),
    ]
    lines = text.splitlines(True)
    result = []
    seen = set()
    for line in lines:
        skip = False
        for pat, key in patterns:
            if pat.match(line):
                if key in seen:
                    skip = True
                else:
                    seen.add(key)
                break
        if not skip:
            result.append(line)
    return ''.join(result)

def clean_conflicting_decls(text):

    text = text.replace('sharedTmpBuffer', 'tmp')

    text = re.sub(r'^\s*AscendC::LocalTensor<uint8_t>\s+tmp\s*;\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'^\s*bool\s+ans\s*=\s*AscendC::PopStackBuffer\s*<[^>]*>\s*\(\s*tmp\s*\)\s*;\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'^\s*const\s+int\s+scratch_base\s*=\s*[^;]*;\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'^\s*int\s+offset_ptr\s*=\s*[^;]*;\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'^\s*int\s+sharedTmpBufferSize\s*=\s*[^;]*;\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'^\s*constexpr\s+int\s+kOutNum\s*=\s*[^;]*;\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'^\s*AscendC::Duplicate<float>\s*\(\s*output\d+\s*,\s*0\.0f\s*,\s*[^)]*\)\s*;\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'^\s*int\s+tensor_len\s*=\s*[^;]*;\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'^\s*int\s+active_len\s*=\s*[^;]*;\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'^\s*AscendC::LocalTensor<float>\s+input1\s*;\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'^\s*input1\.SetAddrWithOffset\s*\([^)]*\)\s*;\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'^\s*input1\.SetSize\s*\([^)]*\)\s*;\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'^\s*offset_ptr\s*\+=\s*input1\.GetSize\s*\(\s*\)\s*;\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'^\s*AscendC::LocalTensor<int32_t>\s+input2Local\s*;\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'^\s*input2Local\.SetAddrWithOffset\s*\([^)]*\)\s*;\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'^\s*input2Local\.SetSize\s*\([^)]*\)\s*;\s*$', '', text, flags=re.MULTILINE)
    return text

def find_max_output_var(text):
    nums = [int(n) for n in re.findall(r"\boutput(\d+)\b", text)]
    if not nums:
        return ""
    return f"output{max(nums)}"

def inject_missing_wrappers(pre_compute_text, all_code):

    wrappers = {
        'ScalarGetSFFValue': '''    __aicore__ inline void ScalarGetSFFValue(const AscendC::LocalTensor<float> &output,
            const AscendC::LocalTensor<float> &input0,
            const int32_t data_length) {
        for (int32_t i = 0; i < data_length; ++i) {
            uint64_t u64 = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(input0.GetValue(i))));
            int64_t pos = AscendC::ScalarGetSFFValue<1>(u64);
            output.SetValue(i, static_cast<float>(pos));
        }
    }''',
        'Muls': '''__aicore__ inline void Muls(const AscendC::LocalTensor<float> &output,
            const AscendC::LocalTensor<float> &input0,
            const int32_t data_length) {

        const float default_scalar = 2.0f;
        AscendC::Muls(output, input0, default_scalar, data_length);
    }''',
    }
    for name, wrapper in wrappers.items():

        call_3arg = re.compile(
            r'\b' + re.escape(name) + r'\s*\(\s*\w+\s*,\s*\w+\s*,\s*\w+\s*\)')

        def_pat = re.compile(
            r'\bvoid\s+' + re.escape(name) + r'\s*\(')
        has_call = bool(call_3arg.search(all_code))
        has_def = bool(def_pat.search(pre_compute_text))
        if has_call and not has_def:
            log.info(f"INJECTING {name} wrapper (call_found=True, def_found=False)")
            pre_compute_text = pre_compute_text.rstrip('\n') + '\n\n' + wrapper + '\n'
        else:
            log.info(f"SKIP {name}: call_found={has_call}, def_found={has_def}")

    mask_call_re = re.compile(r'\b(\w+_(?:Mask1|Mask2))\s*\(')
    called_masks = set(mask_call_re.findall(all_code))
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    wrappers_dir = os.path.join(_script_dir, 'wrappers')
    for mask_name in sorted(called_masks):
        def_pat = re.compile(r'\bvoid\s+' + re.escape(mask_name) + r'\s*\(')
        if def_pat.search(pre_compute_text):
            continue
        wrapper_path = os.path.join(wrappers_dir, mask_name + '.cpp')
        if os.path.isfile(wrapper_path):
            with open(wrapper_path, 'r', encoding='utf-8') as fw:
                wrapper_code = fw.read().strip()
            pre_compute_text = pre_compute_text.rstrip('\n') + '\n\n' + wrapper_code + '\n'
            log.info(f"INJECTING {mask_name} wrapper from {wrapper_path}")
    return pre_compute_text

# Assemble generated calls into the NPU program template.
def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_script_dir, "program_out.cpp")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    base = sys.argv[3] if len(sys.argv) > 3 else None
    json_path = sys.argv[5] if len(sys.argv) > 5 else None
    a, b, c = split_cpp_file(path)
    b = clean_conflicting_decls(b)

    full_compute = a + b + c
    a = inject_missing_wrappers(a, full_compute)
    calls = find_all_calls(c, keywords=("MatMul",))
    if not calls:
        cr_pre, cr_post = c, ""
        matmul_args = []
        matmul_args_first3 = []
    else:

        json_matmul_id_to_slot = {}
        if json_path:
            try:
                with open(normalize_path(json_path), "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                matmul_ids = []
                for nd in data.get("nodes", []):
                    op = str(nd.get("op", "")).lower()
                    if op == "matmul":
                        if nd.get("id") is not None:
                            matmul_ids.append(int(nd.get("id")))
                matmul_ids_sorted = sorted(set(matmul_ids))
                json_matmul_id_to_slot = {nid: i for i, nid in enumerate(matmul_ids_sorted)}
            except Exception:
                pass

        call_slots = []
        for call_idx, call in enumerate(calls):
            node_id = None
            m = re.search(r"//\s*node\s*(\d+)", call.get("line", ""))
            if m:
                try:
                    node_id = int(m.group(1))
                except Exception:
                    node_id = None
            slot = call_idx
            if node_id is not None and node_id in json_matmul_id_to_slot:
                slot = json_matmul_id_to_slot[node_id]
            call_slots.append(slot)

        parts = []
        last = 0
        for call in calls:
            parts.append(c[last:call["start_line"]])
            last = (call["end_line"] + 1) if call["end_line"] < len(c) else len(c)
        parts.append(c[last:])
        matmul_args = [call["args_all"] for call in calls]
        matmul_args_first3 = [call["args_first3"] for call in calls]
        blocks_pre = []
        blocks_post = []
        indent_pre = "        "
        indent_post = "        "
        for idx, seg in enumerate(parts[:-1]):
            seg_mod = seg
            if idx < len(matmul_args_first3) and len(matmul_args_first3[idx]) >= 3:
                x1 = matmul_args_first3[idx][1]
                x2 = matmul_args_first3[idx][2]
                slot = call_slots[idx] if idx < len(call_slots) else idx
                dest = matmul_args_first3[idx][0]
                call_length = matmul_args[idx][-1] if len(matmul_args[idx]) >= 4 else "tensor_len"
                seg_mod = seg_mod + ("" if seg_mod.endswith("\n") else "\n") +\
                    f"{indent_pre}AscendC::DataCopy(aGlobal, {x1}, 16);\n" +\
                    f"{indent_pre}AscendC::DataCopy(bGlobal, {x2}, 16);\n" +\
                    f"{indent_pre}AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>({1 + 2*slot + 1});\n" +\
                    f"{indent_pre}OutputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c) + AscendC::GetBlockIdx() * tiling.M * tiling.N);\n" +\
                    f"{indent_pre}AscendC::CrossCoreWaitFlag({1 + 2*slot + 2});\n" +\
                    f"{indent_pre}AscendC::DataCopy({dest}, cGlobal, 16);\n"
            blocks_pre.append(seg_mod)
        cr_pre = "".join(blocks_pre)

        cr_post = parts[-1] if parts else ""
    full_text = open(normalize_path(path), "r", encoding="utf-8").read()
    last_out = find_max_output_var(full_text)
    indent_pre = "        "
    indent_post = "        "
    if cr_pre.strip():
        lp = cr_pre.splitlines()
        if lp:
            m0 = re.match(r"^(\s*)", lp[-1])
            if m0:
                indent_pre = m0.group(1) or indent_pre
    if cr_post.strip():
        lp2 = cr_post.splitlines()
        if lp2:
            m1 = re.match(r"^(\s*)", lp2[0])
            if m1:
                indent_post = m1.group(1) or indent_post
    cr_pre_mod = clean_conflicting_decls(cr_pre)
    cr_post_mod = clean_conflicting_decls(cr_post)

    diag_start = (
        indent_pre + 'AscendC::printf("[START] Compute begin: tensor_len=%d active_len=%d '
        'input1[0]=%f input2[0]=%d\\n", '
        'tensor_len, active_len, input1.GetValue(0), input2Local.GetValue(0));\n'
    )
    cr_pre_mod = indent_pre + "const int scratch_base = (offset_ptr + 31) & ~31;\n" + diag_start + cr_pre_mod
    if last_out:

        diag = (
            indent_post + 'AscendC::printf("[RESULT] Compute done. Final output (' + last_out + '): '
            'first=%f last=%f len=%d\\n", '
            + last_out + '.GetValue(0), ' + last_out + '.GetValue(tensor_len>1?tensor_len-1:0), tensor_len);\n'
        )
        cr_post_mod = cr_post_mod + ("" if cr_post_mod.endswith("\n") else "\n") + diag +\
            f"{indent_post}AscendC::DataCopy(OutputGlobal, {last_out}, tensor_len);\n"
    try:
        _script_dir2 = os.path.dirname(os.path.abspath(__file__))
        tmpl_path = sys.argv[4] if len(sys.argv) > 4 else os.path.join(_script_dir2, "baremix", "baremix_custom.cpp")
        tmpl_path = normalize_path(tmpl_path)
        with open(tmpl_path, "r", encoding="utf-8") as tf:
            ttext = tf.read()
        mapping = {
            "// HF_PRE_COMPUTE": a,
            "// HF_TENSOR_ALLOCATIONS": b,
            "// HF_COMPUTE_BODY": cr_pre_mod,
            "// HF_COMPUTE_OUTPUT": cr_post_mod,
        }
        def insert_blocks(text, mapping):
            lines = text.splitlines(True)
            out_lines = []
            for ln in lines:
                replaced = False
                for key, val in mapping.items():
                    if key in ln:
                        out_lines.append(val if val.endswith("\n") else val + "\n")
                        replaced = True
                        break
                if not replaced:
                    out_lines.append(ln)
            return "".join(out_lines)
        new_text = insert_blocks(ttext, mapping)
        new_text = remove_duplicate_decls(new_text)
        out_baremix_dir = os.path.dirname(tmpl_path)
        out_baremix = os.path.join(out_baremix_dir, "baremix_custom_program_out.cpp")
        with open(out_baremix, "w", encoding="utf-8") as of:
            of.write(new_text)
    except OSError as exc:
        raise SystemExit(f"Could not assemble the AscendC program: {exc}") from exc
    json.dumps({
        "pre_compute": a,
        "compute_tensors": b,
        "compute_rest": c,
        "compute_rest_pre": cr_pre_mod,
        "compute_rest_post": cr_post_mod,
        "matmul_args": matmul_args,
        "matmul_args_first3": matmul_args_first3,
        "last_output": last_out
    }, ensure_ascii=False)
    print("generate successfully!")

if __name__ == "__main__":
    main()
