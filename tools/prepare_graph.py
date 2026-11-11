import copy
import json
from pathlib import Path
import sys

def unsafe_matmul_nodes(data):
    unsafe = set(map(int, data.get('cycle_nodes', [])))
    for cyc in data.get('cycles', []):
        a, b = (cyc['start'], cyc['end']) if isinstance(cyc, dict) else cyc
        unsafe.update(range(int(a), int(b) + 1))
    for cond in data.get('conditions', []):
        s, m = (cond['split'], cond['merge']) if isinstance(cond, dict) else cond
        unsafe.update(range(int(s) + 1, int(m)))
    return unsafe

def prepare_graph(data):
    result = copy.deepcopy(data)
    unsafe = unsafe_matmul_nodes(result)
    replacements = []
    for node in result.get('nodes', []):
        if node.get('op') != 'MatMul':
            continue
        if int(node['id']) in unsafe:
            node['op'] = 'Add'
            replacements.append({'node': node['id'], 'from': 'MatMul', 'to': 'Add',
                                 'reason': 'loop-or-conditional-control-flow'})
        else:
            node['tensor_len'] = 16
    result['matmul_replacements'] = replacements
    return result

if __name__ == '__main__':
    Path(sys.argv[2]).write_text(json.dumps(prepare_graph(json.loads(Path(sys.argv[1]).read_text())), indent=2) + '\n')
