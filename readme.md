# HardFuzz
This is the open resposity of the paper "HardFuzz: Testing DL Compiler Backends via Hardware-aware Program Generation".
## Requirements

- Python 3.10 or later and NumPy.
- CMake 3.16 or later and a C++17 compiler.
- Huawei CANN 9.0.0-beta.1 with Bisheng and AscendC.
- An Ascend NPU.

#### Usage

##### Test Input Generation

1. Generate computation graphs as DOT and JSON files:

```bash
bash run_generate_graphs.sh NODE_NUM GRAPH_NUM OUTPUT_DIR
```

For example, generate 10 graphs with 35 nodes each:

```bash
bash run_generate_graphs.sh 35 10 ./graphs_out
```

2. Convert a graph into an AscendC program:

```bash
bash run_convert_graph.sh JSON_PATH OUTPUT_DIR
```

For example:

```bash
bash run_convert_graph.sh ./graphs_out/graph_1.json ./baremix_graph
```

Keep the generated program directory directly under the repository so its relative interface headers can be found.

3. Compile and execute the program:

```bash
cd baremix_graph
bash run.sh -r npu -v Ascend910B3
```

##### Differential Testing

Run the following commands from the repository root.

Oracle 1 compares NPU outputs across `-O0`, `-O2`, and `-O3`:

```bash
bash tools/run_oracle1.sh ./baremix_graph --soc Ascend910B3
```

Oracle 2 compares CPU and NPU outputs for compatible programs:

```bash
bash tools/run_oracle2.sh ./baremix_graph --cpu-compatible --soc Ascend910B3
```

The current CPU adapter skips mixed-core MatMul/Mmad programs.

Oracle 3 checks compilation, execution, and output failures:

```bash
bash tools/run_oracle3.sh ./baremix_graph --soc Ascend910B3
```

Run all three oracles together:

```bash
python3 tools/run_case_differential.py ./baremix_graph --mode all --cpu-compatible --soc Ascend910B3
```

Inputs are shared across configurations, and reports are saved under the program's `differential_results` directory. Use `--run-timeout` to set an execution limit.
