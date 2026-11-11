#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

if __package__:
    from .oracle_compare import DTYPES, OutputError, compare_bins, read_tensor
else:
    from oracle_compare import DTYPES, OutputError, compare_bins, read_tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
FP_FLAGS = ["-fno-fast-math", "-ffp-contract=off"]
SKIP_COPY = {"build", "out", "input", "output", "ascendc_kernels_bbit",
             "differential_results", ".git", "__pycache__", "run_multi_data.log"}
MODES = {"build-variant": "oracle1", "cpu-npu": "oracle2"}

def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

def manifest(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*")) if p.is_file()}

def changed_files(root, reference):
    changed = []
    for name, digest in reference.items():
        p = Path(root) / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            changed.append(name)
    return changed

def run_process(command, cwd, env, log_path, timeout):
    started = time.monotonic()
    with Path(log_path).open("w") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            rc = proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            if isinstance(exc, KeyboardInterrupt):
                raise
            timed_out = True
            rc = 124
            log.write(f"\n[HardFuzz] timeout after {timeout}s; process group killed\n")
    return {"command": command, "cwd": str(cwd), "exit_status": rc,
            "signal": -rc if rc < 0 else (signal.SIGKILL if timed_out else None),
            "timed_out": timed_out, "duration_seconds": round(time.monotonic() - started, 3),
            "log": str(log_path)}

def prepare_source(case, results):
    snapshot = results / "source"
    snapshot.mkdir()
    skip = set(SKIP_COPY)
    if results.is_relative_to(case):
        skip.add(results.relative_to(case).parts[0])

    def ignore(directory, names):
        return [n for n in names if n in skip] if Path(directory) == case else [
            n for n in names if n in {".git", "__pycache__"}]

    shutil.copytree(case, snapshot / "case", ignore=ignore, symlinks=False)

    for dep in ("ops", "wrappers"):
        roots = list(case.parents) + [REPO_ROOT]
        source = next((root / dep for root in roots if (root / dep).is_dir()), None)
        if source:
            shutil.copytree(source, snapshot / dep, symlinks=False, ignore_dangling_symlinks=True,
                            ignore=shutil.ignore_patterns(".git", "__pycache__"))
    original = manifest(snapshot)
    dump_json(results / "original_source_manifest.json", original)
    helper = snapshot / "case" / "cmake" / "oracle_flags.cmake"
    helper.parent.mkdir(exist_ok=True)
    shutil.copy2(REPO_ROOT / "baremix/cmake/oracle_flags.cmake", helper)
    cmake = snapshot / "case" / "CMakeLists.txt"
    text = cmake.read_text()
    if not re.search(r"include\(\s*cmake/oracle_flags\.cmake(?:\s|\))", text):
        cmake.write_text(text.rstrip() + "\n\n# HardFuzz oracle: common build-only adaptation.\n"
                         "include(cmake/oracle_flags.cmake)\n")
    fixed = manifest(snapshot)
    dump_json(results / "source_manifest.json", fixed)
    adaptations = [name for name, digest in fixed.items() if original.get(name) != digest]
    return snapshot, fixed, adaptations

def prepare_input(snapshot, results, args, env):
    if args.input_dir or args.skip_gen_data:
        source = Path(args.input_dir).resolve() if args.input_dir else args.case_dir / "input"
        if not source.is_dir():
            raise OutputError(f"input directory does not exist: {source}")
        shutil.copytree(source, results / "input", symlinks=False)
        generation = {"method": "copy", "source": str(source)}
    else:
        generator = snapshot / "case/scripts/gen_data.py"
        if not generator.is_file():
            raise OutputError("missing scripts/gen_data.py; provide --input-dir or --skip-gen-data")
        generation = run_process([sys.executable, "scripts/gen_data.py"], snapshot / "case",
                                 env, results / "input_generation.log", args.input_timeout)
        if generation["exit_status"] != 0:
            return generation, None
        source = snapshot / "case/input"
        if not source.is_dir():
            raise OutputError("input generator succeeded but produced no input directory")
        shutil.copytree(source, results / "input", symlinks=False)
    hashes = manifest(results / "input")
    if not hashes or not any(name.endswith(".bin") for name in hashes):
        raise OutputError("no binary input tensors; refusing stale-input fallback")
    dump_json(results / "input_manifest.json", hashes)
    return generation, hashes

def expected_elements(args, snapshot):
    if args.expected_elements is not None:
        return args.expected_elements

    main = snapshot / "case/main.cpp"
    text = main.read_text() if main.is_file() else ""
    match = re.search(r"\bint\s+tensor_len\s*=\s*(\d+)\s*;", text)
    if match and args.dtype == "float32":
        count = int(match.group(1))
        if 'getenv("BENCHMARK_TENSOR_LEN")' in text:
            try:
                override = int(os.environ.get("BENCHMARK_TENSOR_LEN", "0"))
                if override > 0:
                    count = override
            except ValueError:
                pass
        return count
    raise OutputError("cannot infer output shape; supply --expected-elements (and --dtype)")

def verify_compile_flags(work, mode, opt, artifact):
    evidence = []
    for path in sorted((work / "build").rglob("flags.make")):
        rel = str(path.relative_to(work))
        host = "/ascendc_kernels_bbit.dir/" in rel
        cpu_kernel = "/ascendc_kernels_cpu.dir/" in rel
        npu_kernel = "_device-build/" in rel and bool(
            re.search(r"/device(?:_aiv|_aic)?_obj\.dir/", rel))
        if not (host or (cpu_kernel if mode == "cpu" else npu_kernel)):
            continue
        text = path.read_text(errors="replace")
        match = re.search(r"^CXX_FLAGS\s*=\s*(.*)$", text, re.M)
        if not match:
            continue
        tokens = shlex.split(match.group(1))
        opts = [t for t in tokens if re.fullmatch(r"-O(?:[0-3sgz]|fast)", t)]
        fast = [t for t in tokens if t in ("-ffast-math", "-fno-fast-math", "-Ofast")]
        contract = [t for t in tokens if t.startswith("-ffp-contract=")]
        ok = bool(opts and opts[-1] == f"-O{opt}" and fast and fast[-1] == "-fno-fast-math"
                  and contract and contract[-1] == "-ffp-contract=off")
        evidence.append({"file": rel, "role": "host" if host else "kernel",
                         "flags": match.group(1), "effective_optimization": opts[-1] if opts else None,
                         "valid": ok})
    dump_json(artifact / "compile_flags.json", evidence)
    (artifact / "compile_params.txt").write_text("\n".join(
        f"{e['file']}\n  {e['flags']}\n  valid={e['valid']}" for e in evidence) + "\n")
    if not any(e["role"] == "host" for e in evidence):
        return "missing actual host compiler flag evidence"
    if not any(e["role"] == "kernel" for e in evidence):
        return "missing actual kernel compiler flag evidence"
    if not all(e["valid"] for e in evidence):
        return "effective host/kernel flags do not match the requested oracle configuration"
    return None

def failure(stage, config, reason, results, record=None):
    return {"stage": stage, "configuration": config, "reason": str(reason),
            "input": str(results / "input"),
            "input_manifest": str(results / "input_manifest.json"),
            "source_manifest": str(results / "source_manifest.json"),
            "exit_status": record.get("exit_status") if record else None,
            "diagnostics": record}

def run_config(name, mode, opt, snapshot, source_hashes, input_hashes, args, results, env):
    artifact = results / name
    artifact.mkdir()
    clone = results / "work" / name
    shutil.copytree(snapshot, clone, ignore=shutil.ignore_patterns("input", "output", "__pycache__"))
    work = clone / "case"

    private_data = artifact / "data"
    shutil.copytree(results / "input", private_data / "input")
    cfg = {"name": name, "run_mode": mode, "optimization": f"-O{opt}", "fp_flags": FP_FLAGS,
           "build_type": "Release", "status": "FAIL", "work_dir": str(work),
           "output": str(artifact / "output.bin")}
    cfg_env = dict(env, HARDFUZZ_ORACLE_OPT=str(opt))
    command = ["bash", "compile.sh", "-r", mode, "-v", args.soc, "-b", "Release"]
    if args.install_path:
        command += ["-i", args.install_path]
    print(f"[{name}] compile", flush=True)
    build = run_process(command, work, cfg_env, artifact / "compile.log", args.compile_timeout)
    cfg["compile"] = build
    flags_error = verify_compile_flags(work, mode, opt, artifact)
    if build["exit_status"] != 0:
        return cfg, failure("compile", name, "compilation failed or timed out", results, build)
    mutation = changed_files(clone, source_hashes)
    if mutation:
        return cfg, failure("source_integrity", name, f"source changed during build: {mutation}", results, build)
    if flags_error:
        return cfg, failure("configuration_validation", name, flags_error, results,
                            {**build, "flags": str(artifact / "compile_flags.json")})
    if not (work / "ascendc_kernels_bbit").is_file():
        return cfg, failure("compile_artifact", name, "build succeeded but executable is missing", results, build)
    command = ["bash", "run_once.sh", "-r", mode, "-v", args.soc,
               "--data-dir", str(private_data)]
    if args.install_path:
        command += ["-i", args.install_path]
    print(f"[{name}] run", flush=True)
    run = run_process(command, work, cfg_env, artifact / "run.log", args.run_timeout)
    cfg["run"] = run
    output = work / args.output_file

    cpu_errors = []
    if mode == "cpu":
        cpu_errors = [line for line in Path(run["log"]).read_text(errors="replace").splitlines()
                      if re.search(r"\[(?:ERROR|error)\]|SIG(?:SEGV|BUS|ABRT|ILL|FPE) Signal", line)]
        if cpu_errors:
            cfg["cpu_simulator_errors"] = cpu_errors
    if run["exit_status"] != 0 or cpu_errors:
        if output.is_file():
            shutil.copy2(output, artifact / "failed_output.bin")
        return cfg, failure("run", name, "CPU simulator reported an internal error" if cpu_errors else "execution failed or timed out", results, run)
    if manifest(private_data / "input") != input_hashes or manifest(work / "input") != input_hashes:
        return cfg, failure("input_integrity", name, "runtime modified the fixed input", results, run)
    mutation = changed_files(clone, source_hashes)
    if mutation:
        return cfg, failure("source_integrity", name, f"source changed during run: {mutation}", results, run)
    try:
        read_tensor(output, args.dtype, args.expected_elements)
    except (OutputError, OSError) as exc:
        if output.is_file():
            shutil.copy2(output, artifact / "failed_output.bin")
        return cfg, failure("output_validation", name, exc, results, run)
    shutil.copy2(output, artifact / "output.bin")
    cfg["status"] = "PASS"
    cfg["output_sha256"] = hashlib.sha256((artifact / "output.bin").read_bytes()).hexdigest()
    return cfg, None

def numerical_oracle(names, configs, args, results):
    if any(name not in configs or configs[name]["status"] != "PASS" for name in names):
        return {"status": "SKIPPED_FAILED_CONFIGURATION", "comparisons": []}
    comparisons = []
    for a, b in itertools.combinations(names, 2):
        compared = compare_bins(results / a / "output.bin", results / b / "output.bin",
                                args.threshold, args.dtype, args.expected_elements)
        compared["pair"] = [a, b]
        comparisons.append(compared)
    return {"status": "FAIL" if any(c["status"] == "FAIL" for c in comparisons) else "PASS",
            "comparisons": comparisons}

def write_report(summary, results):
    failures = summary["oracle3"]["failures"]
    summary["oracle3"]["status"] = "CANDIDATE_FAILURE" if failures else "PASS"
    summary["comparisons"] = [c for key in ("oracle1", "oracle2")
                              for c in summary[key].get("comparisons", [])]
    summary["status"] = "FAIL" if failures or any(
        summary[key]["status"] == "FAIL" for key in ("oracle1", "oracle2")) else "PASS"
    if summary["mode"] == "oracle2" and summary["oracle2"]["status"] == "SKIPPED_UNSUPPORTED_CASE":
        summary["status"] = "SKIP"
    dump_json(results / "summary.json", summary)
    lines = [f"HardFuzz oracle report: {summary['status']}", f"Case: {summary['case_dir']}",
             f"Mode: {summary['mode']}; SoC: {summary['soc']}; threshold: {summary['threshold']}"]
    for key in ("oracle1", "oracle2", "oracle3"):
        lines.append(f"{key}: {summary[key]['status']}")
    for compared in summary["comparisons"]:
        lines.append(f"  {compared['pair']}: {compared['status']} "
                     f"max_abs_diff={compared['max_abs_diff']}")
    for item in failures:
        lines.append(f"  FAILURE {item['configuration']}/{item['stage']}: {item['reason']} "
                     f"(exit={item['exit_status']})")
    lines.append(f"Artifacts: {results}")
    (results / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--mode", choices=("all", "oracle1", "oracle2", "oracle3", "build-variant", "cpu-npu"),
                        default="all")
    parser.add_argument("--cpu-compatible", action="store_true",
                        help="confirm the selected case has CPU/NPU-compatible interface and invocation semantics")
    parser.add_argument("--soc", default="Ascend910B3")
    parser.add_argument("--install-path")
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--results-dir", type=Path, help="new directory; existing results are never overwritten")
    parser.add_argument("--input-dir", type=Path, help="fixed input directory containing .bin files")
    parser.add_argument("--skip-gen-data", action="store_true", help="copy CASE_DIR/input without generation")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--expected-elements", type=int, help="required unless baremix tensor_len can be inferred")
    parser.add_argument("--output-file", default="output/output.bin", help="expected output relative to the case directory")
    parser.add_argument("--compile-timeout", type=float, default=300)
    parser.add_argument("--run-timeout", type=float, default=120)
    parser.add_argument("--input-timeout", type=float, default=60)
    args = parser.parse_args(argv)
    args.mode = MODES.get(args.mode, args.mode)
    if args.mode == "oracle2" and not args.cpu_compatible:
        parser.error("Oracle 2 requires --cpu-compatible after checking interface/invocation semantics")
    if not math.isfinite(args.threshold) or args.threshold < 0:
        parser.error("--threshold must be finite and nonnegative")
    if args.expected_elements is not None and args.expected_elements <= 0:
        parser.error("--expected-elements must be positive")
    for name in ("compile_timeout", "run_timeout", "input_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    output = Path(args.output_file)
    if output.is_absolute() or ".." in output.parts or not output.parts or output.parts[0] != "output":
        parser.error("--output-file must be under output/ in the case directory")
    args.case_dir = args.case_dir.resolve()
    for name in ("CMakeLists.txt", "compile.sh", "run_once.sh"):
        if not (args.case_dir / name).is_file():
            parser.error(f"case is missing {name}: {args.case_dir}")
    return args

def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    results = (args.results_dir or args.case_dir / "differential_results" / stamp).resolve()
    try:
        results.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"cannot create fresh results directory: {exc}", file=sys.stderr)
        return 2
    print(f"[results] {results}", flush=True)
    env = dict(os.environ)
    for key in ("EXTRA_CXX_FLAGS", "CXXFLAGS", "CFLAGS", "CPPFLAGS", "LDFLAGS", "HARDFUZZ_ORACLE_OPT"):
        env.pop(key, None)
    env.update(RUN_WITH_TOOLCHAIN="0", CMAKE_GENERATOR="Unix Makefiles")
    dump_json(results / "environment.json", {k: v for k, v in env.items()
              if k.startswith(("BENCHMARK_", "ASCEND_", "CMAKE_")) or
              k in ("PATH", "LD_LIBRARY_PATH", "RUN_WITH_TOOLCHAIN")})
    summary = {"case_dir": str(args.case_dir), "mode": args.mode, "soc": args.soc,
               "threshold": args.threshold, "dtype": args.dtype, "results_dir": str(results),
               "cpu_compatible": args.cpu_compatible, "configurations": {},
               "oracle1": {"status": "NOT_REQUESTED", "comparisons": []},
               "oracle2": {"status": "NOT_REQUESTED", "comparisons": []},
               "oracle3": {"status": "PASS", "failures": []}}
    if args.mode == "all" and not args.cpu_compatible:
        summary["oracle2"]["status"] = "SKIPPED_INCOMPATIBLE_OR_UNCONFIRMED"
    graph_path = args.case_dir / "graph.json"
    if args.cpu_compatible and graph_path.is_file():
        graph = json.loads(graph_path.read_text())
        unsupported = sorted({str(n.get("op")) for n in graph.get("nodes", [])
                              if n.get("op") in {"MatMul", "Mmad"}})
        if unsupported:
            summary["oracle2"] = {"status": "SKIPPED_UNSUPPORTED_CASE", "comparisons": [],
                                  "reason": "Current CANN CPU mixed-core simulation is unsupported for: " + ", ".join(unsupported)}
            args.cpu_compatible = False
            if args.mode == "oracle2":
                write_report(summary, results)
                return 2
    stage, current = "source_preparation", "shared"
    try:
        snapshot, source_hashes, adaptations = prepare_source(args.case_dir, results)
        summary["source_adaptations"] = adaptations
        stage = "input_generation"
        generation, input_hashes = prepare_input(snapshot, results, args, env)
        summary["input_generation"] = generation
        if input_hashes is None:
            summary["oracle3"]["failures"].append(failure(stage, current, "input generation failed", results, generation))
            raise OutputError("input generation failed; no configurations executed")
        if changed_files(snapshot, source_hashes):
            raise OutputError("input generator changed the fixed source")
        stage = "output_schema"
        args.expected_elements = expected_elements(args, snapshot)
        summary["expected_elements"] = args.expected_elements
        if args.mode == "oracle2":
            requested = [("npu_O2", "npu", 2)]
        else:
            requested = [(f"npu_O{opt}", "npu", opt) for opt in (0, 2, 3)]
        if args.cpu_compatible and args.mode in ("all", "oracle2", "oracle3"):
            requested.append(("cpu_O2", "cpu", 2))
        for name, mode, opt in requested:
            stage, current = "configuration", name
            config, error = run_config(name, mode, opt, snapshot, source_hashes, input_hashes,
                                       args, results, env)
            summary["configurations"][name] = config
            if error:
                summary["oracle3"]["failures"].append(error)

            dump_json(results / "summary.json", summary)
        stage, current = "numerical_comparison", "shared"
        if args.mode in ("all", "oracle1"):
            summary["oracle1"] = numerical_oracle(["npu_O0", "npu_O2", "npu_O3"],
                                                  summary["configurations"], args, results)
        if args.mode in ("all", "oracle2") and args.cpu_compatible:
            summary["oracle2"] = numerical_oracle(["cpu_O2", "npu_O2"],
                                                  summary["configurations"], args, results)
    except (OSError, ValueError, shutil.Error) as exc:
        if not summary["oracle3"]["failures"] or str(exc) != "input generation failed; no configurations executed":
            summary["oracle3"]["failures"].append(failure(stage, current, exc, results))
        for key in ("oracle1", "oracle2"):
            if (args.mode == key or args.mode == "all") and summary[key]["status"] == "NOT_REQUESTED":
                summary[key]["status"] = "SKIPPED_FAILED_CONFIGURATION"
    except KeyboardInterrupt:
        summary["oracle3"]["failures"].append(failure(stage, current, "interrupted; process group killed", results))
        write_report(summary, results)
        return 130
    write_report(summary, results)
    return 0 if summary["status"] == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
