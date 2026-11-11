#!/usr/bin/python3

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Sequence

import numpy as np

DTYPE_CHOICES = ("float32", "float16", "int32", "int16")
RANGE_CHOICES = ("normal", "signed_unit", "wide", "minmax", "boundary", "special", "mixed", "extreme")
PROFILE_CHOICES = (
    "legacy",
    "float_normal",
    "float_signed",
    "float_wide",
    "float_extreme",
    "float_special",
    "int16_bounds",
    "int32_bounds",
    "mixed_stress",
)

DTYPE_MAP = {
    "float32": np.float32,
    "float16": np.float16,
    "int32": np.int32,
    "int16": np.int16,
}

SPECIAL_FLOAT_TOKENS = {
    "nan": np.nan,
    "inf": np.inf,
    "+inf": np.inf,
    "pinf": np.inf,
    "-inf": -np.inf,
    "ninf": -np.inf,
    "zero": 0.0,
    "negzero": -0.0,
    "subnormal": np.float32(1e-45),
    "neg_subnormal": np.float32(-1e-45),
    "min": np.finfo(np.float32).min,
    "max": np.finfo(np.float32).max,
    "tiny": np.finfo(np.float32).tiny,
    "eps": np.finfo(np.float32).eps,
    "neg_min": np.finfo(np.float32).min,
    "neg_max": -np.finfo(np.float32).max,
    "neg_tiny": -np.finfo(np.float32).tiny,
    "half_max": np.float32(65504.0),
    "neg_half_max": np.float32(-65504.0),
}

@dataclass
class GenConfig:
    tensor_len: int = 64
    dtype: str = "int32"
    range_mode: str = "normal"
    seed: int = 20260421
    special_tokens: Sequence[str] = ("nan", "inf", "-inf", "1.5", "-2.3", "12345.67")
    profile: Optional[str] = None
    emit_meta: bool = False
    randomize_all: bool = False

def resolve_tensor_len(explicit: Optional[int]) -> int:
    if explicit is not None and explicit > 0:
        return explicit
    try:
        value = int(os.environ.get("BENCHMARK_TENSOR_LEN", "64"))
    except ValueError:
        value = 64
    return value if value > 0 else 64

def apply_profile(profile: str, cfg: GenConfig) -> GenConfig:
    presets = {
        "legacy": GenConfig(dtype="int32", range_mode="normal", seed=20260421),
        "float_normal": GenConfig(dtype="float32", range_mode="normal", seed=20260421),

        "float_signed": GenConfig(dtype="float32", range_mode="signed_unit", seed=20260421),
        "float_wide": GenConfig(dtype="float32", range_mode="wide", seed=20260421),
        "float_extreme": GenConfig(dtype="float32", range_mode="minmax", seed=20260421),
        "float_special": GenConfig(
            dtype="float32",
            range_mode="special",
            seed=20260421,
            special_tokens=("nan", "inf", "-inf", "1.5", "-2.3", "12345.67"),
        ),
        "int16_bounds": GenConfig(dtype="int16", range_mode="minmax", seed=20260421),
        "int32_bounds": GenConfig(dtype="int32", range_mode="minmax", seed=20260421),
        "mixed_stress": GenConfig(dtype="float32", range_mode="mixed", seed=20260421),
    }
    if profile not in presets:
        raise ValueError(f"unknown profile {profile!r}")
    base = presets[profile]
    cfg.dtype = base.dtype
    cfg.range_mode = base.range_mode
    cfg.seed = base.seed
    cfg.special_tokens = base.special_tokens
    cfg.profile = profile
    return cfg

def parse_special_tokens(raw: Optional[str]) -> List[str]:
    if not raw:
        return list(GenConfig().special_tokens)
    return [part.strip() for part in raw.split(",") if part.strip()]

def _dtype_limits(dtype: np.dtype) -> tuple[float, float]:
    if np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        return float(info.min), float(info.max)
    info = np.iinfo(dtype)
    return float(info.min), float(info.max)

def _parse_special_scalar(token: str, dtype: np.dtype) -> float:
    key = token.lower()
    if key in SPECIAL_FLOAT_TOKENS:
        return float(SPECIAL_FLOAT_TOKENS[key])
    try:
        return float(np.array(token, dtype=dtype))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid special token {token!r}") from exc

def _fill_from_pool(out: np.ndarray, pool: Sequence[float], rng: np.random.Generator) -> None:
    if not pool:
        return
    if len(pool) >= out.size:
        out[:] = np.asarray(pool[: out.size], dtype=np.float32)
        return
    idx = rng.integers(0, len(pool), size=out.size)
    out[:] = np.asarray([pool[i] for i in idx], dtype=np.float32)

def generate_input_1(cfg: GenConfig) -> np.ndarray:
    dtype = DTYPE_MAP[cfg.dtype]
    rng = np.random.default_rng(cfg.seed)
    out = np.empty(cfg.tensor_len, dtype=np.float32)
    lo, hi = _dtype_limits(dtype)

    if cfg.range_mode == "normal":
        if np.issubdtype(dtype, np.floating):
            out[:] = rng.uniform(1.0, 100.0, size=cfg.tensor_len).astype(dtype).astype(np.float32)
        else:
            out[:] = rng.integers(1, 100, size=cfg.tensor_len, dtype=dtype).astype(np.float32)
        return out

    if cfg.range_mode == "signed_unit":

        if np.issubdtype(dtype, np.floating):
            out[:] = rng.uniform(-1.0, 1.0, size=cfg.tensor_len).astype(dtype).astype(np.float32)
        else:
            out[:] = rng.integers(-99, 100, size=cfg.tensor_len, dtype=dtype).astype(np.float32)
        return out

    if cfg.range_mode == "wide":
        if np.issubdtype(dtype, np.floating):
            if dtype == np.float16:
                out[:] = rng.uniform(-65504.0, 65504.0, size=cfg.tensor_len).astype(dtype).astype(np.float32)
            else:
                out[:] = rng.uniform(-1e6, 1e6, size=cfg.tensor_len).astype(dtype).astype(np.float32)
        elif dtype == np.int16:
            out[:] = rng.integers(-30000, 30001, size=cfg.tensor_len, dtype=dtype).astype(np.float32)
        else:
            out[:] = rng.integers(-1000000, 1000001, size=cfg.tensor_len, dtype=dtype).astype(np.float32)
        return out

    if cfg.range_mode == "minmax":
        if np.issubdtype(dtype, np.floating):
            pool = [lo, hi, 0.0, -0.0, 1.0, -1.0]
            if dtype == np.float32:
                pool.extend([np.nextafter(0.0, 1.0), np.nextafter(0.0, -1.0)])
        else:
            pool = [lo, hi, 0.0, -1.0, 1.0]
            if dtype == np.int16:
                pool.extend([-32768.0, 32767.0])
            elif dtype == np.int32:
                pool.extend([float(np.iinfo(np.int32).min), float(np.iinfo(np.int32).max)])
        _fill_from_pool(out, pool, rng)
        return out

    if cfg.range_mode == "boundary":
        if np.issubdtype(dtype, np.floating):
            pool = [lo, hi, 0.0, -0.0, 1.0, -1.0, np.finfo(dtype).eps, -np.finfo(dtype).eps]
        elif dtype == np.int16:
            pool = [-32768.0, 32767.0, -1.0, 0.0, 1.0, 255.0, -255.0]
        else:
            pool = [
                float(np.iinfo(np.int32).min),
                float(np.iinfo(np.int32).max),
                -1.0,
                0.0,
                1.0,
                127.0,
                -128.0,
            ]
        _fill_from_pool(out, pool, rng)
        return out

    if cfg.range_mode == "special":
        if np.issubdtype(dtype, np.floating):
            pool = [_parse_special_scalar(token, dtype) for token in cfg.special_tokens]
            _fill_from_pool(out, pool, rng)
            return out
        pool = [lo, hi, 0.0, -1.0, 1.0]
        _fill_from_pool(out, pool, rng)
        return out

    if cfg.range_mode == "extreme":
        if np.issubdtype(dtype, np.floating):
            pool = [
                float(lo), float(hi), 0.0, -0.0, 1.0, -1.0,
                float(np.finfo(dtype).eps), float(-np.finfo(dtype).eps),
                float(np.finfo(dtype).tiny), float(-np.finfo(dtype).tiny),
                np.nan, np.inf, -np.inf,
                float(np.nextafter(0.0, 1.0)), float(np.nextafter(0.0, -1.0)),
            ]
            if dtype == np.float32:
                pool.extend([1e-10, -1e-10, 1e10, -1e10, 3.14159265, -2.71828183])
            elif dtype == np.float16:
                pool.extend([1e-3, -1e-3, 65504.0, -65504.0])
        else:
            pool = [lo, hi, 0.0, 1.0, -1.0, lo + 1, hi - 1, 127.0, -128.0]
        _fill_from_pool(out, pool, rng)
        return out

    if cfg.range_mode == "mixed":
        normal = generate_input_1(
            GenConfig(
                tensor_len=cfg.tensor_len,
                dtype=cfg.dtype,
                range_mode="normal",
                seed=cfg.seed + 1,
            )
        )
        boundary = generate_input_1(
            GenConfig(
                tensor_len=cfg.tensor_len,
                dtype=cfg.dtype,
                range_mode="boundary",
                seed=cfg.seed + 2,
            )
        )
        special = generate_input_1(
            GenConfig(
                tensor_len=cfg.tensor_len,
                dtype="float32",
                range_mode="special",
                seed=cfg.seed + 3,
                special_tokens=cfg.special_tokens,
            )
        )
        pick = rng.integers(0, 3, size=cfg.tensor_len)
        for i, mode in enumerate(pick):
            if mode == 0:
                out[i] = normal[i]
            elif mode == 1:
                out[i] = boundary[i]
            else:
                out[i] = special[i]
        return out

    raise ValueError(f"unsupported range mode {cfg.range_mode!r}")

def generate_input_2(tensor_len: int) -> np.ndarray:
    base = np.array([16, 6, 9, 7, 7, 3, 3, 19], dtype=np.int32)
    if tensor_len <= base.size:
        return base[:tensor_len].copy()
    out = np.empty(tensor_len, dtype=np.int32)
    out[: base.size] = base
    out[base.size :] = 1
    return out

def _make_cfg_for_tensor(base_cfg: GenConfig, seed_offset: int, force_float: bool = False, tensor_len: int = None) -> GenConfig:
    return GenConfig(
        tensor_len=tensor_len if tensor_len is not None else base_cfg.tensor_len,
        dtype="float32" if force_float else base_cfg.dtype,
        range_mode=base_cfg.range_mode,
        seed=base_cfg.seed + seed_offset,
        special_tokens=base_cfg.special_tokens,
    )

def generate_inputs(cfg: GenConfig) -> None:
    if cfg.profile:
        cfg = apply_profile(cfg.profile, cfg)

    m = n = k = 4
    rng = np.random.default_rng(cfg.seed)

    if cfg.randomize_all:

        cfg_a = _make_cfg_for_tensor(cfg, 100, force_float=True, tensor_len=m * k)
        cfg_b = _make_cfg_for_tensor(cfg, 200, force_float=True, tensor_len=k * n)
        cfg_bias = _make_cfg_for_tensor(cfg, 300, force_float=True, tensor_len=n)
        cfg_input2 = _make_cfg_for_tensor(cfg, 400)

        input_a_1d = generate_input_1(cfg_a)
        input_b_1d = generate_input_1(cfg_b)
        input_bias_1d = generate_input_1(cfg_bias)
        input_a = input_a_1d.reshape(m, k).astype(np.float32)
        input_b = input_b_1d.reshape(k, n).astype(np.float32)
        input_bias = input_bias_1d[:n].astype(np.float32)
        input_2 = generate_input_1(cfg_input2).astype(np.int32)
    else:
        input_a = rng.integers(1, 2, size=[m, k], dtype=np.int32).astype(np.float32)
        input_b = rng.integers(1, 2, size=[k, n], dtype=np.int32).astype(np.float32)
        input_bias = rng.integers(1, 2, size=[n], dtype=np.int32).astype(np.float32)
        input_2 = generate_input_2(cfg.tensor_len)

    input_1 = generate_input_1(cfg)

    os.makedirs("input", exist_ok=True)
    input_a.tofile("./input/x1_gm.bin")
    input_b.tofile("./input/x2_gm.bin")
    input_bias.tofile("./input/bias.bin")
    input_1.astype(np.float32).tofile("./input/input1_gm.bin")
    input_2.tofile("./input/input2_gm.bin")

    if cfg.emit_meta:
        meta = asdict(cfg)
        meta["input1_written_dtype"] = "float32"
        meta["input1_min"] = float(np.nanmin(input_1))
        meta["input1_max"] = float(np.nanmax(input_1))
        meta["input1_nan_count"] = int(np.isnan(input_1).sum())
        meta["input1_inf_count"] = int(np.isinf(input_1).sum())
        with open("./input/data_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

def build_config_from_env_and_args(args: argparse.Namespace) -> GenConfig:
    cfg = GenConfig(tensor_len=resolve_tensor_len(args.tensor_len))

    env_dtype = os.environ.get("BENCHMARK_DATA_DTYPE")
    env_range = os.environ.get("BENCHMARK_DATA_RANGE")
    env_seed = os.environ.get("BENCHMARK_DATA_SEED")
    env_profile = os.environ.get("BENCHMARK_DATA_PROFILE")
    env_special = os.environ.get("BENCHMARK_DATA_SPECIAL")

    if args.profile or env_profile:
        cfg.profile = args.profile or env_profile
    if args.dtype or env_dtype:
        cfg.dtype = (args.dtype or env_dtype).lower()
    if args.range_mode or env_range:
        cfg.range_mode = (args.range_mode or env_range).lower()
    if args.seed is not None or env_seed:
        cfg.seed = int(args.seed if args.seed is not None else env_seed)
    if args.special or env_special:
        cfg.special_tokens = parse_special_tokens(args.special or env_special)
    if args.emit_meta:
        cfg.emit_meta = True
    if args.randomize_all:
        cfg.randomize_all = True

    if cfg.dtype not in DTYPE_MAP:
        raise ValueError(f"unsupported dtype {cfg.dtype!r}, choose from {DTYPE_CHOICES}")
    if cfg.range_mode not in RANGE_CHOICES:
        raise ValueError(f"unsupported range {cfg.range_mode!r}, choose from {RANGE_CHOICES}")
    if cfg.profile and cfg.profile not in PROFILE_CHOICES:
        raise ValueError(f"unsupported profile {cfg.profile!r}, choose from {PROFILE_CHOICES}")
    return cfg

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate baremix input bins with configurable dtype/range/special values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tensor-len", type=int, default=None, help="Override BENCHMARK_TENSOR_LEN")
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default=None, help="Semantic generation dtype for input1")
    parser.add_argument("--range", dest="range_mode", choices=RANGE_CHOICES, default=None, help="Value range strategy")
    parser.add_argument(
        "--special",
        default=None,
        help="Comma-separated special scalars for --range special/mixed (e.g. nan,inf,-inf,1.5)",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default=None,
        help="Preset dtype/range combination (overrides --dtype/--range when set)",
    )
    parser.add_argument("--emit-meta", action="store_true", help="Write input/data_meta.json")
    parser.add_argument("--randomize-all", action="store_true",
                        help="Apply dtype/range randomization to ALL tensors (x1, x2, bias, input1, input2)")
    parser.add_argument("--list-options", action="store_true", help="Print supported options and exit")
    return parser

def print_options() -> None:
    print("dtypes:", ", ".join(DTYPE_CHOICES))
    print("ranges:", ", ".join(RANGE_CHOICES))
    print("profiles:", ", ".join(PROFILE_CHOICES))
    print("special tokens:", ", ".join(sorted(SPECIAL_FLOAT_TOKENS.keys())))
    print("env vars: BENCHMARK_TENSOR_LEN, BENCHMARK_DATA_DTYPE, BENCHMARK_DATA_RANGE,")
    print("          BENCHMARK_DATA_SEED, BENCHMARK_DATA_PROFILE, BENCHMARK_DATA_SPECIAL,")
    print("          BENCHMARK_DATA_RANDOMIZE_ALL")

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.list_options:
        print_options()
        return
    cfg = build_config_from_env_and_args(args)
    generate_inputs(cfg)

if __name__ == "__main__":
    main()
