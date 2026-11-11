#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "Usage: $0 NODES NUMS [OUTDIR] [OPERATOR_SELECTION] [GRAPH_STRUCTURE]"
  exit 1
fi
NODES="$1"
NUMS="$2"
OUTDIR="${3:-graphs_out}"
OPERATOR_SELECTION="${4:-${OPERATOR_SELECTION:-mcmc}}"
GRAPH_STRUCTURE="${5:-${GRAPH_STRUCTURE:-structured}}"
MCMC_TAU="${MCMC_TAU:-2.8}"

OPS_BUDGET="${OPS_BUDGET:-0}"
if ! [[ "$NODES" =~ ^[0-9]+$ ]] || [ "$NODES" -le 0 ]; then
  echo "NODES must be a positive integer"
  exit 1
fi
if ! [[ "$NUMS" =~ ^[0-9]+$ ]] || [ "$NUMS" -le 0 ]; then
  echo "NUMS must be a positive integer"
  exit 1
fi
if [[ "$OPERATOR_SELECTION" != "mcmc" && "$OPERATOR_SELECTION" != "random" ]]; then
  echo "OPERATOR_SELECTION must be 'mcmc' or 'random'"
  exit 1
fi
if [[ "$GRAPH_STRUCTURE" != "structured" && "$GRAPH_STRUCTURE" != "random-dag" ]]; then
  echo "GRAPH_STRUCTURE must be 'structured' or 'random-dag'"
  exit 1
fi
if [ "$OPS_BUDGET" != "0" ] && [ "$OPS_BUDGET" != "1" ]; then
  echo "OPS_BUDGET must be 0 or 1"
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$OUTDIR" = /* ]]; then
    OUTDIR_ABS="$OUTDIR"
else
    OUTDIR_ABS="$SCRIPT_DIR/$OUTDIR"
fi
mkdir -p "$OUTDIR_ABS"
MCMC_STATE="$OUTDIR_ABS/.mcmc_state"

SEED_BASE="${SEED_BASE:-1}"
if ! [[ "$SEED_BASE" =~ ^[0-9]+$ ]] || [ "$SEED_BASE" -lt 1 ]; then
  echo "SEED_BASE must be a positive integer"
  exit 1
fi
SEEDS="$(seq "$SEED_BASE" "$((SEED_BASE + NUMS - 1))")"
readarray -t SEED_ARR <<< "$SEEDS"
for i in "${!SEED_ARR[@]}"; do
  SEED_I="${SEED_ARR[$i]}"
  SELECTION_ARGS=(--operator-selection "$OPERATOR_SELECTION")
  if [ "$OPERATOR_SELECTION" = "mcmc" ]; then
    SELECTION_ARGS+=(--mcmc-tau "$MCMC_TAU" --mcmc-state "$MCMC_STATE")
    if [ "$i" -eq 0 ]; then
      SELECTION_ARGS+=(--reset-mcmc-state)
    fi
  fi
  if [ "$OPS_BUDGET" = "1" ]; then
    SELECTION_ARGS+=(--ops-budget)
  fi
  python3 "$SCRIPT_DIR/generate_graph.py" \
    --nodes "$NODES" \
    --seed "$SEED_I" \
    --dot "$OUTDIR_ABS/graph_${SEED_I}.dot" \
    --json "$OUTDIR_ABS/graph_${SEED_I}.json" \
    --graph-structure "$GRAPH_STRUCTURE" \
    --nested-loop-prob 0.9 \
    --nested-loop-depth-max 6 \
    --cycle-len-min 3 \
    --cycle-len-max 20 \
    "${SELECTION_ARGS[@]}"
done
