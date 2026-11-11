#!/bin/bash

CURRENT_DIR=$(
    cd "$(dirname "${BASH_SOURCE:-$0}")"
    pwd
)

BUILD_TYPE="Debug"
INSTALL_PREFIX="${CURRENT_DIR}/out"

SHORT=r:,v:,i:,b:,p:
LONG=run-mode:,soc-version:,install-path:,build-type:,install-prefix:
OPTS=$(getopt -a --options "$SHORT" --longoptions "$LONG" -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
    -r | --run-mode)
        RUN_MODE="$2"
        shift 2
        ;;
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    -i | --install-path)
        ASCEND_INSTALL_PATH="$2"
        shift 2
        ;;
    -b | --build-type)
        BUILD_TYPE="$2"
        shift 2
        ;;
    -p | --install-prefix)
        INSTALL_PREFIX="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR]: Unexpected option: $1"
        break
        ;;
    esac
done

RUN_MODE_LIST="cpu sim npu"
if [[ " $RUN_MODE_LIST " != *" $RUN_MODE "* ]]; then
    echo "[ERROR]: RUN_MODE must be one of [$RUN_MODE_LIST]"
    exit 1
fi

VERSION_LIST="Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4"
if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "[ERROR]: SOC_VERSION must be in [$VERSION_LIST]"
    exit 1
fi

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

if [ "$RUN_MODE" = "cpu" ] && [ -f graph.json ]; then
    python3 - <<'PY_CPU_COMPAT'
import json
import sys
ops = {n.get("op") for n in json.load(open("graph.json")).get("nodes", [])}
unsupported = sorted(ops & {"MatMul", "Mmad"})
if unsupported:
    print("CPU compilation unsupported for current mixed-core cases: " + ", ".join(unsupported), file=sys.stderr)
    sys.exit(2)
PY_CPU_COMPAT
    _cpu_compat=$?
    if [ "$_cpu_compat" -ne 0 ]; then exit "$_cpu_compat"; fi
fi

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
echo "[INFO]: compile phase — soc_version=${SOC_VERSION}  build_type=${BUILD_TYPE}"
source "${_ASCEND_INSTALL_PATH}/bin/setenv.bash"

if [ "${RUN_MODE}" = "sim" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
elif [ "${RUN_MODE}" = "cpu" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib:${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${SOC_VERSION}:${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
fi

rm -rf build out 2>/dev/null || true
mkdir -p build
cmake -B build \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH} || { echo "[ERROR] cmake configure failed"; exit 1; }

cmake --build build -j || { echo "[ERROR] cmake build failed"; exit 1; }
cmake --install build || { echo "[ERROR] cmake install failed"; exit 1; }

rm -f ascendc_kernels_bbit
cp ./out/bin/ascendc_kernels_bbit ./ || { echo "[ERROR] binary copy failed"; exit 1; }

echo "[INFO]: $(date '+%F %T') compile done — binary at ${CURRENT_DIR}/ascendc_kernels_bbit"
