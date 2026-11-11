#!/bin/bash
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

BUILD_TYPE="Debug"
INSTALL_PREFIX="${CURRENT_DIR}/out"

SHORT=r:,v:,i:,b:,p:,
LONG=run-mode:,soc-version:,install-path:,build-type:,install-prefix:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
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
    echo "[ERROR]: RUN_MODE error, This sample only support specify cpu, sim or npu!"
    exit -1
fi

VERSION_LIST="Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4"
if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "[ERROR]: SOC_VERSION should be in [$VERSION_LIST]"
    exit -1
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

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
echo "[INFO]: Current compile soc version is ${SOC_VERSION}"
source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"
if [ "${RUN_MODE}" = "sim" ]; then

    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
elif [ "${RUN_MODE}" = "cpu" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib:${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${SOC_VERSION}:${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
fi

set -e
rm -rf build out
mkdir -p build
cmake -B build \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
cmake --build build -j
cmake --install build

rm -f ascendc_kernels_bbit
cp ./out/bin/ascendc_kernels_bbit ./
rm -rf input output
mkdir -p input output
python3 scripts/gen_data.py

export LD_LIBRARY_PATH="$(pwd)/out/lib:$(pwd)/out/lib64:${_ASCEND_INSTALL_PATH}/lib64:${LD_LIBRARY_PATH:-}"
RUN_WITH_TOOLCHAIN=${RUN_WITH_TOOLCHAIN:-0}
if command -v stdbuf >/dev/null 2>&1; then
    _STDBUF=(stdbuf -oL -eL)
else
    _STDBUF=()
fi
echo "[INFO]: $(date '+%F %T') Build complete; executing ascendc_kernels_bbit (RUN_MODE=${RUN_MODE})"
echo "[INFO]: Device execution may take time; use an external timeout when running generated programs."

set +e
if [[ "${RUN_WITH_TOOLCHAIN}" -eq 1 ]]; then
    if [ "${RUN_MODE}" = "npu" ]; then
        "${_STDBUF[@]}" msprof op --application=./ascendc_kernels_bbit
    elif [ "${RUN_MODE}" = "sim" ]; then
        "${_STDBUF[@]}" msprof op simulator --application=./ascendc_kernels_bbit
    elif [ "${RUN_MODE}" = "cpu" ]; then
        "${_STDBUF[@]}" ./ascendc_kernels_bbit
    else
        "${_STDBUF[@]}" ./ascendc_kernels_bbit
    fi
else
    "${_STDBUF[@]}" ./ascendc_kernels_bbit
fi
BBIT_EC=$?
set -e
echo "[INFO]: $(date '+%F %T') ascendc_kernels_bbit finished; exit code=${BBIT_EC}"
if [[ "${BBIT_EC}" -ne 0 ]]; then
    exit "${BBIT_EC}"
fi

if [ "${RUN_MODE}" = "sim" ]; then
    rm -f *.log *.dump *.vcd *.toml *_log
fi

echo ""
