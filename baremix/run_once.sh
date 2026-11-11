#!/bin/bash

CURRENT_DIR=$(
    cd "$(dirname "${BASH_SOURCE:-$0}")"
    pwd
)

DATA_DIR=""
OUTPUT_DIR=""

SHORT=r:,v:,i:,d:,o:
LONG=run-mode:,soc-version:,install-path:,data-dir:,output-dir:
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
    -d | --data-dir)
        DATA_DIR="$2"
        shift 2
        ;;
    -o | --output-dir)
        OUTPUT_DIR="$2"
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

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
source "${_ASCEND_INSTALL_PATH}/bin/setenv.bash"

export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"

if [ "${RUN_MODE}" = "sim" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
elif [ "${RUN_MODE}" = "cpu" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib:${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${SOC_VERSION}:${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
fi

if [ ! -f "./ascendc_kernels_bbit" ]; then
    echo "[INFO]: ./ascendc_kernels_bbit not found, auto-compiling..."
    bash "${CURRENT_DIR}/compile.sh" -r "$RUN_MODE" -v "$SOC_VERSION" \
        ${ASCEND_INSTALL_PATH:+-i "$ASCEND_INSTALL_PATH"}
fi

rm -rf input output
mkdir -p output

if [ -n "$DATA_DIR" ]; then
    if [ -d "${DATA_DIR}/input" ]; then
        echo "[INFO]: linking input data from ${DATA_DIR}/input -> ./input"

        _data_src=$(cd "${DATA_DIR}/input" && pwd)
        if ln -s "$_data_src" ./input 2>/dev/null; then
            echo "[INFO]: symlink created: ./input -> $_data_src"
        else
            echo "[INFO]: symlink failed, falling back to copy"
            cp -r "${DATA_DIR}/input" ./input
        fi
    else
        echo "[ERROR]: --data-dir specified but ${DATA_DIR}/input not found"
        exit 1
    fi
else

    if [ -f scripts/gen_data.py ]; then
        echo "[INFO]: no --data-dir, generating default data via scripts/gen_data.py"
        python3 scripts/gen_data.py
    else
        echo "[ERROR]: no --data-dir and scripts/gen_data.py not found; cannot prepare input data"
        exit 1
    fi
fi

export LD_LIBRARY_PATH="$(pwd)/out/lib:$(pwd)/out/lib64:${_ASCEND_INSTALL_PATH}/lib64:${LD_LIBRARY_PATH:-}"
RUN_WITH_TOOLCHAIN=${RUN_WITH_TOOLCHAIN:-0}

if command -v stdbuf >/dev/null 2>&1; then
    _STDBUF=(stdbuf -oL -eL)
else
    _STDBUF=()
fi

echo "[INFO]: $(date '+%F %T') ready to execute ascendc_kernels_bbit (RUN_MODE=${RUN_MODE})"

set +e
if [[ "${RUN_WITH_TOOLCHAIN}" -eq 1 ]]; then
    if [ "${RUN_MODE}" = "npu" ]; then
        "${_STDBUF[@]}" msprof op --application=./ascendc_kernels_bbit
    elif [ "${RUN_MODE}" = "sim" ]; then
        "${_STDBUF[@]}" msprof op simulator --application=./ascendc_kernels_bbit
    else
        "${_STDBUF[@]}" ./ascendc_kernels_bbit
    fi
else
    "${_STDBUF[@]}" ./ascendc_kernels_bbit
fi
BBIT_EC=$?
set -e

echo "[INFO]: $(date '+%F %T') ascendc_kernels_bbit finished, exit code=${BBIT_EC}"

if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "${OUTPUT_DIR}/output"
    if [ -f ./output/output.bin ]; then
        cp ./output/output.bin "${OUTPUT_DIR}/output/output.bin"
        echo "[INFO]: output.bin saved to ${OUTPUT_DIR}/output/"
    fi

    echo "${BBIT_EC}" > "${OUTPUT_DIR}/exit_code.txt"
fi

if [ "${RUN_MODE}" = "sim" ]; then
    rm -f *.log *.dump *.vcd *.toml *_log
fi

if [[ "${BBIT_EC}" -ne 0 ]]; then
    echo "[WARN]: ascendc_kernels_bbit exited with code ${BBIT_EC}"
    exit "${BBIT_EC}"
fi
