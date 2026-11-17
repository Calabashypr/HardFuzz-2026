#!/usr/bin/env bash
# Load CANN 9 in the current shell.
export HARDFUZZ_CANN9_ROOT="${HARDFUZZ_CANN9_ROOT:-/usr/local/Ascend/cann-9.0.0-beta.1-isolated/cann}"
if [[ ! -r "$HARDFUZZ_CANN9_ROOT/compiler/version.info" ]] ||
   ! grep -qx 'Version=9.0.0-beta.1' "$HARDFUZZ_CANN9_ROOT/compiler/version.info"; then
    echo "[HardFuzz] Expected CANN 9.0.0-beta.1 at $HARDFUZZ_CANN9_ROOT" >&2
    return 1 2>/dev/null || exit 1
fi
if [[ ! -r "$HARDFUZZ_CANN9_ROOT/bin/setenv.bash" ]]; then
    echo "[HardFuzz] Missing CANN9 bin/setenv.bash" >&2
    return 1 2>/dev/null || exit 1
fi
_hf_remove_old_sdk_paths() {
    local name="$1" item result=''
    local -a items
    IFS=: read -r -a items <<< "${!name-}"
    for item in "${items[@]}"; do
        case "$item" in
            /usr/local/Ascend/ascend-toolkit/*|/usr/local/Ascend/cann-*|"$HOME"/Ascend/ascend-toolkit/*) continue ;;
        esac
        [[ -z "$item" ]] && continue
        result="${result:+$result:}$item"
    done
    printf -v "$name" '%s' "$result"
    export "$name"
}
for _hf_variable in PATH LD_LIBRARY_PATH PYTHONPATH CMAKE_PREFIX_PATH; do
    _hf_remove_old_sdk_paths "$_hf_variable"
done
# Clear the old SDK override.
unset ASCEND_OPP_KERNEL_PATH
source "$HARDFUZZ_CANN9_ROOT/bin/setenv.bash"
_hf_status=$?
unset -f _hf_remove_old_sdk_paths
unset _hf_variable
if [[ $_hf_status -ne 0 ]]; then
    unset _hf_status
    return 1 2>/dev/null || exit 1
fi
unset _hf_status
echo "[HardFuzz] CANN 9.0.0-beta.1: $ASCEND_HOME_PATH"
