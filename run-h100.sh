#!/usr/bin/env bash
set -euo pipefail
umask 077

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

mode=${1:-}
case "$mode" in
    install|smoke|bench) ;;
    *) fail "Usage: bash run-h100.sh {install|smoke|bench}" ;;
esac

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
storage=${KANDINSKY_H100_STORE:-$root}
assets=https://raw.githubusercontent.com/Mishak7/.../main
repo="$root/kandinsky-s2v"
revision=6d319f0
config="$repo/src/kandinsky_s2v/configs/s2v_lite_distill.yaml"

export UV_PROJECT_ENVIRONMENT="$storage/venv"
export UV_CACHE_DIR="$storage/uv-cache"
export UV_PYTHON_INSTALL_DIR="$storage/python"
export XDG_CACHE_HOME="$storage/cache"
export HF_HOME="$storage/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python="$UV_PROJECT_ENVIRONMENT/bin/python"

if [[ $mode == install ]]; then
    export CUDA_VISIBLE_DEVICES=""
    uv=$(command -v uv || true)
    if [[ -x "$root/bin/uv" ]]; then
        uv="$root/bin/uv"
    fi
    [[ -n "$uv" ]] || fail "uv is missing. Install it to the project's bin directory first."
    mkdir -p "$storage"
    df -h "$root" "$storage"
    if [[ ! -e "$repo" ]]; then
        git clone https://github.com/kandinskylab/kandinsky-s2v.git "$repo"
        git -C "$repo" checkout --detach "$revision"
    fi
    [[ $(git -C "$repo" rev-parse --short=7 HEAD) == "$revision" ]] ||
        fail "Existing checkout is not at $revision: $repo"
    cd "$repo"
    # The original lock contains the yanked PyAV 18.1.0 build.
    if awk '/^name = "av"$/ { getline; found = ($0 == "version = \"18.1.0\"") } END { exit !found }' uv.lock; then
        UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-6}" "$uv" lock --upgrade-package av==18.1.1
    fi
    UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-6}" UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=2 \
        "$uv" sync --locked --no-dev --python 3.13
    exit 0
fi

[[ -x "$python" && -s "$config" ]] || fail "Run the install step first."

fetch_verified() {
    local relative=$1 expected=$2 target temporary
    target="$root/$relative"
    if [[ -f "$target" && $(sha256sum "$target" | awk '{print $1}') == "$expected" ]]; then
        return
    fi

    temporary=$(mktemp "$target.download.XXXXXX")
    if ! curl -fsSL --retry 2 "$assets/$relative" -o "$temporary" ||
        [[ $(sha256sum "$temporary" | awk '{print $1}') != "$expected" ]]; then
        rm -f -- "$temporary"
        fail "Could not download and verify $relative"
    fi
    mv -- "$temporary" "$target"
}

gpu_ids=(4)
if [[ $mode == bench ]]; then
    gpu_ids=(4 5 6 7)
fi

gpu_uuids=()
for gpu in "${gpu_ids[@]}"; do
    gpu_info=$(nvidia-smi -i "$gpu" --query-gpu=uuid,name,memory.used,utilization.gpu --format=csv,noheader,nounits)
    printf 'GPU %s: %s\n' "$gpu" "$gpu_info"
    if ! uuid=$(awk -F', *' '
        NF == 4 && $1 ~ /^GPU-[0-9a-fA-F-]+$/ && $2 ~ /H100/ &&
        $3 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/ && $3 <= 128 && $4 == 0 {
            print $1
            valid++
        }
        END { exit !(NR == 1 && valid == 1) }
    ' <<<"$gpu_info"); then
        fail "GPU $gpu is busy or does not match the expected H100. No benchmark was started."
    fi
    gpu_uuids+=("$uuid")
done

mkdir -p "$root/inputs"
fetch_verified inputs/ivan.png d0950f3677c2ff62db554c9d7d1bce5efe73bdfd3eb9f2dacce0664641cf6ce2
fetch_verified inputs/ref.wav e052569e65cbaf0f97de4d8574a216eb588f39ae469ea0c989cdf7d37af1677a
if [[ $mode == bench ]]; then
    fetch_verified benchmark_kandinsky_warm.py 6d1d1acea7ea33a2f35b3c3bea76030db8f1db15feb643d80beca79b88038456
fi

export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(IFS=,; printf '%s' "${gpu_uuids[*]}")
"$python" - "${#gpu_ids[@]}" <<'PY'
import sys

import torch
from flash_attn_interface import flash_attn_func

expected = int(sys.argv[1])
if torch.cuda.device_count() != expected:
    raise SystemExit(f"Unexpected CUDA device count: {torch.cuda.device_count()} (expected {expected})")
for index in range(expected):
    if torch.cuda.get_device_capability(index) != (9, 0) or "H100" not in torch.cuda.get_device_name(index):
        raise SystemExit(f"CUDA device {index} is not an H100")
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, FA3 import OK, {expected} H100 visible")
PY

mkdir -p "$root/outputs"
out=$(mktemp -d "$root/outputs/$mode.XXXXXX")
printf 'Results: %s\n' "$out"
cd "$repo"

inputs=(--config "$config" --image "$root/inputs/ivan.png" --audio "$root/inputs/ref.wav")
if [[ $mode == smoke ]]; then
    export HF_HUB_OFFLINE=0
    "$python" -m kandinsky_s2v.cli generate "${inputs[@]}" \
        --attention-engine flash_attention_3 --chunks 1 --steps 8 --seed 42 \
        --out "$out/lite-1gpu.mp4" 2>&1 | tee "$out/run.log"
else
    export HF_HUB_OFFLINE=1
    "$UV_PROJECT_ENVIRONMENT/bin/torchrun" --standalone --nproc-per-node "${#gpu_ids[@]}" \
        "$root/benchmark_kandinsky_warm.py" "${inputs[@]}" \
        --attention-engine flash_attention_3 --out-dir "$out" --steps 8 4 --repeats 2 \
        2>&1 | tee "$out/run.log"
fi
