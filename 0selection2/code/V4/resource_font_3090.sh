#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
export CUDA_VISIBLE_DEVICES=''

container=b8ced5d8a8cdafdc5006f827a0d36c319c61056a65316af4ff9937453fa0635c
source_font=/public/home/shiys/.cache/selector_fonts/20261026/NotoSansCJK-Regular.ttc
target_font=/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc
expected=5dcd1c336cc9344cb77c03a0cd8982ca8a7dc97d620fd6c9c434e02dcb1ceeb3

test "$(id -u)" = 0
test "$(hostname -s)" = "${container:0:12}"
test "${SLURM_JOB_ID:-}" = 1465
grep -Fq "/docker/$container" /proc/self/cgroup
test "$(sha256sum "$source_font" | cut -d ' ' -f 1)" = "$expected"
if test -e "$target_font"; then
    test "$(sha256sum "$target_font" | cut -d ' ' -f 1)" = "$expected"
    action=reused_identical_existing_font
else
    install -D -m 0644 "$source_font" "$target_font"
    action=installed_missing_report_font
fi
test "$(sha256sum "$target_font" | cut -d ' ' -f 1)" = "$expected"
printf '{"complete":true,"gpu_compute":false,"action":"%s","container":"%s","font_sha256":"%s","font_path":"%s"}\n' "$action" "$container" "$expected" "$target_font"
