#!/usr/bin/env bash

set -Eeuo pipefail

readonly REPO="PoopBear/RIFT"
readonly REPO_REV="4d171611ae164ff575000395159119b97ef5ad5c"
readonly STATS_REL="galaxea_indoor_cleaning_3cam224/dataset_stats.json"
readonly STATS_SHA256="52ab72e1185d0fd5c15b7e29187b9606f7e610f5d7d386e4c59ba217442673b2"
readonly HEADROOM_BYTES=$((5 * 1024 * 1024 * 1024))
readonly RETRIES="${DOWNLOAD_RETRIES:-3}"

usage() {
  cat <<'EOF'
Download the minimal assets needed for Galaxea real-robot inference.

Usage:
  GALAXEA_STATS_SOURCE=/path/to/dataset_stats.json \
    download_galaxea_assets.sh [ASSET_ROOT]

Defaults:
  ASSET_ROOT=$GALAXEA_ASSET_ROOT, or ./galaxea-real-assets
  HF_ENDPOINT=https://hf-mirror.com

Required input:
  GALAXEA_STATS_SOURCE must point to the 445-episode Galaxea
  dataset_stats.json unless a verified copy already exists under ASSET_ROOT.

The script downloads only:
  - FastWAM, FastWAM-Joint, and RIFT Galaxea checkpoints
  - Wan2.2 VAE
  - UMT5-XXL bf16 encoder and tokenizer

It intentionally does not download the Wan2.2 DiT shards or ActionDiT.
The inference server must set skip_dit_load_from_pretrain=true and
action_dit_pretrained_path=null before loading the complete trained checkpoint.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

retry() {
  local attempt=1
  local delay=5

  while ! "$@"; do
    if (( attempt >= RETRIES )); then
      return 1
    fi
    printf 'Download failed (attempt %d/%d); retrying in %ds...\n' \
      "$attempt" "$RETRIES" "$delay" >&2
    sleep "$delay"
    ((attempt += 1))
    ((delay *= 2))
  done
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
(( $# <= 1 )) || { usage >&2; exit 2; }

for command_name in awk df flock mkdir sha256sum stat; do
  command -v "$command_name" >/dev/null 2>&1 || die "missing command: $command_name"
done

if command -v hf >/dev/null 2>&1; then
  HF_CLI=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=(huggingface-cli)
else
  die "missing HF CLI; install huggingface_hub in the inference environment"
fi

ROOT="${1:-${GALAXEA_ASSET_ROOT:-$PWD/galaxea-real-assets}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$ROOT"
exec 9>"$ROOT/.download.lock"
flock -n 9 || die "another asset download is using $ROOT"

readonly REPO_DIR="$ROOT/$REPO"
readonly ASSET_DIR="$REPO_DIR/assets"
readonly STATS_DST="$ROOT/$STATS_REL"

CKPT_FILES=(
  "fastwam/galaxea_indoor_cleaning_3cam224_10ep/fastwam_step014830.pt"
  "fastwam_joint/galaxea_indoor_cleaning_3cam224_10ep/fastwam_joint_step014830.pt"
  "rift/galaxea_indoor_cleaning_3cam224_10ep/rift_step014830.pt"
)

ASSET_FILES=(
  "assets/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
  "assets/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
  "assets/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json"
  "assets/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model"
  "assets/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json"
  "assets/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json"
)

REL_PATHS=(
  "PoopBear/RIFT/fastwam/galaxea_indoor_cleaning_3cam224_10ep/fastwam_step014830.pt"
  "PoopBear/RIFT/fastwam_joint/galaxea_indoor_cleaning_3cam224_10ep/fastwam_joint_step014830.pt"
  "PoopBear/RIFT/rift/galaxea_indoor_cleaning_3cam224_10ep/rift_step014830.pt"
  "PoopBear/RIFT/assets/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
  "PoopBear/RIFT/assets/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
  "PoopBear/RIFT/assets/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json"
  "PoopBear/RIFT/assets/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model"
  "PoopBear/RIFT/assets/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json"
  "PoopBear/RIFT/assets/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json"
  "$STATS_REL"
)

EXPECTED_SIZES=(
  12041813369
  12041813369
  12061290937
  1409401152
  11361845432
  6623
  4548313
  16837417
  61728
  105188
)

EXPECTED_SHA256=(
  "9e324cad3dbb13d623e4aed51b8534226a13079f1d9e3f3fb59f1f67ec0ec19f"
  "7616e7861c240360d6fab1ccca75d54b4857734264210c6a1c1a2f017cbcd869"
  "052c0713bfd1f04222599deea43f02cebff963e7453ec7f2e53f81863d8d3939"
  "0e913a2ca571c75fcb63385a8edadcca73454af5842590996cb1ad11e4142590996"
  "d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23"
  "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d"
  "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"
  "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"
  "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05"
  "$STATS_SHA256"
)

if [[ -e "$STATS_DST" ]]; then
  [[ -f "$STATS_DST" ]] || die "stats destination is not a file: $STATS_DST"
  current_stats_sha="$(sha256sum "$STATS_DST" | awk '{print $1}')"
  [[ "$current_stats_sha" == "$STATS_SHA256" ]] || \
    die "wrong Galaxea stats at $STATS_DST (refusing to overwrite it)"
else
  STATS_SOURCE="${GALAXEA_STATS_SOURCE:-}"
  [[ -n "$STATS_SOURCE" ]] || die \
    "set GALAXEA_STATS_SOURCE to the 445-episode Galaxea dataset_stats.json"
  [[ -f "$STATS_SOURCE" ]] || die "stats source is not a local file: $STATS_SOURCE"
  source_stats_sha="$(sha256sum "$STATS_SOURCE" | awk '{print $1}')"
  [[ "$source_stats_sha" == "$STATS_SHA256" ]] || \
    die "GALAXEA_STATS_SOURCE has the wrong SHA256: $source_stats_sha"
  mkdir -p "$(dirname "$STATS_DST")"
  cp -- "$STATS_SOURCE" "$STATS_DST"
fi

missing_bytes=0
for i in "${!REL_PATHS[@]}"; do
  path="$ROOT/${REL_PATHS[$i]}"
  expected_size="${EXPECTED_SIZES[$i]}"
  if [[ ! -f "$path" ]] || [[ "$(stat -c %s "$path" 2>/dev/null || printf 0)" != "$expected_size" ]]; then
    ((missing_bytes += expected_size))
  fi
done

available_bytes="$(df -PB1 "$ROOT" | awk 'NR == 2 {print $4}')"
required_bytes=$((missing_bytes + HEADROOM_BYTES))
(( available_bytes >= required_bytes )) || die \
  "insufficient disk: need $required_bytes bytes including headroom; have $available_bytes"

printf 'Asset root: %s\n' "$ROOT"
printf 'HF endpoint: %s\n' "$HF_ENDPOINT"
printf 'Missing payload: %d bytes\n' "$missing_bytes"

mkdir -p "$REPO_DIR"

retry "${HF_CLI[@]}" download "$REPO" \
  "${CKPT_FILES[@]}" "${ASSET_FILES[@]}" \
  --revision "$REPO_REV" --local-dir "$REPO_DIR"

for i in "${!REL_PATHS[@]}"; do
  path="$ROOT/${REL_PATHS[$i]}"
  [[ -f "$path" ]] || die "missing downloaded asset: $path"
  actual_size="$(stat -c %s "$path")"
  [[ "$actual_size" == "${EXPECTED_SIZES[$i]}" ]] || die \
    "wrong size for ${REL_PATHS[$i]}: expected ${EXPECTED_SIZES[$i]}, got $actual_size"
done

manifest="$ROOT/SHA256SUMS"
: >"$manifest"
for i in "${!REL_PATHS[@]}"; do
  printf '%s  %s\n' "${EXPECTED_SHA256[$i]}" "${REL_PATHS[$i]}" >>"$manifest"
done

(
  cd "$ROOT"
  sha256sum -c SHA256SUMS
)

cat <<EOF

Assets are ready.

export GALAXEA_ASSET_ROOT='$ROOT'
export DIFFSYNTH_MODEL_BASE_PATH='$ASSET_DIR'
export GALAXEA_NORM_STATS='$STATS_DST'

FastWAM checkpoint:
  $REPO_DIR/${CKPT_FILES[0]}
FastWAM-Joint checkpoint:
  $REPO_DIR/${CKPT_FILES[1]}
RIFT checkpoint:
  $REPO_DIR/${CKPT_FILES[2]}
EOF
