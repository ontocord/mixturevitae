#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=scan_decontam
#SBATCH --account=AIFAC_L01_028
#SBATCH -p boost_usr_prod
#SBATCH --threads-per-core=1
#SBATCH --time=23:59:00
#SBATCH --output=/leonardo_work/AIFAC_L01_028/hraj0000/mixturevitae/decontaminate/slurm-out/%j-precompress.out

# ==== CONFIG (edit if needed) ====
CLEAN_ROOT="/leonardo_work/AIFAC_L01_028/hraj0000/mixturevitae/cleaned_dataset/mixtuevitae211bt-clean"
# How many files to compress concurrently (each uses pigz -p1 thread)
PARALLEL_FILES="${PARALLEL_FILES:-$SLURM_CPUS_PER_TASK}"
# pigz compression level (1=fastest, 9=smallest)
PIGZ_LEVEL="${PIGZ_LEVEL:-9}"
# Keep original .jsonl beside .jsonl.gz
KEEP_ORIGINAL=1
# Where to log the list of produced .gz files (and their source)
MANIFEST="${MANIFEST:-$CLEAN_ROOT/.compression_manifest.tsv}"
# ================================

set -euo pipefail

echo "[$(date -Is)] Node: $(hostname)"
echo "[$(date -Is)] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "[$(date -Is)] CLEAN_ROOT=${CLEAN_ROOT}"
echo "[$(date -Is)] PARALLEL_FILES=${PARALLEL_FILES}"
echo "[$(date -Is)] PIGZ_LEVEL=${PIGZ_LEVEL}"

# --- Modules / binaries ---
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load pigz || true
fi

if ! command -v pigz >/dev/null 2>&1; then
  echo "ERROR: pigz not found in PATH. Try 'module load pigz' or ask admins." >&2
  exit 1
fi

# --- Sanity checks ---
if [[ ! -d "$CLEAN_ROOT" ]]; then
  echo "ERROR: CLEAN_ROOT does not exist: $CLEAN_ROOT" >&2
  exit 1
fi

mkdir -p "$(dirname "$MANIFEST")"
: > "$MANIFEST"  # truncate manifest

# Find all .jsonl files that DO NOT yet have a corresponding .jsonl.gz
echo "[$(date -Is)] Scanning for uncompressed .jsonl files…"
mapfile -d '' TO_COMPRESS < <(
  find "$CLEAN_ROOT" -type f -name '*.jsonl' -print0 \
  | while IFS= read -r -d '' f; do
      if [[ ! -f "${f}.gz" ]]; then
        printf '%s\0' "$f"
      fi
    done
)

NUM_TOTAL=$(find "$CLEAN_ROOT" -type f -name '*.jsonl' | wc -l || true)
NUM_TODO=${#TO_COMPRESS[@]}
echo "[$(date -Is)] Total .jsonl found: ${NUM_TOTAL}"
echo "[$(date -Is)] To compress now:   ${NUM_TODO}"

# Helper: compress one file using pigz with 1 thread (RAM-friendly)
compress_one() {
  local src="$1"
  local lvl="$2"
  local keep="$3"

  # -1..-9 compression, -p1 => one thread per file, -k keep input if requested
  local keep_flag=
  [[ "$keep" == "1" ]] && keep_flag="-k"

  pigz -f -p 1 -${lvl} ${keep_flag} -- "$src"
}

export -f compress_one

# Parallel compression (xargs); each pigz uses 1 CPU thread to keep memory low.
# We run up to $PARALLEL_FILES files at a time.
if [[ "$NUM_TODO" -gt 0 ]]; then
  printf '%s\0' "${TO_COMPRESS[@]}" \
  | xargs -0 -n 1 -P "${PARALLEL_FILES}" -I{} bash -c 'compress_one "$@"' _ {} "${PIGZ_LEVEL}" "${KEEP_ORIGINAL}"
else
  echo "[$(date -Is)] Nothing to compress."
fi

# Verify archives we produced in this run (or all .gz, cheap enough)
echo "[$(date -Is)] Verifying .gz integrity with pigz -t …"
# Gather all .jsonl.gz (including previously existing)
mapfile -d '' ALL_GZ < <(find "$CLEAN_ROOT" -type f -name '*.jsonl.gz' -print0)
NUM_GZ=${#ALL_GZ[@]}
echo "[$(date -Is)] Total .jsonl.gz present: ${NUM_GZ}"

verify_one() {
  local gz="$1"
  # pigz -t returns 0 on success; print path for any failure
  if ! pigz -t -- "$gz"; then
    echo "VERIFICATION_FAILED: $gz" >&2
    return 1
  fi
}
export -f verify_one

if [[ "$NUM_GZ" -gt 0 ]]; then
  printf '%s\0' "${ALL_GZ[@]}" \
  | xargs -0 -n 1 -P "${PARALLEL_FILES}" -I{} bash -c 'verify_one "$@"' _ {}
fi

# Write manifest: <relative_path_in_tree>\t<abs_path_to_gz>\t<size_bytes>
echo -e "#dest_rel\tabs_path\tsize_bytes" >> "$MANIFEST"
while IFS= read -r -d '' gz; do
  rel="${gz#$CLEAN_ROOT/}"
  size=$(stat -c%s "$gz")
  echo -e "$rel\t$gz\t$size" >> "$MANIFEST"
done < <(printf '%s\0' "${ALL_GZ[@]}" | sort -z)

echo "[$(date -Is)] Wrote manifest to: $MANIFEST"
echo "[$(date -Is)] Done."
