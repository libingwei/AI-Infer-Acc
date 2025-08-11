#!/usr/bin/env bash
set -euo pipefail

# Fix TensorRT/cuDNN symlinks in a given lib directory by creating:
#  - libXXX.so.MAJOR -> libXXX.so.MAJOR.minor.patch (best available)
#  - libXXX.so       -> libXXX.so.MAJOR.minor.patch (best available)
# Then print an LD_LIBRARY_PATH export hint.

usage() {
  echo "Usage: $0 <tensorrt-lib-dir>" >&2
  echo "  Example: $0 /kaggle/input/TensorRT-10.4.0.26/targets/x86_64-linux-gnu/lib" >&2
}

if [[ $# -ne 1 ]]; then
  usage; exit 1
fi

LIBDIR="$1"
if [[ ! -d "$LIBDIR" ]]; then
  echo "ERROR: Directory not found: $LIBDIR" >&2
  exit 2
fi

is_real() {
  local f="$1"
  [[ -e "$f" ]] || return 1
  local real
  real=$(readlink -f -- "$f" || true)
  [[ -e "$real" ]] || return 1
  # Filter out tiny placeholder files (<1MB)
  local sz
  sz=$(stat -c %s -- "$real" 2>/dev/null || echo 0)
  [[ "$sz" -gt 1000000 ]]
}

echo "Fixing symlinks in: $LIBDIR"

# Collect unique roots (libXXX) from versioned files
shopt -s nullglob
declare -A ROOTS=()
for f in "$LIBDIR"/lib*.so.[0-9]*; do
  # only consider real candidates
  if ! is_real "$f"; then continue; fi
  bn=$(basename -- "$f")
  # libname.so.10.4.0 -> root=libname
  root=${bn%%.so.*}
  ROOTS["$root"]=1
done

fix_one_root() {
  local root="$1"
  local pattern="$LIBDIR/${root}.so.*"
  local cand=()
  # shellcheck disable=SC2206
  cand=($pattern)
  # keep real ones only
  local real=()
  for c in "${cand[@]}"; do
    if is_real "$c"; then real+=("$c"); fi
  done
  [[ ${#real[@]} -gt 0 ]] || return 0
  # sort by version (natural sort)
  IFS=$'\n' read -r -d '' -a real_sorted < <(printf '%s\n' "${real[@]}" | sort -V && printf '\0')
  local best
  best="${real_sorted[-1]}"
  local best_bn
  best_bn=$(basename -- "$best")

  # parse major from best
  local maj
  maj=$(sed -n 's/^.*\.so\.\([0-9]\+\).*$/\1/p' <<<"$best_bn" || true)

  local base="$LIBDIR/${root}.so"
  local soname="$LIBDIR/${root}.so.$maj"

  # Create/update symlinks to best
  ln -sfn -- "$best_bn" "$base"
  if [[ -n "$maj" ]]; then
    ln -sfn -- "$best_bn" "$soname"
  fi
  printf '  %-30s -> %s\n' "$(basename -- "$base")" "$best_bn"
  if [[ -n "$maj" ]]; then
    printf '  %-30s -> %s\n' "$(basename -- "$soname")" "$best_bn"
  fi
}

for k in "${!ROOTS[@]}"; do
  fix_one_root "$k"
done

echo
echo "Done. To use these fixed libs in this session, run:"
echo "  export LD_LIBRARY_PATH=\"$LIBDIR:\$LD_LIBRARY_PATH\""
