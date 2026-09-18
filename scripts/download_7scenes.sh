#!/bin/bash
# Download + extract 7Scenes scenes into usgloc/data/7scenes_source/.
# Usage: download_7scenes.sh [scene ...]   (default: all 7 scenes)
# Layout after extraction matches usgloc.datasets.SevenScenes expectations:
#   <root>/<scene>/{TrainSplit.txt,TestSplit.txt,seq-XX/frame-XXXXXX.{color.png,depth.png,pose.txt}}
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)/data/7scenes_source"
BASE="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"
SCENES=("$@")
[ ${#SCENES[@]} -eq 0 ] && SCENES=(chess fire heads office pumpkin redkitchen stairs)

# The 7Scenes zips contain STORED entries with inconsistent size fields, which
# makes unzip exit 1 (warning) despite extracting correctly -- tolerate that.
unzip_tolerant() {
  unzip -q -o "$@" || { rc=$?; [ "$rc" -le 1 ] || return "$rc"; }
}

mkdir -p "$ROOT"
cd "$ROOT"
for ds in "${SCENES[@]}"; do
  if [ -f "$ds/TrainSplit.txt" ] && ls "$ds"/seq-*/ >/dev/null 2>&1 \
     && ! ls "$ds"/*.zip >/dev/null 2>&1; then
    echo "=== $ds already extracted, skipping"
    continue
  fi
  if [ ! -d "$ds" ]; then
    echo "=== Downloading $ds"
    wget -c -q --show-progress "$BASE/$ds.zip"
    unzip_tolerant "$ds.zip"
    rm "$ds.zip"
  fi
  # each scene zip contains per-sequence zips
  for z in "$ds"/*.zip; do
    [ -e "$z" ] || continue
    unzip_tolerant "$z" -d "$ds"
    rm "$z"
  done
  echo "=== $ds done: $(du -sh "$ds" | cut -f1)"
done
echo "ALL REQUESTED SCENES DONE"
