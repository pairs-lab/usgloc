#!/bin/bash
# Download + extract 12Scenes (Valentin et al., 3DV'16) into
# usgloc/data/12scenes_source/. Layout per scene: <building>/<scene>/data/
# frame-XXXXXX.{color.jpg,depth.png,pose.txt} + split.txt + info.txt.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)/data/12scenes_source"
BASE="http://graphics.stanford.edu/projects/reloc/data"
mkdir -p "$ROOT"
cd "$ROOT"
for b in apt1 apt2 office1 office2; do
  if [ -d "$b" ] && [ -z "$(find "$b" -maxdepth 0 -empty)" ]; then
    echo "=== $b already present, skipping"
    continue
  fi
  echo "=== Downloading $b"
  wget -c -q --show-progress "$BASE/$b.zip"
  unzip -q -o "$b.zip"
  rm "$b.zip"
  echo "=== $b done: $(du -sh "$b" | cut -f1)"
done
echo "ALL 12SCENES BUILDINGS DONE"
