#!/usr/bin/env bash
# Generate poster frames for each video at t=1s.
# Run from inside docs/.
set -e
mkdir -p static/images
for v in videos/*.mp4; do
    [ -f "$v" ] || continue
    base=$(basename "$v" .mp4)
    out="static/images/poster_${base}.png"
    if [ ! -f "$out" ]; then
        ffmpeg -loglevel error -ss 00:00:01 -i "$v" -frames:v 1 "$out"
        echo "wrote $out"
    fi
done
