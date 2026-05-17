#!/usr/bin/env bash
# Render scripts/sec_iv_workflow.mmd to PNG via mermaid.ink and drop it
# into the paper's images/ directory.  Requires curl + python3.
#
# mermaid.ink's pako endpoint expects a JSON envelope
# ``{"code": ..., "mermaid": {...}}`` that has been zlib-compressed and
# base64-urlsafe-encoded (without padding).
#
# Usage: scripts/render_sec_iv_workflow.sh [<out.png>]
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
src="$here/sec_iv_workflow.mmd"
out="${1:-$here/../../papers/conferences/ICARM/_IEEE_ARM__Generative_3D_Object_Modeling_for_Robust_Robot_Manipulation_in_ROS_2/images/sec_iv_workflow.png}"

[[ -f "$src" ]] || { echo "missing $src" >&2; exit 1; }

payload=$(python3 - <<PY
import zlib, base64, json
src = open("$src").read()
data = {"code": src, "mermaid": {"theme": "default"}}
js = json.dumps(data).encode()
print(base64.urlsafe_b64encode(zlib.compress(js, 9)).decode().rstrip("="))
PY
)
url="https://mermaid.ink/img/pako:${payload}?type=png"

mkdir -p "$(dirname "$out")"
curl --silent --show-error --fail --location -o "$out" "$url"
echo "wrote $out ($(stat -c%s "$out") bytes)"
