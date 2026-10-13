#!/bin/bash
# Check the EmbeddingGemma 2 release files against the hashes pinned in
# src/context_orchestrator/local_model.py and print the `gh release create`
# command that publishes them. It does not publish anything itself.
#
#   scripts/publish-eg2-model.sh DIR
#
# DIR holds eg2-mixed.onnx, eg2-mixed.onnx_data and tokenizer.model (built
# with the recipe in model_licenses/embeddinggemma-2/NOTICE). LICENSE and
# NOTICE are copied in from the package, and SHA256SUMS is written.
set -eu
dir="${1:-}"
if [ -z "$dir" ] || [ ! -d "$dir" ]; then
  echo "usage: $0 DIR" >&2
  exit 2
fi
here="$(cd "$(dirname "$0")/.." && pwd)"
pkg="$here/src/context_orchestrator"
lic="$pkg/model_licenses/embeddinggemma-2"
tag=$(sed -n 's/^RELEASE_TAG = "\(.*\)"$/\1/p' "$pkg/local_model.py")
[ -n "$tag" ] || { echo "RELEASE_TAG not found in local_model.py" >&2; exit 1; }

cp "$lic/LICENSE" "$lic/NOTICE" "$dir/"
fail=0
# "name": (size, "sha256"), lines in FILES
sed -n 's/^    "\([^"]*\)": (\([0-9]*\), "\([0-9a-f]*\)"),$/\1 \2 \3/p' "$pkg/local_model.py" > "$dir/.pins"
[ -s "$dir/.pins" ] || { echo "no FILES pins found in local_model.py" >&2; exit 1; }
while read -r name size sha; do
  f="$dir/$name"
  if [ ! -f "$f" ]; then echo "MISSING  $name" >&2; fail=1; continue; fi
  got_size=$(wc -c < "$f" | tr -d ' ')
  got_sha=$(shasum -a 256 "$f" | awk '{print $1}')
  if [ "$got_size" != "$size" ] || [ "$got_sha" != "$sha" ]; then
    echo "MISMATCH $name: $got_size $got_sha (pinned $size $sha)" >&2; fail=1
  else
    echo "ok       $name ($size bytes)"
  fi
done < "$dir/.pins"
rm -f "$dir/.pins"
[ "$fail" -eq 0 ] || exit 1
(cd "$dir" && shasum -a 256 eg2-mixed.onnx eg2-mixed.onnx_data tokenizer.model LICENSE NOTICE > SHA256SUMS)
echo
echo "All files match the pins. To publish (run it yourself):"
echo
cat <<CMD
gh release create $tag --repo contorch/context-orchestrator \\
  --title "EmbeddingGemma 2 int8 model files ($tag)" \\
  --notes-file "$lic/NOTICE" --latest=false \\
  "$dir/eg2-mixed.onnx" "$dir/eg2-mixed.onnx_data" "$dir/tokenizer.model" \\
  "$dir/LICENSE" "$dir/NOTICE" "$dir/SHA256SUMS"
CMD
