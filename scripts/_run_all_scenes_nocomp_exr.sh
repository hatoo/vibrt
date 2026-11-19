#!/bin/bash
# Run EXR-based engine accuracy comparison across all 7 test scenes.
# All scenes rendered with compositor=OFF, view=Raw, identical SPP/PCT.

set -e
BLENDER="C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
SPP=${SPP:-64}
PCT=${PCT:-25}

declare -A SCENES=(
  [bmw27]="test_scenes/bmw27/bmw27/bmw27_gpu.blend"
  [classroom]="test_scenes/classroom/classroom/classroom.blend"
  [flat_archiviz]="test_scenes/flat_archiviz/flat_archiviz/flat-archiviz.blend"
  [ies_light]="test_scenes/ies_light/ies_light/ies_light.blend"
  [junk_shop]="test_scenes/junk_shop/junk_shop/junk_shop.blend"
  [lone_monk]="test_scenes/lone_monk/lone_monk/lone_monk.blend"
  [pabellon]="test_scenes/pabellon/3d/pavillon_barcelone_v1.2.blend"
)

for tag in bmw27 classroom flat_archiviz ies_light junk_shop lone_monk pabellon; do
  blend="${SCENES[$tag]}"
  echo ""
  echo "########  rendering: $tag  ########"
  SPP=$SPP PCT=$PCT "$BLENDER" -b "$blend" \
    --python scripts/_render_scene_nocomp_exr.py -- \
    --tag "$tag" --spp "$SPP" --pct "$PCT" 2>&1 | grep -E "^RESULT|^=== |error" || true
done
