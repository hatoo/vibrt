# Regenerate test scenes (and optionally render previews) from their
# procedural generators.
#
# Usage:
#   make                      # regenerate every test_scenes/*/scene.json
#   make veach_mis            # regenerate one scene by directory name
#   make previews             # regenerate scenes and render preview.png for each
#   make veach_mis-preview    # render preview for one scene
#   make addon                # rebuild blender/vibrt_blender.zip
#   make clean                # remove generated scene.json, scene.bin, preview.png
#
# Overridable:
#   PYTHON  (default: py)
#   VIBRT   (default: ./target/release/vibrt.exe)
#   SPP     (default: 128)

PYTHON ?= py
VIBRT  ?= ./target/release/vibrt.exe
SPP    ?= 128

SCENE_SCRIPTS := $(wildcard test_scenes/*/make_scene.py)
SCENES        := $(patsubst test_scenes/%/make_scene.py,%,$(SCENE_SCRIPTS))
SCENE_JSONS   := $(SCENE_SCRIPTS:make_scene.py=scene.json)
SCENE_BINS    := $(SCENE_SCRIPTS:make_scene.py=scene.bin)
PREVIEW_PNGS  := $(SCENE_SCRIPTS:make_scene.py=preview.png)

PREVIEW_TARGETS := $(addsuffix -preview,$(SCENES))

ADDON_ZIP     := blender/vibrt_blender.zip
ADDON_SOURCES := $(wildcard blender/vibrt_blender/*.py)

.PHONY: all scenes previews addon clean $(SCENES) $(PREVIEW_TARGETS)

all: scenes

scenes: $(SCENE_JSONS)
previews: $(PREVIEW_PNGS)
addon: $(ADDON_ZIP)

# Shorthand: `make <scene>` regenerates scene.json;
#            `make <scene>-preview` renders preview.png.
$(SCENES): %: test_scenes/%/scene.json
$(PREVIEW_TARGETS): %-preview: test_scenes/%/preview.png

# Running make_scene.py writes scene.json and scene.bin side-by-side.
test_scenes/%/scene.json: test_scenes/%/make_scene.py
	cd $(dir $<) && $(PYTHON) make_scene.py

test_scenes/%/preview.png: test_scenes/%/scene.json
	$(VIBRT) $< --spp $(SPP) --output $@

$(ADDON_ZIP): $(ADDON_SOURCES) blender/build_addon.py
	$(PYTHON) blender/build_addon.py

clean:
	rm -f $(SCENE_JSONS) $(SCENE_BINS) $(PREVIEW_PNGS) $(ADDON_ZIP)
