#!/bin/bash
#
# Runs the samples in offscreen mode and stores the rendered frames as ppm files
#
# Configuration is passed via environment variables:
#   OUT=build/ppm    Directory that images and logs are written to
#   FRAMES=1         Number of frames to render per sample
#   ORBIT=0          Set to 1 to rotate the camera around the scene while rendering
#   WIDTH=/HEIGHT=   Render resolution (defaults to the size the samples use)
#   VALIDATION=1     Set to 0 to run without validation layers
#   TIMEOUT=180      Seconds after which a sample is considered stuck
#   SAMPLES="a b"    Only run the given samples (defaults to all built samples)
#
# Examples:
#   ./run_offscreen.sh                                  One image per sample
#   FRAMES=36 ORBIT=1 OUT=build/ppm_orbit ./run_offscreen.sh    A turntable per sample
#   SAMPLES="triangle gears" FRAMES=60 ./run_offscreen.sh       Only two samples
#
# Samples that render more than one frame store one file per frame, so they get a
# directory of their own. renderheadless and computeheadless don't use the offscreen
# options, they always render without a window.

set -u
cd "$(dirname "$0")"

BIN=build/bin
OUT=${OUT:-build/ppm}
FRAMES=${FRAMES:-1}
ORBIT=${ORBIT:-0}
VALIDATION=${VALIDATION:-1}
TIMEOUT=${TIMEOUT:-180}
SAMPLES=${SAMPLES:-}

if [ ! -d "$BIN" ]; then
	echo "No samples found in $BIN, build them first"
	exit 1
fi
[ -n "$SAMPLES" ] || SAMPLES=$(ls "$BIN")

LOGS=$OUT/_logs
SUMMARY=$OUT/_summary.txt
mkdir -p "$LOGS"
: > "$SUMMARY"

args=(--offscreen --offscreenframes "$FRAMES")
[ "$ORBIT" = "1" ] && args+=(--offscreenorbit)
[ "$VALIDATION" = "1" ] && args+=(-v)
[ -n "${WIDTH:-}" ] && args+=(-w "$WIDTH")
[ -n "${HEIGHT:-}" ] && args+=(-h "$HEIGHT")

echo "Running ${FRAMES} frame(s) per sample into ${OUT} (orbit=${ORBIT}, validation=${VALIDATION})"

total=0
withimages=0
witherrors=0
for name in $SAMPLES; do
	bin=$BIN/$name
	[ -x "$bin" ] || continue
	total=$((total + 1))

	if [ "$FRAMES" -gt 1 ]; then
		mkdir -p "$OUT/$name"
		target=$OUT/$name/$name.ppm
	else
		target=$OUT/$name.ppm
	fi

	# Note: stdin is closed, as some samples wait for a key press before they exit
	timeout "$TIMEOUT" "$bin" "${args[@]}" --offscreenfilename "$target" < /dev/null > "$LOGS/$name.log" 2>&1
	code=$?

	errors=$(grep -c "ERROR:" "$LOGS/$name.log")
	if [ "$FRAMES" -gt 1 ]; then
		images=$(ls "$OUT/$name"/*.ppm 2>/dev/null | wc -l)
	else
		images=$([ -f "$target" ] && echo 1 || echo 0)
	fi
	[ "$images" -gt 0 ] && withimages=$((withimages + 1))
	[ "$errors" -gt 0 ] && witherrors=$((witherrors + 1))

	printf "%-34s exit=%-4s errors=%-4s images=%s\n" "$name" "$code" "$errors" "$images" | tee -a "$SUMMARY"
done

# renderheadless stores its image in the working directory, keep it with the other images
[ -f headless.ppm ] && mv headless.ppm "$OUT/renderheadless.ppm"
# Files that samples write to the working directory and that aren't part of the output
rm -f imgui.ini shaders/glsl/shaderobjects/phong.frag.bin shaders/glsl/shaderobjects/phong.vert.bin

echo
echo "$total samples run, $withimages produced images, $witherrors reported validation errors"
echo "Images in $OUT, per sample logs in $LOGS, summary in $SUMMARY"
