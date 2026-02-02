# Gradual Intra Refresh (GIR) Implementation Plan

## Overview
Implement manual Gradual Intra Refresh for H.264 video encoding to achieve IDR-like error recovery without bitrate spikes by progressively refreshing the frame in horizontal slice bands.

## Target File
- `screenshot.cpp` - all modifications here
- FFmpeg files (`vulkan_encode_h264.c`, `vulkan_encode.c`, `vulkan_video.c`) are **reference only**

---

## Phase 1: Extension Detection and Capability Checking

### 1.1 Add GIR configuration to EncoderConfig (~line 890)
```cpp
struct EncoderConfig {
    // ... existing fields ...

    // GIR Configuration
    bool enableGIR = false;           // Enable Gradual Intra Refresh
    uint32_t girSliceCount = 4;       // Number of horizontal slice bands
    uint32_t girRefreshPeriod = 4;    // Frames to complete full refresh
};
```

### 1.2 Add capability/state tracking (~line 960)
```cpp
// GIR capability detection
bool intraRefreshExtensionSupported = false;
bool differentSliceTypeSupported = false;
bool constrainedIntraPredSupported = false;

// GIR state
uint32_t girCurrentSlice = 0;
uint64_t girCycleStartFrame = 0;
```

### 1.3 Extension detection in queryCapabilities() (~line 1166)
Check for:
- `VK_KHR_video_encode_intra_refresh` extension
- `VK_VIDEO_ENCODE_H264_CAPABILITY_DIFFERENT_SLICE_TYPE_BIT_KHR` flag
- `VK_VIDEO_ENCODE_H264_STD_CONSTRAINED_INTRA_PRED_FLAG_SET_BIT_KHR` flag
- Verify `maxSliceCount >= girSliceCount`

---

## Phase 2: Multi-Slice Frame Structure

### 2.1 Slice band calculation
```cpp
uint32_t mbWidth = (config.width + 15) / 16;
uint32_t mbHeight = (config.height + 15) / 16;
uint32_t mbRowsPerSlice = (mbHeight + sliceCount - 1) / sliceCount;
```

### 2.2 Slice data structures (~line 2232)
Replace single slice with array:
```cpp
std::vector<StdVideoEncodeH264SliceHeader> sliceHeaders(sliceCount);
std::vector<VkVideoEncodeH264NaluSliceInfoKHR> sliceInfos(sliceCount);
```

### 2.3 Per-slice configuration
For each slice `i`:
- `first_mb_in_slice = i * mbRowsPerSlice * mbWidth`
- `slice_type = (i == girCurrentSlice) ? I : P`
- Different QP values possible for intra vs inter slices

### 2.4 Update VkVideoEncodeH264PictureInfoKHR (~line 2328)
```cpp
h264PicInfo.naluSliceEntryCount = config.enableGIR ? config.girSliceCount : 1;
h264PicInfo.pNaluSliceEntries = sliceInfos.data();
```

---

## Phase 3: GIR Cycling Logic

### 3.1 updateGIRState() function
```cpp
void updateGIRState() {
    if (!config.enableGIR) return;

    uint32_t frameInCycle = decodingOrderFrameNum % config.girRefreshPeriod;
    girCurrentSlice = frameInCycle % config.girSliceCount;
}
```

### 3.2 Integration with IDR frames
- IDR frames reset the GIR cycle
- After full GIR cycle, error recovery equivalent to IDR achieved

---

## Phase 4: Reference Constraint Approach

### Critical Limitation
Without `VK_KHR_video_encode_intra_refresh`, **Vulkan cannot constrain motion vector search regions per-slice**. This means:
- P-slices can reference any part of the reference frame
- Error propagation may still occur from dirty to clean regions

### Mitigation Strategies (Partial Solutions)

#### 4.1 Enable Constrained Intra Prediction
In PPS creation (~line 1291):
```cpp
pps.flags.constrained_intra_pred_flag = config.enableGIR ? 1 : 0;
```
This prevents intra blocks from using inter-predicted neighbors, limiting error propagation through intra prediction (but not inter).

#### 4.2 Slice Boundary Deblocking Control
Option to disable deblocking at slice boundaries:
```cpp
sliceHeader.disable_deblocking_filter_idc =
    STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_ENABLED_EXCEPT_SLICE_BOUNDARY;
```

#### 4.3 Potential Future: Dual Reference Chain
Track "clean" vs "dirty" regions per reference frame and select references accordingly. Complex to implement and may not be fully effective without hardware support.

---

## Implementation Checklist

1. [ ] Add GIR fields to `EncoderConfig`
2. [ ] Add GIR state variables to class
3. [ ] Implement extension/capability detection
4. [ ] Add validation for GIR configuration
5. [ ] Create multi-slice data structures
6. [ ] Implement `updateGIRState()` function
7. [ ] Modify `recordEncodeCommands()` for multi-slice
8. [ ] Enable constrained intra prediction when GIR active
9. [ ] Add logging for GIR cycle state
10. [ ] Test with various slice counts (2, 4, 8)

---

## Testing and Verification

1. **Extension detection**: Verify intra refresh extension detection logs
2. **Multi-slice output**: Use `ffprobe -show_frames` to verify slice count per frame
3. **Slice types**: Verify mixed I/P slices appear in same frame
4. **Visual quality**: Test with deliberate error injection to verify recovery
5. **Bitrate**: Compare bitrate distribution vs full IDR frames

---

## Notes

- The `codedOffset`/`codedExtent` fields in `VkVideoPictureResourceInfoKHR` are for cropping, **not** for constraining motion estimation regions
- Reference list manipulation (RefPicList0/1) controls *which* reference frames are used, not *which regions* within them
- True GIR with proper reference constraints requires driver/hardware support via `VK_KHR_video_encode_intra_refresh`

---

## Capability Test Results (2025-02-02)

**Status: IMPLEMENTED - Hardware GIR supported with NVIDIA driver 590+**

```
GIR (Gradual Intra Refresh) Capability Check:
    VK_KHR_video_encode_intra_refresh extension: SUPPORTED
    DIFFERENT_SLICE_TYPE capability: NOT SUPPORTED
    CONSTRAINED_INTRA_PRED std syntax: NOT SUPPORTED
    -> Hardware GIR supported via extension
```

### What These Mean

| Capability | Purpose | Status |
|------------|---------|--------|
| `VK_KHR_video_encode_intra_refresh` | Hardware-accelerated GIR with automatic reference constraints | Not supported |
| `DIFFERENT_SLICE_TYPE` | Allow I-slices and P-slices in the same frame | **Not supported** |
| `CONSTRAINED_INTRA_PRED` | Prevent intra blocks from referencing inter-predicted neighbors | Not supported |

**Critical Blocker**: Without `DIFFERENT_SLICE_TYPE`, every slice in a frame must be the same type. True GIR requires mixing I-slices (for refresh) and P-slices (for compression) in the same frame.

---

## Alternatives for Error Recovery

Since GIR is not possible, these are the available options:

| Approach | Description | Bitrate Impact | Recovery Latency |
|----------|-------------|----------------|------------------|
| **Periodic IDR** | Current implementation | Spike every GOP | GOP frames |
| **Smaller GOP** | Reduce gopSize (e.g., 30→15) | ~10-20% higher avg | Faster recovery |
| **All-Intra** | gopSize=1 | 3-5x higher | Instant |
| **Adaptive IDR** | Request IDR on packet loss | Spike on demand | On-demand |

### Recommendation

For streaming with packet loss concerns:
1. Use **Adaptive IDR** - request keyframe when decoder reports errors (already implemented via `requestKeyframe()`)
2. Keep moderate GOP size (e.g., 60 frames / 2 seconds)
3. Implement feedback channel for keyframe requests

---

## Code Changes Made

Added GIR capability detection in `screenshot.cpp`:

### Member Variables (~line 1037)
```cpp
bool girIntraRefreshExtensionSupported = false;
bool girDifferentSliceTypeSupported = false;
bool girConstrainedIntraPredSupported = false;
```

### Capability Query (in queryCapabilities())
```cpp
girIntraRefreshExtensionSupported = vulkanDevice->extensionSupported("VK_KHR_video_encode_intra_refresh");
girDifferentSliceTypeSupported = (h264Capabilities.flags & VK_VIDEO_ENCODE_H264_CAPABILITY_DIFFERENT_SLICE_TYPE_BIT_KHR) != 0;
girConstrainedIntraPredSupported = (h264Capabilities.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_CONSTRAINED_INTRA_PRED_FLAG_SET_BIT_KHR) != 0;
```

### Public Getters (~line 1120)
```cpp
bool hasIntraRefreshExtension() const;
bool hasDifferentSliceTypeSupport() const;
bool hasConstrainedIntraPredSupport() const;
bool canDoManualGIR() const;
bool canDoFullManualGIR() const;
```

---

## Conclusion

GIR is **now implemented** using the `VK_KHR_video_encode_intra_refresh` extension available in NVIDIA driver 590+. The implementation:

- Detects extension support at runtime
- Queries supported intra refresh modes and capabilities
- Enables GIR mode when creating the video session (if `config.enableGIR = true`)
- Chains `VkVideoEncodeIntraRefreshInfoKHR` into each encode command
- Tracks and advances the refresh cycle index automatically
- Resets the cycle on IDR frames

### Usage

To enable GIR, set in `EncoderConfig`:
```cpp
encoderConfig.enableGIR = true;
encoderConfig.girCycleDuration = 30;  // Frames for full refresh (default: 30)
```

GIR will only be activated if:
1. The extension is supported by the driver
2. At least one intra refresh mode is available
3. `config.enableGIR` is set to `true`

---

## Multi-Slice Requirement Discovery (2025-02-02)

### The Problem

Initial GIR implementation chained `VkVideoEncodeIntraRefreshInfoKHR` correctly but failed validation:

```
VUID-vkCmdEncodeVideoKHR-pEncodeInfo-10846:
naluSliceEntryCount (1) does not match intraRefreshCycleDuration (30)
but the intra refresh mode is VK_VIDEO_ENCODE_INTRA_REFRESH_MODE_PER_PICTURE_PARTITION_BIT_KHR
```

### Root Cause

Per-picture partition GIR mode requires:
- `naluSliceEntryCount == intraRefreshCycleDuration`
- For 30-frame GIR cycle: **30 slices per picture**
- Each slice covers `totalMBs / numSlices` macroblocks

### Additional Validation Requirements Found

| Validation Error | Cause | Fix |
|------------------|-------|-----|
| VUID-10839 | GIR flag set but no GIR info | Only set `VK_VIDEO_ENCODE_INTRA_REFRESH_BIT_KHR` for non-IDR frames |
| VUID-10842 | Dirty regions > 0 without flag | Set flag when reference has dirty regions |
| VUID-10844 | cycleDuration < 2 | Minimum cycle is 2 frames |
| VUID-10846 | sliceCount != cycleDuration | Multi-slice encoding required |

### Correct pNext Chain

The `VkVideoEncodeIntraRefreshInfoKHR` must be chained into `VkVideoEncodeInfoKHR.pNext`, NOT into `VkVideoEncodeH264PictureInfoKHR.pNext`:

```cpp
// CORRECT:
encodeInfo.pNext -> girInfo -> h264PicInfo

// WRONG (original bug):
encodeInfo.pNext -> h264PicInfo
h264PicInfo.pNext -> girInfo  // Driver ignores this!
```

### Current Status: IMPLEMENTED

Multi-slice encoding is now implemented for GIR per-picture partition mode:
- 30 slices per picture (matching girCycleDuration)
- Slice at `girCurrentIndex` is I-type, others are P-type
- Validated with no Vulkan validation errors
- Successfully encodes and decodes 1000 frames

---

## Multi-Slice Encoding Implementation Plan

### Macroblock Distribution

For 1280x720 resolution with 30 slices:
- MB width: `(1280 + 15) / 16 = 80`
- MB height: `(720 + 15) / 16 = 45`
- Total MBs: `80 × 45 = 3600`
- MBs per slice: `3600 / 30 = 120`
- `first_mb_in_slice`: 0, 120, 240, 360, ... 3480

### Data Structure Changes

Replace single slice with vectors:
```cpp
// Current (single slice)
StdVideoEncodeH264SliceHeader sliceHeader = {};
VkVideoEncodeH264NaluSliceInfoKHR sliceInfo = { ... };
h264PicInfo.naluSliceEntryCount = 1;

// Multi-slice for GIR
std::vector<StdVideoEncodeH264SliceHeader> sliceHeaders(numSlices);
std::vector<VkVideoEncodeH264NaluSliceInfoKHR> sliceInfos(numSlices);
h264PicInfo.naluSliceEntryCount = numSlices;
```

### Slice Configuration

```cpp
uint32_t numSlices = girEnabled ? config.girCycleDuration : 1;
uint32_t mbsPerSlice = totalMBs / numSlices;

for (uint32_t i = 0; i < numSlices; i++) {
    sliceHeaders[i].first_mb_in_slice = i * mbsPerSlice;
    sliceHeaders[i].slice_type = isP ? P_SLICE : I_SLICE;
    sliceInfos[i].pStdSliceHeader = &sliceHeaders[i];
}
```

### Hardware Validation

Must check before enabling GIR:
```cpp
if (h264Capabilities.maxSliceCount < config.girCycleDuration) {
    LOGW("GIR requires %u slices but hardware max is %u",
         config.girCycleDuration, h264Capabilities.maxSliceCount);
}
```

### Implementation Checklist

1. [x] Add `girActiveMode` member variable
2. [x] Add `maxSliceCount >= girCycleDuration` validation
3. [x] Replace single slice structs with vectors
4. [x] Calculate `first_mb_in_slice` per slice
5. [x] Update `h264PicInfo.naluSliceEntryCount` dynamically
6. [x] Remove `encoderConfig.enableGIR = false` workaround
7. [x] Test with validation layers
8. [x] Verify output playability

**Implementation Status: COMPLETE (2025-02-02)**

---

## References

- [Vulkan Video Encode H.264 Extension](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_video_encode_h264.html)
- [VkVideoEncodeH264CapabilitiesKHR](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkVideoEncodeH264CapabilitiesKHR.html)
- [H.264 Gradual Decoder Refresh](https://www.itu.int/rec/T-REC-H.264) - ITU-T Rec. H.264 Annex A
