# GPU Timestamp Latency Measurement Plan

## Objective
Measure accurate GPU latency from render completion to encode completion, with statistics reporting every 100 frames.

## Architecture Overview

### 1. Query Pool Setup
```cpp
// Per-frame queries (support maxConcurrentFrames in flight)
VkQueryPool timestampQueryPool;
const uint32_t QUERIES_PER_FRAME = 6;  // render_start, render_end, convert_start, convert_end, encode_start, encode_end
const uint32_t TOTAL_QUERIES = maxConcurrentFrames * QUERIES_PER_FRAME;

enum TimestampIndex {
    TIMESTAMP_RENDER_START = 0,
    TIMESTAMP_RENDER_END = 1,
    TIMESTAMP_CONVERT_START = 2,
    TIMESTAMP_CONVERT_END = 3,
    TIMESTAMP_ENCODE_START = 4,
    TIMESTAMP_ENCODE_END = 5
};
```

### 2. Timestamp Insertion Points

#### Render Command Buffer (buildCommandBuffer)
```cpp
// At start of command buffer
vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    timestampQueryPool, frameQueryOffset + TIMESTAMP_RENDER_START);

// After vkCmdEndRenderPass()
vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    timestampQueryPool, frameQueryOffset + TIMESTAMP_RENDER_END);
```

#### Color Conversion Command Buffer (encodeCurrentFrame)
```cpp
// At start of color conversion
vkCmdWriteTimestamp(colorConvertCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    timestampQueryPool, frameQueryOffset + TIMESTAMP_CONVERT_START);

// After rgbToNv12Converter.recordCommands()
vkCmdWriteTimestamp(colorConvertCmdBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    timestampQueryPool, frameQueryOffset + TIMESTAMP_CONVERT_END);
```

#### Encode Command Buffer (VulkanH264Encoder::recordEncodeCommands)
```cpp
// At start of encode (before vkCmdBeginVideoCodingKHR)
vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    timestampQueryPool, frameQueryOffset + TIMESTAMP_ENCODE_START);

// After vkCmdEndVideoCodingKHR()
vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    timestampQueryPool, frameQueryOffset + TIMESTAMP_ENCODE_END);
```

### 3. Statistics Tracking Structure
```cpp
struct LatencyStatistics {
    // Per-frame storage (for 100 frames)
    std::vector<double> renderLatencies;       // render_end - render_start
    std::vector<double> convertLatencies;      // convert_end - convert_start
    std::vector<double> encodeLatencies;       // encode_end - encode_start
    std::vector<double> renderToEncodeLatencies; // encode_end - render_end (KEY METRIC)
    std::vector<double> totalLatencies;        // encode_end - render_start

    uint32_t frameCount = 0;

    void addFrame(double render, double convert, double encode, double renderToEncode, double total) {
        renderLatencies.push_back(render);
        convertLatencies.push_back(convert);
        encodeLatencies.push_back(encode);
        renderToEncodeLatencies.push_back(renderToEncode);
        totalLatencies.push_back(total);
        frameCount++;

        if (frameCount >= 100) {
            reportStatistics();
            reset();
        }
    }

    void reportStatistics() {
        // Calculate and print avg/min/max for each metric
    }

    void reset() {
        renderLatencies.clear();
        convertLatencies.clear();
        encodeLatencies.clear();
        renderToEncodeLatencies.clear();
        totalLatencies.clear();
        frameCount = 0;
    }
};
```

### 4. Timestamp Retrieval and Conversion

```cpp
void retrieveTimestamps(uint32_t frameIndex) {
    // Get timestamp period (nanoseconds per tick)
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    float timestampPeriod = properties.limits.timestampPeriod;

    // Get query results for this frame
    uint32_t queryOffset = frameIndex * QUERIES_PER_FRAME;
    uint64_t timestamps[QUERIES_PER_FRAME];

    VkResult result = vkGetQueryPoolResults(
        device, timestampQueryPool,
        queryOffset, QUERIES_PER_FRAME,
        sizeof(timestamps), timestamps,
        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
    );

    if (result == VK_SUCCESS) {
        // Convert to milliseconds
        double renderTime = (timestamps[TIMESTAMP_RENDER_END] - timestamps[TIMESTAMP_RENDER_START])
                          * timestampPeriod / 1000000.0;
        double convertTime = (timestamps[TIMESTAMP_CONVERT_END] - timestamps[TIMESTAMP_CONVERT_START])
                           * timestampPeriod / 1000000.0;
        double encodeTime = (timestamps[TIMESTAMP_ENCODE_END] - timestamps[TIMESTAMP_ENCODE_START])
                          * timestampPeriod / 1000000.0;
        double renderToEncode = (timestamps[TIMESTAMP_ENCODE_END] - timestamps[TIMESTAMP_RENDER_END])
                              * timestampPeriod / 1000000.0;
        double totalTime = (timestamps[TIMESTAMP_ENCODE_END] - timestamps[TIMESTAMP_RENDER_START])
                         * timestampPeriod / 1000000.0;

        latencyStats.addFrame(renderTime, convertTime, encodeTime, renderToEncode, totalTime);
    }
}
```

### 5. Reset Query Pool Before Use
```cpp
// Reset queries before recording (outside render pass)
vkCmdResetQueryPool(cmdBuffer, timestampQueryPool, frameQueryOffset, QUERIES_PER_FRAME);
```

## Implementation Steps

### Step 1: Create Query Pool (in prepareVideoEncoding/prepareVideoEncodingHeadless)
```cpp
VkQueryPoolCreateInfo queryPoolInfo = {
    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
    .queryType = VK_QUERY_TYPE_TIMESTAMP,
    .queryCount = TOTAL_QUERIES,
};
VK_CHECK_RESULT(vkCreateQueryPool(device, &queryPoolInfo, nullptr, &timestampQueryPool));
```

### Step 2: Add Timestamp Writes
- Modify `buildCommandBuffer()` - add render timestamps
- Modify `encodeCurrentFrame()` - add color conversion timestamps
- Modify `VulkanH264Encoder::recordEncodeCommands()` - add encode timestamps

### Step 3: Retrieve Results
- Call `retrieveTimestamps()` after encode completion
- Must wait until GPU work is done (after `vkWaitForFences(encodeFence)`)

### Step 4: Statistics Reporting
- Implement `LatencyStatistics::reportStatistics()`
- Print table with avg/min/max for each metric every 100 frames

## Key Metrics to Report

### Primary Metric (Most Important)
**Render-to-Encode Latency**: `encode_end - render_end`
- This is the time from when rendering finishes to when encode completes
- Includes: color conversion + encode

### Secondary Metrics
1. **Render Time**: `render_end - render_start` (graphics work)
2. **Color Conversion Time**: `convert_end - convert_start` (compute work)
3. **Encode Time**: `encode_end - encode_start` (video encode work)
4. **Total Pipeline Time**: `encode_end - render_start` (complete frame time)

## Output Format (Every 100 Frames)
```
========== Latency Statistics (Frames 0-100) ==========
Render Time:           avg=2.34ms, min=2.10ms, max=2.89ms
Color Conversion:      avg=0.45ms, min=0.41ms, max=0.52ms
Encode Time:           avg=3.21ms, min=2.98ms, max=3.67ms
Render-to-Encode:      avg=3.66ms, min=3.39ms, max=4.19ms  <-- KEY
Total Pipeline:        avg=6.00ms, min=5.49ms, max=6.78ms
=======================================================
```

## Future-Proofing: Multiple Frames in Flight

### Current: Single Frame
```cpp
uint32_t frameQueryOffset = 0;  // Always use same queries
```

### Future: Ring Buffer
```cpp
uint32_t frameQueryOffset = (encodedFrameCount % maxConcurrentFrames) * QUERIES_PER_FRAME;
// Each in-flight frame uses different query indices
```

### Retrieval Delay
```cpp
// Don't retrieve immediately - wait N frames to ensure GPU completion
if (encodedFrameCount >= maxConcurrentFrames) {
    uint32_t retrieveFrame = (encodedFrameCount - maxConcurrentFrames) % maxConcurrentFrames;
    retrieveTimestamps(retrieveFrame);
}
```

## Considerations

### 1. Query Pool Validation
- Check `VkPhysicalDeviceLimits::timestampComputeAndGraphics` == VK_TRUE
- Check `VkQueueFamilyProperties::timestampValidBits` > 0 for each queue

### 2. Cross-Queue Timestamps
- Timestamps from different queues (graphics vs video) may not be directly comparable
- Use queue submit times as reference points if needed
- For render-to-encode latency, use encode_start timestamp as bridge

### 3. Timestamp Precision
- `timestampPeriod` varies by GPU (typically 1ns on modern hardware)
- Check `timestampValidBits` to understand precision limits

### 4. Headless Mode
- Same approach works for both windowed and headless
- Just use offscreen.colorImage instead of swapchain images

## Testing Plan
1. Enable timestamps with current single-frame encoding
2. Verify timestamp values are reasonable (ms range, not wildly off)
3. Verify statistics reporting every 100 frames
4. Test with different GOP sizes and resolutions
5. Future: Test with multiple frames in flight
