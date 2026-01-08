/*
* Vulkan Example - Taking screenshots
* 
* This sample shows how to get the conents of the swapchain (render output) and store them to disk (see saveScreenshot)
*
* Copyright (C) 2016-2025 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"
#include <fstream>
#include <chrono>
#include <atomic>
#include <cmath>
#include <bitset>
#include <cstdio>

// Logging macros - can be disabled individually or all at once
// Define DISABLE_LOGGING to disable all logging
// Or define individual levels: DISABLE_LOGD, DISABLE_LOGI, DISABLE_LOGW, DISABLE_LOGE
#define DISABLE_LOGD 1

#if defined(DISABLE_LOGGING) || defined(DISABLE_LOGD)
    #define LOGD(fmt, ...) ((void)0)
#else
    #define LOGD(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
#endif

#if defined(DISABLE_LOGGING) || defined(DISABLE_LOGI)
    #define LOGI(fmt, ...) ((void)0)
#else
    #define LOGI(fmt, ...) printf("[INFO]  " fmt "\n", ##__VA_ARGS__)
#endif

#if defined(DISABLE_LOGGING) || defined(DISABLE_LOGW)
    #define LOGW(fmt, ...) ((void)0)
#else
    #define LOGW(fmt, ...) printf("[WARN]  " fmt "\n", ##__VA_ARGS__)
#endif

#if defined(DISABLE_LOGGING) || defined(DISABLE_LOGE)
    #define LOGE(fmt, ...) ((void)0)
#else
    #define LOGE(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#endif

// RGB to NV12 Color Conversion Pipeline
// Uses a compute shader to convert RGB swapchain images to NV12 format for video encoding
class RGBtoNV12Converter {
public:
    struct NV12Image {
        // Separate single-plane images for compute shader storage (multi-planar formats don't support STORAGE_BIT)
        VkImage imageY = VK_NULL_HANDLE;         // Y plane image (R8_UNORM, full resolution)
        VkImage imageUV = VK_NULL_HANDLE;        // UV plane image (R8G8_UNORM, half resolution)
        VkDeviceMemory memoryY = VK_NULL_HANDLE;
        VkDeviceMemory memoryUV = VK_NULL_HANDLE;
        VkImageView viewY = VK_NULL_HANDLE;      // Y plane view (full resolution)
        VkImageView viewUV = VK_NULL_HANDLE;     // UV plane view (half resolution)
        
        // Multi-planar NV12 image for video encoding (separate from storage images)
        VkImage encodeImage = VK_NULL_HANDLE;    // NV12 multi-planar image for video encode
        VkDeviceMemory encodeMemory = VK_NULL_HANDLE;
        VkImageView encodeView = VK_NULL_HANDLE; // View for video encode
        
        uint32_t width = 0;
        uint32_t height = 0;
    };

private:
    // Device references (not owned)
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    vks::VulkanDevice* vulkanDevice = nullptr;

    // Compute pipeline resources
    VkPipeline computePipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkSampler inputSampler = VK_NULL_HANDLE;  // Sampler for reading swapchain images
    
    // Per-swapchain-image descriptor sets and NV12 images
    std::vector<VkDescriptorSet> descriptorSets;
    std::vector<NV12Image> nv12Images;
    
    // Swapchain image views for binding
    std::vector<VkImageView> swapchainImageViews;

    // Configuration
    uint32_t width = 0;
    uint32_t height = 0;
    bool needsSwizzle = false;  // BGR to RGB swizzle flag
    bool isInitialized = false;
    
    // Video profile for encode-compatible images (not owned)
    const VkVideoProfileListInfoKHR* videoProfileList = nullptr;

    // Push constant for shader
    struct PushConstants {
        int32_t swizzle;
        int32_t width;
        int32_t height;
    };

public:
    RGBtoNV12Converter() = default;

    ~RGBtoNV12Converter() {
        cleanup();
    }

    // Initialize the converter
    // videoProfileList is required when using VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR
    bool initialize(vks::VulkanDevice* vulkanDevice, uint32_t width, uint32_t height, 
                   VkFormat swapchainFormat, const std::vector<VkImage>& swapchainImages,
                   const std::string& shaderPath,
                   const VkVideoProfileListInfoKHR* videoProfileList = nullptr) {
        this->vulkanDevice = vulkanDevice;
        this->device = vulkanDevice->logicalDevice;
        this->physicalDevice = vulkanDevice->physicalDevice;
        this->width = width;
        this->height = height;
        this->videoProfileList = videoProfileList;

        // Check if swapchain format is BGR and needs swizzle
        std::vector<VkFormat> formatsBGR = { 
            VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SNORM 
        };
        needsSwizzle = (std::find(formatsBGR.begin(), formatsBGR.end(), swapchainFormat) == formatsBGR.end());

        // Create swapchain image views for compute shader input
        if (!createSwapchainImageViews(swapchainImages, swapchainFormat)) {
            return false;
        }

        // Create NV12 output images (one per swapchain image)
        if (!createNV12Images(static_cast<uint32_t>(swapchainImages.size()))) {
            return false;
        }

        // Create compute pipeline
        if (!createComputePipeline(shaderPath)) {
            return false;
        }

        // Create descriptor pool and sets
        if (!createDescriptorSets(static_cast<uint32_t>(swapchainImages.size()))) {
            return false;
        }

        isInitialized = true;
        LOGI("RGB to NV12 converter initialized (%ux%u, swizzle=%s)", width, height, needsSwizzle ? "yes" : "no");
        return true;
    }

    // Record compute dispatch commands for color conversion
    // If srcQueueFamily != dstQueueFamily, we need to release ownership to dstQueueFamily (video encode)
    // srcLayout: the current layout of the source image (PRESENT_SRC_KHR for windowed, TRANSFER_SRC_OPTIMAL for headless)
    void recordCommands(VkCommandBuffer cmdBuffer, uint32_t imageIndex, VkImage srcImage,
                        uint32_t srcQueueFamily = VK_QUEUE_FAMILY_IGNORED, 
                        uint32_t dstQueueFamily = VK_QUEUE_FAMILY_IGNORED,
                        VkImageLayout srcLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
        if (!isInitialized || imageIndex >= nv12Images.size()) return;

        NV12Image& nv12 = nv12Images[imageIndex];
        
        // Determine if cross-queue ownership transfer is needed
        bool needsOwnershipTransfer = (srcQueueFamily != VK_QUEUE_FAMILY_IGNORED) && 
                                       (dstQueueFamily != VK_QUEUE_FAMILY_IGNORED) &&
                                       (srcQueueFamily != dstQueueFamily);
        
        bool hasEncodeImage = (nv12.encodeImage != VK_NULL_HANDLE);

        // Transition source (swapchain/offscreen) image to SHADER_READ_ONLY_OPTIMAL for sampled read
        VkImageMemoryBarrier srcBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = srcLayout,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = srcImage,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
        };

        // Transition Y plane image to GENERAL for compute write
        VkImageMemoryBarrier yBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = nv12.imageY,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
        };

        // Transition UV plane image to GENERAL for compute write
        VkImageMemoryBarrier uvBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = nv12.imageUV,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
        };

        VkImageMemoryBarrier barriers[] = { srcBarrier, yBarrier, uvBarrier };
        vkCmdPipelineBarrier(cmdBuffer,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 3, barriers);

        // Bind compute pipeline and descriptor set
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, 
            pipelineLayout, 0, 1, &descriptorSets[imageIndex], 0, nullptr);

        // Push constants including dimensions for sampled image
        PushConstants pushConstants = { needsSwizzle ? 1 : 0, static_cast<int32_t>(width), static_cast<int32_t>(height) };
        vkCmdPushConstants(cmdBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 
            0, sizeof(PushConstants), &pushConstants);

        // Dispatch compute shader (16x16 workgroups)
        uint32_t groupCountX = (width + 15) / 16;
        uint32_t groupCountY = (height + 15) / 16;
        vkCmdDispatch(cmdBuffer, groupCountX, groupCountY, 1);

        if (hasEncodeImage) {
            LOGD("hasEncodeImage");
            // Transition Y/UV to TRANSFER_SRC, encode image planes to TRANSFER_DST
            VkImageMemoryBarrier copyBarriers[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = nv12.imageY,
                    .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
                },
                {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = nv12.imageUV,
                    .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
                },
                {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = 0,
                    .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = nv12.encodeImage,
                    .subresourceRange = { VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 1, 0, 1 }
                },
                {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = 0,
                    .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = nv12.encodeImage,
                    .subresourceRange = { VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 1, 0, 1 }
                }
            };
            vkCmdPipelineBarrier(cmdBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 4, copyBarriers);

            // Copy Y plane to encode image plane 0
            VkImageCopy yCopy = {
                .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                .srcOffset = { 0, 0, 0 },
                .dstSubresource = { VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1 },
                .dstOffset = { 0, 0, 0 },
                .extent = { width, height, 1 }
            };
            vkCmdCopyImage(cmdBuffer, nv12.imageY, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           nv12.encodeImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &yCopy);

            // Copy UV plane to encode image plane 1
            VkImageCopy uvCopy = {
                .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                .srcOffset = { 0, 0, 0 },
                .dstSubresource = { VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1 },
                .dstOffset = { 0, 0, 0 },
                .extent = { width / 2, height / 2, 1 }
            };
            vkCmdCopyImage(cmdBuffer, nv12.imageUV, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           nv12.encodeImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &uvCopy);

            // Transition encode image to VIDEO_ENCODE_SRC layout (with optional ownership transfer)
            // For queue family ownership transfer, this is the release barrier - we use NONE for 
            // dst stage/access since the acquire barrier on the video queue will handle synchronization
            VkPipelineStageFlags2 dstStage = needsOwnershipTransfer ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR;
            VkAccessFlags2 dstAccess = needsOwnershipTransfer ? VK_ACCESS_2_NONE : VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR;
            
            VkImageMemoryBarrier2 encodeBarriers[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    .dstStageMask = dstStage,
                    .dstAccessMask = dstAccess,
                    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
                    .srcQueueFamilyIndex = needsOwnershipTransfer ? srcQueueFamily : VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = needsOwnershipTransfer ? dstQueueFamily : VK_QUEUE_FAMILY_IGNORED,
                    .image = nv12.encodeImage,
                    .subresourceRange = { VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 1, 0, 1 }
                },
                {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    .dstStageMask = dstStage,
                    .dstAccessMask = dstAccess,
                    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
                    .srcQueueFamilyIndex = needsOwnershipTransfer ? srcQueueFamily : VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = needsOwnershipTransfer ? dstQueueFamily : VK_QUEUE_FAMILY_IGNORED,
                    .image = nv12.encodeImage,
                    .subresourceRange = { VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 1, 0, 1 }
                }
            };
            VkDependencyInfo depInfo = {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .imageMemoryBarrierCount = 2,
                .pImageMemoryBarriers = encodeBarriers,
            };
            vkCmdPipelineBarrier2(cmdBuffer, &depInfo);
        }

        // Transition source image back to original layout
        VkImageMemoryBarrier srcPostBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout = srcLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = srcImage,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
        };

        // srcStageMask is COMPUTE_SHADER because that's where the swapchain image was read
        // The transfer stage only touched the NV12 images, not the swapchain image
        vkCmdPipelineBarrier(cmdBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, nullptr, 0, nullptr, 1, &srcPostBarrier);
    }

    // Get NV12 image for a given swapchain index
    const NV12Image& getNV12Image(uint32_t index) const {
        return nv12Images[index];
    }

    // Get the number of NV12 images
    uint32_t getImageCount() const {
        return static_cast<uint32_t>(nv12Images.size());
    }

    bool isReady() const { return isInitialized; }

    // Cleanup all resources
    void cleanup() {
        if (device == VK_NULL_HANDLE) return;

        vkDeviceWaitIdle(device);

        // Destroy NV12 images (separate Y and UV planes, and encode image)
        for (auto& img : nv12Images) {
            if (img.encodeView != VK_NULL_HANDLE) {
                vkDestroyImageView(device, img.encodeView, nullptr);
            }
            if (img.encodeImage != VK_NULL_HANDLE) {
                vkDestroyImage(device, img.encodeImage, nullptr);
            }
            if (img.encodeMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device, img.encodeMemory, nullptr);
            }
            if (img.viewUV != VK_NULL_HANDLE) {
                vkDestroyImageView(device, img.viewUV, nullptr);
            }
            if (img.viewY != VK_NULL_HANDLE) {
                vkDestroyImageView(device, img.viewY, nullptr);
            }
            if (img.imageUV != VK_NULL_HANDLE) {
                vkDestroyImage(device, img.imageUV, nullptr);
            }
            if (img.imageY != VK_NULL_HANDLE) {
                vkDestroyImage(device, img.imageY, nullptr);
            }
            if (img.memoryUV != VK_NULL_HANDLE) {
                vkFreeMemory(device, img.memoryUV, nullptr);
            }
            if (img.memoryY != VK_NULL_HANDLE) {
                vkFreeMemory(device, img.memoryY, nullptr);
            }
        }
        nv12Images.clear();

        // Destroy swapchain image views
        for (auto& view : swapchainImageViews) {
            if (view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, view, nullptr);
            }
        }
        swapchainImageViews.clear();

        // Destroy descriptor resources
        if (descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            descriptorPool = VK_NULL_HANDLE;
        }
        descriptorSets.clear();

        if (descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            descriptorSetLayout = VK_NULL_HANDLE;
        }

        // Destroy pipeline
        if (computePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, computePipeline, nullptr);
            computePipeline = VK_NULL_HANDLE;
        }

        if (pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            pipelineLayout = VK_NULL_HANDLE;
        }

        // Destroy sampler
        if (inputSampler != VK_NULL_HANDLE) {
            vkDestroySampler(device, inputSampler, nullptr);
            inputSampler = VK_NULL_HANDLE;
        }

        isInitialized = false;
    }

private:
    // Create image views for swapchain images (for compute shader input)
    bool createSwapchainImageViews(const std::vector<VkImage>& swapchainImages, VkFormat format) {
        swapchainImageViews.resize(swapchainImages.size());

        for (size_t i = 0; i < swapchainImages.size(); i++) {
            VkImageViewCreateInfo viewInfo = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = swapchainImages[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = format,
                .components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                               VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY },
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
            };

            VkResult result = vkCreateImageView(device, &viewInfo, nullptr, &swapchainImageViews[i]);
            if (result != VK_SUCCESS) {
                LOGE("Failed to create swapchain image view %zu", i);
                return false;
            }
        }

        return true;
    }

    // Create separate Y and UV images for compute shader storage output
    // Note: Multi-planar formats (like VK_FORMAT_G8_B8R8_2PLANE_420_UNORM) do not support
    // VK_IMAGE_USAGE_STORAGE_BIT, so we use separate single-plane images instead
    // Additionally, we create a proper multi-planar NV12 image for video encoding
    bool createNV12Images(uint32_t count) {
        nv12Images.resize(count);

        for (uint32_t i = 0; i < count; i++) {
            NV12Image& img = nv12Images[i];
            img.width = width;
            img.height = height;

            // Create Y plane image (R8_UNORM, full resolution) - STORAGE only, no video encode
            VkImageCreateInfo yImageInfo = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = VK_FORMAT_R8_UNORM,
                .extent = { width, height, 1 },
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            };

            VkResult result = vkCreateImage(device, &yImageInfo, nullptr, &img.imageY);
            if (result != VK_SUCCESS) {
                LOGE("Failed to create Y plane image %u: %d", i, result);
                return false;
            }

            // Allocate memory for Y plane
            VkMemoryRequirements yMemReqs;
            vkGetImageMemoryRequirements(device, img.imageY, &yMemReqs);

            VkMemoryAllocateInfo yAllocInfo = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = yMemReqs.size,
                .memoryTypeIndex = vulkanDevice->getMemoryType(yMemReqs.memoryTypeBits, 
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
            };

            result = vkAllocateMemory(device, &yAllocInfo, nullptr, &img.memoryY);
            if (result != VK_SUCCESS) {
                LOGE("Failed to allocate Y plane memory %u", i);
                return false;
            }

            vkBindImageMemory(device, img.imageY, img.memoryY, 0);

            // Create UV plane image (R8G8_UNORM, half resolution) - STORAGE only, no video encode
            VkImageCreateInfo uvImageInfo = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = VK_FORMAT_R8G8_UNORM,
                .extent = { width / 2, height / 2, 1 },
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            };

            result = vkCreateImage(device, &uvImageInfo, nullptr, &img.imageUV);
            if (result != VK_SUCCESS) {
                LOGE("Failed to create UV plane image %u: %d", i, result);
                return false;
            }

            // Allocate memory for UV plane
            VkMemoryRequirements uvMemReqs;
            vkGetImageMemoryRequirements(device, img.imageUV, &uvMemReqs);

            VkMemoryAllocateInfo uvAllocInfo = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = uvMemReqs.size,
                .memoryTypeIndex = vulkanDevice->getMemoryType(uvMemReqs.memoryTypeBits, 
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
            };

            result = vkAllocateMemory(device, &uvAllocInfo, nullptr, &img.memoryUV);
            if (result != VK_SUCCESS) {
                LOGE("Failed to allocate UV plane memory %u", i);
                return false;
            }

            vkBindImageMemory(device, img.imageUV, img.memoryUV, 0);

            // Create Y plane view
            VkImageViewCreateInfo yViewInfo = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = img.imageY,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = VK_FORMAT_R8_UNORM,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
            };

            result = vkCreateImageView(device, &yViewInfo, nullptr, &img.viewY);
            if (result != VK_SUCCESS) {
                LOGE("Failed to create Y plane view %u", i);
                return false;
            }

            // Create UV plane view
            VkImageViewCreateInfo uvViewInfo = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = img.imageUV,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = VK_FORMAT_R8G8_UNORM,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
            };

            result = vkCreateImageView(device, &uvViewInfo, nullptr, &img.viewUV);
            if (result != VK_SUCCESS) {
                LOGE("Failed to create UV plane view %u", i);
                return false;
            }

            // Create multi-planar NV12 image for video encoding (non-disjoint, regular memory binding)
            if (videoProfileList != nullptr) {
                VkImageCreateInfo encodeImageInfo = {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                    .pNext = videoProfileList,
                    .flags = 0,  // No DISJOINT flag - use regular memory binding
                    .imageType = VK_IMAGE_TYPE_2D,
                    .format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,  // NV12 format
                    .extent = { width, height, 1 },
                    .mipLevels = 1,
                    .arrayLayers = 1,
                    .samples = VK_SAMPLE_COUNT_1_BIT,
                    .tiling = VK_IMAGE_TILING_OPTIMAL,
                    .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                };

                result = vkCreateImage(device, &encodeImageInfo, nullptr, &img.encodeImage);
                if (result != VK_SUCCESS) {
                    LOGE("Failed to create encode NV12 image %u: %d", i, result);
                    return false;
                }

                // Get memory requirements for the non-disjoint multi-planar image
                VkMemoryRequirements memReqs;
                vkGetImageMemoryRequirements(device, img.encodeImage, &memReqs);

                VkMemoryAllocateInfo encodeAllocInfo = {
                    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    .allocationSize = memReqs.size,
                    .memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
                };

                result = vkAllocateMemory(device, &encodeAllocInfo, nullptr, &img.encodeMemory);
                if (result != VK_SUCCESS) {
                    LOGE("Failed to allocate encode image memory %u", i);
                    return false;
                }

                // Bind memory (regular binding for non-disjoint image)
                result = vkBindImageMemory(device, img.encodeImage, img.encodeMemory, 0);
                if (result != VK_SUCCESS) {
                    LOGE("Failed to bind encode image memory %u: %d", i, result);
                    return false;
                }

                // Create image view for encode (viewing all planes)
                VkImageViewCreateInfo encodeViewInfo = {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .pNext = nullptr,
                    .image = img.encodeImage,
                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                    .format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                    .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
                };

                result = vkCreateImageView(device, &encodeViewInfo, nullptr, &img.encodeView);
                if (result != VK_SUCCESS) {
                    LOGE("Failed to create encode image view %u: %d", i, result);
                    return false;
                }
            }
        }

        LOGI("Created %u Y+UV image pairs (%ux%u)%s", count, width, height,
             videoProfileList ? " with NV12 encode images" : "");
        return true;
    }

    // Create compute pipeline for RGB to NV12 conversion
    bool createComputePipeline(const std::string& shaderPath) {
        // Create sampler for reading swapchain images
        VkSamplerCreateInfo samplerInfo = {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_NEAREST,
            .minFilter = VK_FILTER_NEAREST,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_FALSE,
            .compareEnable = VK_FALSE,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        };

        VkResult result = vkCreateSampler(device, &samplerInfo, nullptr, &inputSampler);
        if (result != VK_SUCCESS) {
            LOGE("Failed to create input sampler");
            return false;
        }

        // Descriptor set layout: binding 0 = input sampled image, binding 1 = Y output, binding 2 = UV output
        std::array<VkDescriptorSetLayoutBinding, 3> bindings = {{
            {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            {
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            {
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            }
        }};

        VkDescriptorSetLayoutCreateInfo layoutInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data(),
        };

        result = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
        if (result != VK_SUCCESS) {
            LOGE("Failed to create descriptor set layout");
            return false;
        }

        // Push constant range for swizzle flag
        VkPushConstantRange pushConstantRange = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
        };

        // Pipeline layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange,
        };

        result = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
        if (result != VK_SUCCESS) {
            LOGE("Failed to create pipeline layout");
            return false;
        }

        // Load compute shader
        std::string shaderFile = shaderPath + "screenshot/rgb_to_nv12.comp.spv";
        std::ifstream file(shaderFile, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            LOGE("Failed to open shader file: %s", shaderFile.c_str());
            return false;
        }

        size_t fileSize = static_cast<size_t>(file.tellg());
        std::vector<char> shaderCode(fileSize);
        file.seekg(0);
        file.read(shaderCode.data(), fileSize);
        file.close();

        VkShaderModuleCreateInfo shaderModuleInfo = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = shaderCode.size(),
            .pCode = reinterpret_cast<const uint32_t*>(shaderCode.data()),
        };

        VkShaderModule shaderModule;
        result = vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &shaderModule);
        if (result != VK_SUCCESS) {
            LOGE("Failed to create shader module");
            return false;
        }

        // Compute pipeline
        VkPipelineShaderStageCreateInfo shaderStage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shaderModule,
            .pName = "main",
        };

        VkComputePipelineCreateInfo pipelineInfo = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = shaderStage,
            .layout = pipelineLayout,
        };

        result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline);
        vkDestroyShaderModule(device, shaderModule, nullptr);

        if (result != VK_SUCCESS) {
            LOGE("Failed to create compute pipeline");
            return false;
        }

        LOGI("RGB to NV12 compute pipeline created");
        return true;
    }

    // Create descriptor pool and sets
    bool createDescriptorSets(uint32_t count) {
        // Descriptor pool - need both sampler and storage image types
        std::array<VkDescriptorPoolSize, 2> poolSizes = {{
            {
                .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = count,  // 1 sampled image per set
            },
            {
                .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = count * 2,  // 2 storage images per set (Y and UV)
            }
        }};

        VkDescriptorPoolCreateInfo poolInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = count,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data(),
        };

        VkResult result = vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
        if (result != VK_SUCCESS) {
            LOGE("Failed to create descriptor pool");
            return false;
        }

        // Allocate descriptor sets
        descriptorSets.resize(count);
        std::vector<VkDescriptorSetLayout> layouts(count, descriptorSetLayout);

        VkDescriptorSetAllocateInfo allocInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = count,
            .pSetLayouts = layouts.data(),
        };

        result = vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data());
        if (result != VK_SUCCESS) {
            LOGE("Failed to allocate descriptor sets");
            return false;
        }

        // Update descriptor sets
        for (uint32_t i = 0; i < count; i++) {
            VkDescriptorImageInfo inputImageInfo = {
                .sampler = inputSampler,
                .imageView = swapchainImageViews[i],
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            VkDescriptorImageInfo yImageInfo = {
                .imageView = nv12Images[i].viewY,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };

            VkDescriptorImageInfo uvImageInfo = {
                .imageView = nv12Images[i].viewUV,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };

            std::array<VkWriteDescriptorSet, 3> writes = {{
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptorSets[i],
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &inputImageInfo,
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptorSets[i],
                    .dstBinding = 1,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .pImageInfo = &yImageInfo,
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptorSets[i],
                    .dstBinding = 2,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .pImageInfo = &uvImageInfo,
                }
            }};

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        LOGI("Created %u descriptor sets for RGB to NV12 conversion", count);
        return true;
    }
};

// H264 Encoder Infrastructure Class
// Manages Vulkan Video encoding resources for H.264 output
class VulkanH264Encoder {
public:
    // Encoder configuration
    struct EncoderConfig {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t gopSize = 60;       // GOP size: 60 frames = 1 second at 60fps (I+P frames)
        uint32_t qp = 23;            // Constant QP for CQP rate control
        uint32_t maxFrameRate = 0;   // Max frame rate limiter (0 = unlimited)
        std::string outputPath = "recording.h264";
        bool useVBR = false;         // Enable VBR rate control
        uint32_t averageBitrate = 0; // Average bitrate (bits/s)
        uint32_t maxBitrate = 0;     // Max bitrate (bits/s)
    };

    // Per-frame encoding state
    struct EncodeFrame {
        VkImage nv12Image = VK_NULL_HANDLE;
        VkDeviceMemory nv12Memory = VK_NULL_HANDLE;
        VkImageView nv12ViewY = VK_NULL_HANDLE;      // Y plane view
        VkImageView nv12ViewUV = VK_NULL_HANDLE;     // UV plane view
        VkImageView nv12ViewFull = VK_NULL_HANDLE;   // Full image view for encode
    };

    // DPB (Decoded Picture Buffer) slot for reference frames
    struct DPBSlot {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        int32_t slotIndex = -1;
        bool inUse = false;
    };

private:
    // Device references (not owned)
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    vks::VulkanDevice* vulkanDevice = nullptr;

    // Function pointers for Vulkan Video extension
    PFN_vkCreateVideoSessionKHR fp_vkCreateVideoSessionKHR = nullptr;
    PFN_vkDestroyVideoSessionKHR fp_vkDestroyVideoSessionKHR = nullptr;
    PFN_vkGetVideoSessionMemoryRequirementsKHR fp_vkGetVideoSessionMemoryRequirementsKHR = nullptr;
    PFN_vkBindVideoSessionMemoryKHR fp_vkBindVideoSessionMemoryKHR = nullptr;
    PFN_vkCreateVideoSessionParametersKHR fp_vkCreateVideoSessionParametersKHR = nullptr;
    PFN_vkDestroyVideoSessionParametersKHR fp_vkDestroyVideoSessionParametersKHR = nullptr;
    PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR fp_vkGetPhysicalDeviceVideoCapabilitiesKHR = nullptr;
    PFN_vkGetEncodedVideoSessionParametersKHR fp_vkGetEncodedVideoSessionParametersKHR = nullptr;
    PFN_vkCmdBeginVideoCodingKHR fp_vkCmdBeginVideoCodingKHR = nullptr;
    PFN_vkCmdEndVideoCodingKHR fp_vkCmdEndVideoCodingKHR = nullptr;
    PFN_vkCmdEncodeVideoKHR fp_vkCmdEncodeVideoKHR = nullptr;
    PFN_vkCmdControlVideoCodingKHR fp_vkCmdControlVideoCodingKHR = nullptr;
    
    // Encode state
    bool sessionReset = false;
    uint32_t currentAppliedBitrate = 0;
    bool spsPpsWritten = false;  // Track if SPS/PPS has been written to file
    VkCommandBuffer encodeCommandBuffer = VK_NULL_HANDLE;
    VkFence encodeFence = VK_NULL_HANDLE;
    VkSemaphore encodeSemaphore = VK_NULL_HANDLE;
    
    // Video session resources
    VkVideoSessionKHR videoSession = VK_NULL_HANDLE;
    VkVideoSessionParametersKHR sessionParams = VK_NULL_HANDLE;
    std::vector<VkDeviceMemory> sessionMemory;
    
    // Video profile
    VkVideoProfileInfoKHR videoProfile{};
    VkVideoProfileListInfoKHR videoProfileList{};
    VkVideoEncodeH264ProfileInfoKHR h264Profile{};
    
    // Capabilities
    VkVideoCapabilitiesKHR videoCapabilities{};
    VkVideoEncodeCapabilitiesKHR encodeCapabilities{};
    VkVideoEncodeH264CapabilitiesKHR h264Capabilities{};
    
    // DPB resources
    static constexpr uint32_t MAX_DPB_SLOTS = 16;
    std::array<DPBSlot, MAX_DPB_SLOTS> dpbSlots{};
    uint32_t activeDPBSlots = 0;
    std::bitset<MAX_DPB_SLOTS> activeSlotsInSession;
    
    // Bitstream output buffer
    VkBuffer bitstreamBuffer = VK_NULL_HANDLE;
    VkDeviceMemory bitstreamMemory = VK_NULL_HANDLE;
    VkDeviceSize bitstreamBufferSize = 0;
    void* bitstreamMappedPtr = nullptr;
    
    // SPS/PPS data (generated once and reused)
    std::vector<uint8_t> spsData;
    std::vector<uint8_t> ppsData;
    bool spsGenerated = false;
    
    // Query pool for encode results
    VkQueryPool queryPool = VK_NULL_HANDLE;
    
    // Frame state
    // A sequential counter used in H.264 to identify frames in decoding order.
    // Resets to 0 for IDR frames (spec requirement: IDR frames reinitialize the DPB)
    // Increments by 1 for each subsequent frame
    uint64_t decodingOrderFrameNum = 0;
    // A sequential counter of frames encoded never resets, used to identify frames in display order.
    uint64_t streamFrameNum = 0;
    uint64_t lastIDRFrame = 0;
    uint16_t idrPicId = 0;
    
    // Reference frame tracking for P-frames
    int32_t lastRefSlotIndex = -1;      // DPB slot index of last reconstructed frame
    uint32_t lastRefFrameNum = 0;       // frame_num of last reference
    int32_t lastRefPicOrderCnt = 0;     // PicOrderCnt of last reference
    StdVideoH264PictureType lastRefPicType = STD_VIDEO_H264_PICTURE_TYPE_IDR;  // Picture type of last reference

    // User-triggered keyframe (IDR) request flag and current frame type tracking
    std::atomic<bool> forceIDRRequested{false};
    bool currentFrameIsIDR{false};
    
    // Configuration
    EncoderConfig config{};
    
    // Output file
    std::ofstream outputFile;
    bool isInitialized = false;
    
    // Video queue family
    uint32_t videoQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    VkQueue videoQueue = VK_NULL_HANDLE;
    VkCommandPool videoCommandPool = VK_NULL_HANDLE;

public:
    VulkanH264Encoder() = default;
    
    ~VulkanH264Encoder() {
        cleanup();
    }
    
    // Initialize the encoder
    bool initialize(vks::VulkanDevice* vulkanDevice, VkInstance instance, const EncoderConfig& cfg) {
        this->vulkanDevice = vulkanDevice;
        this->device = vulkanDevice->logicalDevice;
        this->physicalDevice = vulkanDevice->physicalDevice;
        this->config = cfg;

        // Load function pointers
        fp_vkCreateVideoSessionKHR = reinterpret_cast<PFN_vkCreateVideoSessionKHR>(vkGetDeviceProcAddr(device, "vkCreateVideoSessionKHR"));
        fp_vkDestroyVideoSessionKHR = reinterpret_cast<PFN_vkDestroyVideoSessionKHR>(vkGetDeviceProcAddr(device, "vkDestroyVideoSessionKHR"));
        fp_vkGetVideoSessionMemoryRequirementsKHR = reinterpret_cast<PFN_vkGetVideoSessionMemoryRequirementsKHR>(vkGetDeviceProcAddr(device, "vkGetVideoSessionMemoryRequirementsKHR"));
        fp_vkBindVideoSessionMemoryKHR = reinterpret_cast<PFN_vkBindVideoSessionMemoryKHR>(vkGetDeviceProcAddr(device, "vkBindVideoSessionMemoryKHR"));
        fp_vkCreateVideoSessionParametersKHR = reinterpret_cast<PFN_vkCreateVideoSessionParametersKHR>(vkGetDeviceProcAddr(device, "vkCreateVideoSessionParametersKHR"));
        fp_vkDestroyVideoSessionParametersKHR = reinterpret_cast<PFN_vkDestroyVideoSessionParametersKHR>(vkGetDeviceProcAddr(device, "vkDestroyVideoSessionParametersKHR"));
        
        fp_vkGetPhysicalDeviceVideoCapabilitiesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceVideoCapabilitiesKHR"));
        fp_vkGetEncodedVideoSessionParametersKHR = reinterpret_cast<PFN_vkGetEncodedVideoSessionParametersKHR>(vkGetDeviceProcAddr(device, "vkGetEncodedVideoSessionParametersKHR"));

        // Load encode command function pointers
        fp_vkCmdBeginVideoCodingKHR = reinterpret_cast<PFN_vkCmdBeginVideoCodingKHR>(vkGetDeviceProcAddr(device, "vkCmdBeginVideoCodingKHR"));
        fp_vkCmdEndVideoCodingKHR = reinterpret_cast<PFN_vkCmdEndVideoCodingKHR>(vkGetDeviceProcAddr(device, "vkCmdEndVideoCodingKHR"));
        fp_vkCmdEncodeVideoKHR = reinterpret_cast<PFN_vkCmdEncodeVideoKHR>(vkGetDeviceProcAddr(device, "vkCmdEncodeVideoKHR"));
        fp_vkCmdControlVideoCodingKHR = reinterpret_cast<PFN_vkCmdControlVideoCodingKHR>(vkGetDeviceProcAddr(device, "vkCmdControlVideoCodingKHR"));

        if (!fp_vkCreateVideoSessionKHR || !fp_vkDestroyVideoSessionKHR || !fp_vkGetVideoSessionMemoryRequirementsKHR ||
            !fp_vkBindVideoSessionMemoryKHR || !fp_vkCreateVideoSessionParametersKHR || !fp_vkDestroyVideoSessionParametersKHR ||
            !fp_vkGetPhysicalDeviceVideoCapabilitiesKHR || !fp_vkGetEncodedVideoSessionParametersKHR ||
            !fp_vkCmdBeginVideoCodingKHR || !fp_vkCmdEndVideoCodingKHR ||
            !fp_vkCmdEncodeVideoKHR || !fp_vkCmdControlVideoCodingKHR) {
            LOGE("Failed to load Vulkan Video extension functions");
            return false;
        }
        
        // Open output file
        outputFile.open(config.outputPath, std::ios::binary | std::ios::trunc);
        if (!outputFile.is_open()) {
            LOGE("Failed to open output file: %s", config.outputPath.c_str());
            return false;
        }
        
        isInitialized = true;
        return true;
    }

    // Request a one-off IDR on the next frame. No-op if gopSize==1.
    void requestKeyframe() {
        if (config.gopSize > 1) {
            forceIDRRequested.store(true);
            LOGD("[GOP] User requested IDR for next frame");
        } else {
            LOGD("[GOP] User requested IDR, but gopSize==1 (noop)");
        }
    }
    
    // Check if H264 encoding is supported
    bool isH264Supported() const {
        return vulkanDevice && vulkanDevice->extensionSupported(VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME);
    }
    
    // Setup video profiles (call before creating resources that need video profile)
    // This sets up the profile structures without creating the video session
    bool setupProfiles() {
        if (!vulkanDevice) return false;
        
        // Setup H.264 profile (Main profile, level 4.1)
        h264Profile = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PROFILE_INFO_KHR,
            .pNext = nullptr,
            .stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_MAIN,
        };
        
        // Setup video profile
        videoProfile = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
            .pNext = &h264Profile,
            .videoCodecOperation = VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR,
            .chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
            .lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
            .chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
        };
        
        // Setup profile list for resource creation
        videoProfileList = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR,
            .pNext = nullptr,
            .profileCount = 1,
            .pProfiles = &videoProfile,
        };
        
        return true;
    }
    
    // Query video encode capabilities
    bool queryCapabilities() {
        if (!vulkanDevice) return false;
        
        // Ensure profiles are set up
        if (videoProfile.sType != VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR) {
            setupProfiles();
        }
        
        // Query capabilities
        h264Capabilities = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_KHR,
            .pNext = nullptr,
        };
        
        encodeCapabilities = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_CAPABILITIES_KHR,
            .pNext = &h264Capabilities,
        };
        
        videoCapabilities = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR,
            .pNext = &encodeCapabilities,
        };
        
        VkResult result = fp_vkGetPhysicalDeviceVideoCapabilitiesKHR(
            physicalDevice, &videoProfile, &videoCapabilities);
        
        if (result != VK_SUCCESS) {
            LOGE("Failed to query video capabilities: %d", result);
            return false;
        }
        
        LOGI("H.264 Encode Capabilities:");
        LOGI("  Max coded extent: %ux%u", videoCapabilities.maxCodedExtent.width, videoCapabilities.maxCodedExtent.height);
        LOGI("  Min coded extent: %ux%u", videoCapabilities.minCodedExtent.width, videoCapabilities.minCodedExtent.height);
        LOGI("  Max DPB slots: %u", videoCapabilities.maxDpbSlots);
        LOGI("  Max active refs: %u", videoCapabilities.maxActiveReferencePictures);
        LOGI("  Min bitstream alignment: %lu", videoCapabilities.minBitstreamBufferSizeAlignment);
        LOGI("  Supported encode feedback flags: 0x%x", encodeCapabilities.supportedEncodeFeedbackFlags);
        LOGI("  Rate control modes: 0x%x", encodeCapabilities.rateControlModes);
        LOGI("  H.264 capabilities:");
        LOGI("    Max level: %d", h264Capabilities.maxLevelIdc);
        LOGI("    Max slice count: %u", h264Capabilities.maxSliceCount);
        LOGI("    Max PPicture L0 ref count: %u", h264Capabilities.maxPPictureL0ReferenceCount);
        LOGI("    Max BPicture L0 ref count: %u", h264Capabilities.maxBPictureL0ReferenceCount);
        LOGI("    Max L1 ref count: %u", h264Capabilities.maxL1ReferenceCount);
        LOGI("    Max temporal layer count: %u", h264Capabilities.maxTemporalLayerCount);
        LOGI("    Preferred max L0 ref count: %u", h264Capabilities.maxQp);
        LOGI("    Flags: 0x%x", h264Capabilities.flags);
        
        return true;
    }
    
    // Create the video session
    bool createVideoSession() {
        if (!device || !isInitialized) return false;
        
        // Use the video encode queue family index from the base device
        // This ensures we use the queue family that was actually created
        videoQueueFamilyIndex = vulkanDevice->queueFamilyIndices.videoEncode;
        
        LOGI("Using video encode queue family index: %u", videoQueueFamilyIndex);
        
        if (videoQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED || videoQueueFamilyIndex == 0xFFFFFFFF) {
            LOGE("No video encode queue family found in device");
            return false;
        }
        
        // Get video queue
        vkGetDeviceQueue(device, videoQueueFamilyIndex, 0, &videoQueue);
        
        if (videoQueue == VK_NULL_HANDLE) {
            LOGE("Failed to get video encode queue");
            return false;
        }
        
        LOGI("Got video encode queue: %p", (void*)videoQueue);
        
        // Create video session
        VkVideoSessionCreateInfoKHR sessionCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,
            .pNext = nullptr,
            .queueFamilyIndex = videoQueueFamilyIndex,
            .flags = 0,
            .pVideoProfile = &videoProfile,
            .pictureFormat = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
            .maxCodedExtent = { config.width, config.height },
            .referencePictureFormat = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
            .maxDpbSlots = std::min(videoCapabilities.maxDpbSlots, MAX_DPB_SLOTS),
            .maxActiveReferencePictures = videoCapabilities.maxActiveReferencePictures,
            .pStdHeaderVersion = &videoCapabilities.stdHeaderVersion,
        };
        
        VkResult result = fp_vkCreateVideoSessionKHR(device, &sessionCreateInfo, nullptr, &videoSession);
        if (result != VK_SUCCESS) {
            LOGE("Failed to create video session: %d", result);
            return false;
        }
        
        // Bind memory to video session
        if (!bindVideoSessionMemory()) {
            return false;
        }
        
        LOGI("Video session created successfully");
        return true;
    }
    
    // Create session parameters with SPS/PPS
    bool createSessionParameters() {
        if (!videoSession) return false;
        
        // H.264 SPS (Sequence Parameter Set)
        uint32_t width = config.width;
        uint32_t height = config.height;
        
        // Calculate cropping if resolution is not a multiple of 16
        bool croppingNeeded = (width % 16 != 0) || (height % 16 != 0);
        uint32_t cropRight = 0;
        uint32_t cropBottom = 0;
        
        if (croppingNeeded) {
            // H.264 macroblocks are 16x16
            uint32_t paddedWidth = (width + 15) / 16 * 16;
            uint32_t paddedHeight = (height + 15) / 16 * 16;
            
            cropRight = paddedWidth - width;
            cropBottom = paddedHeight - height;
            
            // Chroma format 4:2:0 means crop offsets are in units of 2 pixels
            // (Standard requires frame_crop_*_offset to be in units of CropUnitX/Y defined by chroma format)
            // For 4:2:0: CropUnitX = 2, CropUnitY = 2 * (frame_mbs_only_flag ? 1 : 2) = 2
            cropRight /= 2;
            cropBottom /= 2;
        }

        StdVideoH264SequenceParameterSet sps = {};
        sps.flags.constraint_set0_flag = 0;
        sps.flags.constraint_set1_flag = 0;
        sps.flags.constraint_set2_flag = 0;
        sps.flags.constraint_set3_flag = 0;
        sps.flags.constraint_set4_flag = 0;
        sps.flags.constraint_set5_flag = 0;
        sps.flags.direct_8x8_inference_flag = 1;
        sps.flags.frame_mbs_only_flag = 1;
        
        if (croppingNeeded) {
            sps.flags.frame_cropping_flag = 1;
            sps.frame_crop_right_offset = cropRight;
            sps.frame_crop_bottom_offset = cropBottom;
            sps.frame_crop_left_offset = 0;
            sps.frame_crop_top_offset = 0;
            LOGI("[SPS] Cropping enabled: right=%u (%upx), bottom=%u (%upx)",
                 cropRight, cropRight*2, cropBottom, cropBottom*2);
        }
        
        sps.profile_idc = STD_VIDEO_H264_PROFILE_IDC_MAIN;
        sps.level_idc = STD_VIDEO_H264_LEVEL_IDC_4_1;
        sps.seq_parameter_set_id = 0;
        sps.chroma_format_idc = STD_VIDEO_H264_CHROMA_FORMAT_IDC_420;
        sps.bit_depth_luma_minus8 = 0;
        sps.bit_depth_chroma_minus8 = 0;
        sps.log2_max_frame_num_minus4 = 4;  // max_frame_num = 256
        sps.pic_order_cnt_type = STD_VIDEO_H264_POC_TYPE_0;  // POC type 0 is more widely supported
        sps.log2_max_pic_order_cnt_lsb_minus4 = 4;  // max_pic_order_cnt_lsb = 2^8 = 256
        sps.max_num_ref_frames = 1;
        sps.pic_width_in_mbs_minus1 = (width + 15) / 16 - 1;
        sps.pic_height_in_map_units_minus1 = (height + 15) / 16 - 1;
        
        // H.264 PPS (Picture Parameter Set)
        StdVideoH264PictureParameterSet pps = {};
        pps.flags.entropy_coding_mode_flag = 0;  // CAVLC (0) instead of CABAC (1) for simpler encoding
        pps.flags.bottom_field_pic_order_in_frame_present_flag = 0;
        pps.flags.weighted_pred_flag = 0;
        pps.flags.deblocking_filter_control_present_flag = 1;
        pps.flags.constrained_intra_pred_flag = 0;
        pps.flags.redundant_pic_cnt_present_flag = 0;
        pps.flags.transform_8x8_mode_flag = 0;
        pps.flags.pic_scaling_matrix_present_flag = 0;
        pps.seq_parameter_set_id = 0;
        pps.pic_parameter_set_id = 0;
        pps.num_ref_idx_l0_default_active_minus1 = 0;
        pps.num_ref_idx_l1_default_active_minus1 = 0;
        pps.weighted_bipred_idc = STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_DEFAULT;
        pps.pic_init_qp_minus26 = 0;
        pps.chroma_qp_index_offset = 0;
        pps.second_chroma_qp_index_offset = 0;
        
        // Vulkan H.264 add info structures
        VkVideoEncodeH264SessionParametersAddInfoKHR h264AddInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
            .pNext = nullptr,
            .stdSPSCount = 1,
            .pStdSPSs = &sps,
            .stdPPSCount = 1,
            .pStdPPSs = &pps,
        };
        
        VkVideoEncodeH264SessionParametersCreateInfoKHR h264ParamsInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
            .pNext = nullptr,
            .maxStdSPSCount = 1,
            .maxStdPPSCount = 1,
            .pParametersAddInfo = &h264AddInfo,
        };
        
        VkVideoSessionParametersCreateInfoKHR paramsCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
            .pNext = &h264ParamsInfo,
            .flags = 0,
            .videoSessionParametersTemplate = VK_NULL_HANDLE,
            .videoSession = videoSession,
        };
        
        VkResult result = fp_vkCreateVideoSessionParametersKHR(device, &paramsCreateInfo, nullptr, &sessionParams);
        if (result != VK_SUCCESS) {
            LOGE("Failed to create session parameters: %d", result);
            return false;
        }
        
        LOGI("Session parameters created successfully");
        return true;
    }
    
    // Allocate bitstream buffer for encoded output
    bool createBitstreamBuffer() {
        // Size: worst case ~3 bytes per pixel + alignment
        VkDeviceSize size = static_cast<VkDeviceSize>(config.width) * config.height * 3;
        size = (size + videoCapabilities.minBitstreamBufferSizeAlignment - 1) 
             & ~(videoCapabilities.minBitstreamBufferSizeAlignment - 1);
        size = std::max(size, static_cast<VkDeviceSize>(1 << 16));  // At least 64KB
        
        VkBufferCreateInfo bufferInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = &videoProfileList,
            .flags = 0,
            .size = size,
            .usage = VK_BUFFER_USAGE_VIDEO_ENCODE_DST_BIT_KHR,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        
        VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, &bitstreamBuffer);
        if (result != VK_SUCCESS) {
            LOGE("Failed to create bitstream buffer: %d", result);
            return false;
        }
        
        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, bitstreamBuffer, &memReqs);
        
        VkMemoryAllocateInfo allocInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memReqs.size,
            .memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, 
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        };
        
        result = vkAllocateMemory(device, &allocInfo, nullptr, &bitstreamMemory);
        if (result != VK_SUCCESS) {
            LOGE("Failed to allocate bitstream memory: %d", result);
            return false;
        }
        
        vkBindBufferMemory(device, bitstreamBuffer, bitstreamMemory, 0);
        vkMapMemory(device, bitstreamMemory, 0, size, 0, &bitstreamMappedPtr);
        
        bitstreamBufferSize = size;
        LOGI("Bitstream buffer created: %lu bytes", (unsigned long)size);
        return true;
    }
    
    // Create query pool for encode feedback
    bool createQueryPool() {
        // The pNext chain must include VkVideoProfileInfoKHR for video encode feedback queries
        VkQueryPoolVideoEncodeFeedbackCreateInfoKHR feedbackInfo = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_VIDEO_ENCODE_FEEDBACK_CREATE_INFO_KHR,
            .pNext = &videoProfile,  // VkVideoProfileInfoKHR required by spec
            .encodeFeedbackFlags = VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BUFFER_OFFSET_BIT_KHR |
                                   VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BYTES_WRITTEN_BIT_KHR,
        };
        
        VkQueryPoolCreateInfo queryPoolInfo = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = &feedbackInfo,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR,
            .queryCount = 2,  // Double-buffering
        };
        
        VkResult result = vkCreateQueryPool(device, &queryPoolInfo, nullptr, &queryPool);
        if (result != VK_SUCCESS) {
            LOGE("Failed to create query pool: %d", result);
            return false;
        }
        
        LOGI("Query pool created");
        return true;
    }
    
    // Write NAL unit to file with start code
    void writeNALUnit(const uint8_t* data, size_t size) {
        if (!outputFile.is_open() || !data || size == 0) return;
        
        // NAL start code
        // startCode not needed data already includes start codes
        // static const uint8_t startCode[] = { 0x00, 0x00, 0x00, 0x01 };
        // outputFile.write(reinterpret_cast<const char*>(startCode), sizeof(startCode));
        outputFile.write(reinterpret_cast<const char*>(data), size);
    }
    
    // Retrieve and write SPS/PPS parameter sets to file
    // This should be called before the first IDR frame
    bool writeSpsPps() {
        if (!outputFile.is_open() || !sessionParams || spsPpsWritten) return false;
        
        // Set up H.264 get info structure requesting SPS id=0 and PPS id=0
        VkVideoEncodeH264SessionParametersGetInfoKHR h264GetInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_GET_INFO_KHR,
            .pNext = nullptr,
            .writeStdSPS = VK_TRUE,
            .writeStdPPS = VK_TRUE,
            .stdSPSId = 0,
            .stdPPSId = 0,
        };
        
        VkVideoEncodeSessionParametersGetInfoKHR getInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_GET_INFO_KHR,
            .pNext = &h264GetInfo,
            .videoSessionParameters = sessionParams,
        };
        
        // First call to get required buffer size
        VkVideoEncodeH264SessionParametersFeedbackInfoKHR h264Feedback = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_FEEDBACK_INFO_KHR,
            .pNext = nullptr,
        };
        
        VkVideoEncodeSessionParametersFeedbackInfoKHR feedback = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_FEEDBACK_INFO_KHR,
            .pNext = &h264Feedback,
        };
        
        size_t dataSize = 0;
        VkResult result = fp_vkGetEncodedVideoSessionParametersKHR(device, &getInfo, &feedback, &dataSize, nullptr);
        if (result != VK_SUCCESS) {
            LOGE("Failed to get SPS/PPS size: %d", result);
            return false;
        }
        
        if (dataSize == 0) {
            LOGE("SPS/PPS data size is 0");
            return false;
        }
        
        // Allocate buffer and retrieve data
        std::vector<uint8_t> paramData(dataSize);
        result = fp_vkGetEncodedVideoSessionParametersKHR(device, &getInfo, &feedback, &dataSize, paramData.data());
        if (result != VK_SUCCESS) {
            LOGE("Failed to get SPS/PPS data: %d", result);
            return false;
        }
        
        // Write to file - data already includes start codes (0x00 0x00 0x00 0x01)
        outputFile.write(reinterpret_cast<const char*>(paramData.data()), dataSize);
        
        LOGI("Wrote SPS/PPS to file: %zu bytes", dataSize);
        LOGI("  SPS written: %s", h264Feedback.hasStdSPSOverrides ? "with overrides" : "as-is");
        LOGI("  PPS written: %s", h264Feedback.hasStdPPSOverrides ? "with overrides" : "as-is");
        
        spsPpsWritten = true;
        return true;
    }
    
    // Check if SPS/PPS has been written
    bool isSpsPpsWritten() const { return spsPpsWritten; }
    
    uint32_t getCurrentBitrate0() const {
        if (!config.useVBR) return 0;
        
        // Dynamic bitrate modification demo
        // Vary bitrate based on frame counter
        double factor = (std::sin(static_cast<double>(streamFrameNum) * 0.05) + 1.0) * 0.5; // 0.0 to 1.0
        
        // Interpolate between 50% average and max*2
        uint32_t minRate = config.averageBitrate / 2;
        uint32_t range = (config.maxBitrate * 2) - minRate;
        auto bitrateRequest = minRate + static_cast<uint32_t>(range * factor);
        LOGD("[Bitrate] Frame %lu: Requesting bitrate %u bps (factor=%f)",
             (unsigned long)streamFrameNum, bitrateRequest, factor);
        return bitrateRequest;
    }

    std::pair<uint32_t, uint32_t> getCurrentBitrate() const {
        if (!config.useVBR) return {0, 0};

        if (streamFrameNum < 400) {
            LOGD("[Bitrate] Frame %lu: Requesting bitrate %u",
                 (unsigned long)streamFrameNum, config.averageBitrate);
            return {config.averageBitrate, config.averageBitrate};
        }
        LOGD("[Bitrate] Frame %lu: Requesting bitrate %u",
             (unsigned long)streamFrameNum, config.averageBitrate * 2);
        return {config.averageBitrate * 2, config.maxBitrate * 2};
    }
    
    // Check if next frame should be IDR
    bool isNextFrameIDR() const {
        bool result = (decodingOrderFrameNum == 0) || (config.gopSize > 0 && (decodingOrderFrameNum % config.gopSize == 0));
        LOGD("[GOP] Frame %lu: isNextFrameIDR = %d (gopSize=%u, mod=%lu)",
             (unsigned long)decodingOrderFrameNum, result, config.gopSize,
             (unsigned long)(decodingOrderFrameNum % config.gopSize));
        return result;
    }
    
    // Accessors
    VkVideoSessionKHR getVideoSession() const { return videoSession; }
    VkVideoSessionParametersKHR getSessionParams() const { return sessionParams; }
    VkBuffer getBitstreamBuffer() const { return bitstreamBuffer; }
    VkDeviceSize getBitstreamBufferSize() const { return bitstreamBufferSize; }
    void* getBitstreamMappedPtr() const { return bitstreamMappedPtr; }
    VkQueryPool getQueryPool() const { return queryPool; }
    const VkVideoProfileInfoKHR& getVideoProfile() const { return videoProfile; }
    const VkVideoProfileListInfoKHR& getVideoProfileList() const { return videoProfileList; }
    const VkVideoCapabilitiesKHR& getCapabilities() const { return videoCapabilities; }
    const VkVideoEncodeCapabilitiesKHR& getEncodeCapabilities() const { return encodeCapabilities; }
    const VkVideoEncodeH264CapabilitiesKHR& getH264Capabilities() const { return h264Capabilities; }
    uint32_t getVideoQueueFamilyIndex() const { return videoQueueFamilyIndex; }
    VkQueue getVideoQueue() const { return videoQueue; }
    const EncoderConfig& getConfig() const { return config; }
    bool isReady() const { return isInitialized && videoSession != VK_NULL_HANDLE && sessionParams != VK_NULL_HANDLE; }
    
    // Create DPB (Decoded Picture Buffer) images for reference frames
    bool createDPBImages() {
        if (!videoSession) return false;
        
        // For P-frame encoding (gopSize > 1), we need 2 DPB slots for ping-pong buffering
        // For I-frame only (gopSize == 1), we need 1 DPB slot
        uint32_t numDPBSlots = (config.gopSize > 1) ? 2 : 1;
        activeDPBSlots = numDPBSlots;
        
        for (uint32_t i = 0; i < numDPBSlots; i++) {
            DPBSlot& slot = dpbSlots[i];
            slot.slotIndex = static_cast<int32_t>(i);
            
            // Create DPB image
            VkImageCreateInfo imageInfo = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .pNext = &videoProfileList,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                .extent = { config.width, config.height, 1 },
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            };
            
            VkResult result = vkCreateImage(device, &imageInfo, nullptr, &slot.image);
            if (result != VK_SUCCESS) {
                LOGE("Failed to create DPB image %u: %d", i, result);
                return false;
            }
            
            // Allocate memory
            VkMemoryRequirements memReqs;
            vkGetImageMemoryRequirements(device, slot.image, &memReqs);
            
            VkMemoryAllocateInfo allocInfo = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = memReqs.size,
                .memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
            };
            
            result = vkAllocateMemory(device, &allocInfo, nullptr, &slot.memory);
            if (result != VK_SUCCESS) {
                LOGE("Failed to allocate DPB image memory %u", i);
                return false;
            }
            
            vkBindImageMemory(device, slot.image, slot.memory, 0);
            
            // Create image view for DPB
            VkImageViewCreateInfo viewInfo = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .image = slot.image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
            };
            
            result = vkCreateImageView(device, &viewInfo, nullptr, &slot.view);
            if (result != VK_SUCCESS) {
                LOGE("Failed to create DPB image view %u", i);
                return false;
            }
            
            slot.inUse = false;
        }
        
        LOGI("Created %u DPB images", numDPBSlots);
        return true;
    }
    
    // Full initialization sequence - call after initialize()
    bool setupVideoSession() {
        if (!isInitialized) {
            LOGE("Encoder not initialized");
            return false;
        }
        
        // Step 1: Query capabilities
        if (!queryCapabilities()) {
            LOGE("Failed to query video capabilities");
            return false;
        }
        
        // Step 2: Create video session
        if (!createVideoSession()) {
            LOGE("Failed to create video session");
            return false;
        }
        
        // Step 3: Create session parameters (SPS/PPS)
        if (!createSessionParameters()) {
            LOGE("Failed to create session parameters");
            return false;
        }
        
        // Step 4: Create DPB images
        if (!createDPBImages()) {
            LOGE("Failed to create DPB images");
            return false;
        }
        
        // Step 5: Create bitstream buffer
        if (!createBitstreamBuffer()) {
            LOGE("Failed to create bitstream buffer");
            return false;
        }
        
        // Step 6: Create query pool
        if (!createQueryPool()) {
            LOGE("Failed to create query pool");
            return false;
        }
        
        // Step 7: Create command pool for video queue
        if (!createVideoCommandPool()) {
            LOGE("Failed to create video command pool");
            return false;
        }
        
        // Step 8: Create encode command buffer and sync objects
        if (!createEncodeSyncObjects()) {
            LOGE("Failed to create encode sync objects");
            return false;
        }
        
        LOGI("Video encode session fully initialized");
        return true;
    }
    
    // Encode a single frame
    // srcImage: NV12 multi-planar image (G8_B8R8_2PLANE_420_UNORM)
    // srcView: View of the NV12 image
    // srcQueueFamily: queue family index that released ownership (for cross-queue sync)
    bool encodeFrame(VkImage srcImage, VkImageView srcView,
                     VkQueue graphicsQueue, VkSemaphore waitSemaphore = VK_NULL_HANDLE,
                     uint32_t srcQueueFamily = VK_QUEUE_FAMILY_IGNORED) {
        if (!isReady()) return false;

        LOGD("[Encode] Starting frame %lu/%lu", (unsigned long)decodingOrderFrameNum, (unsigned long)streamFrameNum);
        
        // Wait for previous encode to complete
        LOGD("[Encode] Waiting for previous encode fence...");
        vkWaitForFences(device, 1, &encodeFence, VK_TRUE, UINT64_MAX);
        LOGD("[Encode] Previous fence signaled, resetting...");
        vkResetFences(device, 1, &encodeFence);
        
        // Reset command buffer
        vkResetCommandBuffer(encodeCommandBuffer, 0);
        
        // Begin command buffer
        VkCommandBufferBeginInfo beginInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        VK_CHECK_RESULT(vkBeginCommandBuffer(encodeCommandBuffer, &beginInfo));
        
        // Record encode commands with queue family ownership transfer if needed
        LOGD("[Encode] Recording encode commands...");
        recordEncodeCommands(encodeCommandBuffer, srcImage, srcView, srcQueueFamily);
        
        VK_CHECK_RESULT(vkEndCommandBuffer(encodeCommandBuffer));
        LOGD("[Encode] Command buffer recorded");
        
        // Submit encode command buffer
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo submitInfo = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = waitSemaphore != VK_NULL_HANDLE ? 1u : 0u,
            .pWaitSemaphores = waitSemaphore != VK_NULL_HANDLE ? &waitSemaphore : nullptr,
            .pWaitDstStageMask = waitSemaphore != VK_NULL_HANDLE ? &waitStage : nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &encodeCommandBuffer,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr,
        };
        
        LOGD("[Encode] Submitting to video queue (waitSemaphore=%s)...",
             waitSemaphore != VK_NULL_HANDLE ? "yes" : "no");
        VkResult result = vkQueueSubmit(videoQueue, 1, &submitInfo, encodeFence);
        if (result != VK_SUCCESS) {
            LOGE("Failed to submit encode command buffer: %d", result);
            return false;
        }
        LOGD("[Encode] Submitted successfully");
        
        // Wait for encode to complete and read back results
        LOGD("[Encode] Waiting for encode fence...");
        result = vkWaitForFences(device, 1, &encodeFence, VK_TRUE, UINT64_MAX);
        LOGD("[Encode] Encode fence signaled (result=%d)", result);
        
        // Also wait on the queue to ensure all work is done
        LOGD("[Encode] Waiting for video queue idle...");
        result = vkQueueWaitIdle(videoQueue);
        LOGD("[Encode] Video queue idle (result=%d)", result);
        
        // Query encode results
        // Try reading raw bytes first to see what the driver actually writes
        uint8_t rawData[64] = {0};
        
        // First try with just the status bit to see if query is available at all
        result = vkGetQueryPoolResults(device, queryPool, 0, 1, sizeof(rawData), rawData,
            sizeof(rawData), VK_QUERY_RESULT_WITH_STATUS_BIT_KHR);
        
        LOGD("[Encode] Raw query result: %d", result);
#ifndef DISABLE_LOGD
        // Build hex dump string
        char hexBuf[96]; // 32 bytes * 3 chars per byte
        char* p = hexBuf;
        for (int i = 0; i < 32; i++) {
            p += sprintf(p, "%02x ", rawData[i]);
        }
        LOGD("[Encode] Raw bytes: %s", hexBuf);
#endif
        
        if (result == VK_NOT_READY) {
            // Try with 64-bit flag
            result = vkGetQueryPoolResults(device, queryPool, 0, 1, sizeof(rawData), rawData,
                sizeof(rawData), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_STATUS_BIT_KHR);
            LOGD("[Encode] With 64-bit: result=%d", result);
        }
        
        // Parse results - format depends on flags used
        // With VK_QUERY_RESULT_WITH_STATUS_BIT_KHR:
        //   - First comes the feedback data (offset, bytesWritten) 
        //   - Then status as int32_t
        // Without VK_QUERY_RESULT_64_BIT, each value is 32-bit
        struct EncodeFeedback32 {
            uint32_t offset;
            uint32_t bytesWritten;
            int32_t status;
        };
        EncodeFeedback32* fb32 = reinterpret_cast<EncodeFeedback32*>(rawData);
        LOGD("[Encode] As 32-bit: offset=%u, bytes=%u, status=%d",
             fb32->offset, fb32->bytesWritten, fb32->status);
        
        // Also try 64-bit interpretation
        struct EncodeFeedback64 {
            uint64_t offset;
            uint64_t bytesWritten;
            int64_t status;
        };
        EncodeFeedback64* fb64 = reinterpret_cast<EncodeFeedback64*>(rawData);
        LOGD("[Encode] As 64-bit: offset=%lu, bytes=%lu, status=%ld",
             (unsigned long)fb64->offset, (unsigned long)fb64->bytesWritten, (long)fb64->status);
        
        // If query not ready, something went wrong with the encode
        if (result == VK_NOT_READY) {
            LOGE("[Encode] Query not ready after GPU completed - encode may have been skipped");
            // Check if maybe the status field has useful info
            LOGE("[Encode] Status from raw data: 32-bit=%d, 64-bit=%ld",
                 fb32->status, (long)fb64->status);
            return false;
        }
        
        // Check status from 32-bit interpretation (without VK_QUERY_RESULT_64_BIT)
        if (fb32->status != VK_QUERY_RESULT_STATUS_COMPLETE_KHR) {
            LOGE("[Encode] Encode status not complete: %d (COMPLETE=%d, ERROR=%d)",
                 fb32->status, VK_QUERY_RESULT_STATUS_COMPLETE_KHR, VK_QUERY_RESULT_STATUS_ERROR_KHR);
            // Still continue to see what data we got
        }
        
        // Write encoded data to file using 32-bit values
        if (result == VK_SUCCESS && fb32->bytesWritten > 0 && bitstreamMappedPtr) {
            // For IDR frames, ensure SPS/PPS is written first
            // Use the decided frame type from command recording
            bool isIDR = currentFrameIsIDR;
            if (isIDR && !spsPpsWritten) {
                writeSpsPps();
            }
            
            LOGD("[Encode] Writing %u bytes at offset %u%s",
                 fb32->bytesWritten, fb32->offset, isIDR ? " (IDR frame)" : "");
            
            const uint8_t* data = static_cast<const uint8_t*>(bitstreamMappedPtr) + fb32->offset;
            writeNALUnit(data, static_cast<size_t>(fb32->bytesWritten));
        }
        
        decodingOrderFrameNum++;
        streamFrameNum++;
        return result == VK_SUCCESS;
    }
    
    // Get DPB slot for encoding
    const DPBSlot& getDPBSlot(uint32_t index) const {
        return dpbSlots[index % activeDPBSlots];
    }
    
    uint32_t getActiveDPBSlots() const { return activeDPBSlots; }
    VkCommandPool getVideoCommandPool() const { return videoCommandPool; }
    
    // Cleanup all resources
    void cleanup() {
        if (device == VK_NULL_HANDLE) return;
        
        vkDeviceWaitIdle(device);
        
        if (outputFile.is_open()) {
            outputFile.close();
        }
        
        if (queryPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device, queryPool, nullptr);
            queryPool = VK_NULL_HANDLE;
        }
        
        // Cleanup DPB slots
        for (auto& slot : dpbSlots) {
            if (slot.view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, slot.view, nullptr);
                slot.view = VK_NULL_HANDLE;
            }
            if (slot.image != VK_NULL_HANDLE) {
                vkDestroyImage(device, slot.image, nullptr);
                slot.image = VK_NULL_HANDLE;
            }
            if (slot.memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, slot.memory, nullptr);
                slot.memory = VK_NULL_HANDLE;
            }
        }
        
        if (bitstreamMappedPtr) {
            vkUnmapMemory(device, bitstreamMemory);
            bitstreamMappedPtr = nullptr;
        }
        
        if (bitstreamBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, bitstreamBuffer, nullptr);
            bitstreamBuffer = VK_NULL_HANDLE;
        }
        
        if (bitstreamMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, bitstreamMemory, nullptr);
            bitstreamMemory = VK_NULL_HANDLE;
        }
        
        if (sessionParams != VK_NULL_HANDLE) {
            fp_vkDestroyVideoSessionParametersKHR(device, sessionParams, nullptr);
            sessionParams = VK_NULL_HANDLE;
        }
        
        if (videoSession != VK_NULL_HANDLE) {
            fp_vkDestroyVideoSessionKHR(device, videoSession, nullptr);
            videoSession = VK_NULL_HANDLE;
        }
        
        if (videoCommandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, videoCommandPool, nullptr);
            videoCommandPool = VK_NULL_HANDLE;
        }
        
        if (encodeFence != VK_NULL_HANDLE) {
            vkDestroyFence(device, encodeFence, nullptr);
            encodeFence = VK_NULL_HANDLE;
        }
        
        if (encodeSemaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, encodeSemaphore, nullptr);
            encodeSemaphore = VK_NULL_HANDLE;
        }
        
        for (auto& mem : sessionMemory) {
            if (mem != VK_NULL_HANDLE) {
                vkFreeMemory(device, mem, nullptr);
            }
        }
        sessionMemory.clear();
        
        // Reset encoder logic state
        activeDPBSlots = 0;
        activeSlotsInSession.reset();
        
        // Reset frame counters
        decodingOrderFrameNum = 0;
        streamFrameNum = 0;
        lastIDRFrame = 0;
        idrPicId = 0;
        
        // Reset reference tracking
        lastRefSlotIndex = -1;
        lastRefFrameNum = 0;
        lastRefPicOrderCnt = 0;
        lastRefPicType = STD_VIDEO_H264_PICTURE_TYPE_IDR;
        
        // Reset SPS generation state so new resolution generates new SPS
        spsGenerated = false;
        spsPpsWritten = false;
        
        // Reset session state
        sessionReset = false;
        currentAppliedBitrate = 0;
        
        isInitialized = false;
    }

private:
    // Helper to determine if current frame should be encoded as P-frame
    bool isPFrame() const {
        bool notIDR = !isNextFrameIDR();
        bool result = config.gopSize > 1 && notIDR && decodingOrderFrameNum > 0;
        LOGD("[GOP] Frame %lu: isPFrame = %d (gopSize>1: %d, notIDR: %d, frameCounter>0: %d, lastRefSlot: %d)",
             (unsigned long)decodingOrderFrameNum, result, (config.gopSize > 1), notIDR,
             (decodingOrderFrameNum > 0), lastRefSlotIndex);
        return result;
    }
    
    // Bind memory to video session (required before use)
    bool bindVideoSessionMemory() {
        uint32_t memReqCount = 0;
        fp_vkGetVideoSessionMemoryRequirementsKHR(device, videoSession, &memReqCount, nullptr);
        
        if (memReqCount == 0) return true;
        
        std::vector<VkVideoSessionMemoryRequirementsKHR> memReqs(memReqCount);
        for (auto& req : memReqs) {
            req.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR;
            req.pNext = nullptr;
        }
        fp_vkGetVideoSessionMemoryRequirementsKHR(device, videoSession, &memReqCount, memReqs.data());
        
        std::vector<VkBindVideoSessionMemoryInfoKHR> bindInfos(memReqCount);
        sessionMemory.resize(memReqCount);
        
        for (uint32_t i = 0; i < memReqCount; i++) {
            // Try to find device local memory first, fall back to any matching memory type
            VkBool32 memTypeFound = VK_FALSE;
            uint32_t memTypeIndex = vulkanDevice->getMemoryType(
                memReqs[i].memoryRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                &memTypeFound);
            if (!memTypeFound) {
                // Fallback: find any memory type that matches the requirements
                memTypeIndex = vulkanDevice->getMemoryType(
                    memReqs[i].memoryRequirements.memoryTypeBits,
                    0);  // No specific property requirements
            }
            VkMemoryAllocateInfo allocInfo = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = memReqs[i].memoryRequirements.size,
                .memoryTypeIndex = memTypeIndex,
            };
            
            VkResult result = vkAllocateMemory(device, &allocInfo, nullptr, &sessionMemory[i]);
            if (result != VK_SUCCESS) {
                LOGE("Failed to allocate video session memory: %d", result);
                return false;
            }
            
            bindInfos[i] = {
                .sType = VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR,
                .pNext = nullptr,
                .memoryBindIndex = memReqs[i].memoryBindIndex,
                .memory = sessionMemory[i],
                .memoryOffset = 0,
                .memorySize = memReqs[i].memoryRequirements.size,
            };
        }
        
        VkResult result = fp_vkBindVideoSessionMemoryKHR(device, videoSession, memReqCount, bindInfos.data());
        if (result != VK_SUCCESS) {
            LOGE("Failed to bind video session memory: %d", result);
            return false;
        }
        
        LOGI("Video session memory bound (%u allocations)", memReqCount);
        return true;
    }
    
    // Create command pool for video encode queue
    bool createVideoCommandPool() {
        VkCommandPoolCreateInfo poolInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = videoQueueFamilyIndex,
        };
        
        VkResult result = vkCreateCommandPool(device, &poolInfo, nullptr, &videoCommandPool);
        if (result != VK_SUCCESS) {
            LOGE("Failed to create video command pool: %d", result);
            return false;
        }
        
        LOGI("Video command pool created");
        return true;
    }
    
    // Create command buffer and synchronization objects for encoding
    bool createEncodeSyncObjects() {
        // Allocate command buffer
        VkCommandBufferAllocateInfo allocInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = videoCommandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        
        VkResult result = vkAllocateCommandBuffers(device, &allocInfo, &encodeCommandBuffer);
        if (result != VK_SUCCESS) {
            LOGE("Failed to allocate encode command buffer: %d", result);
            return false;
        }
        
        // Create fence
        VkFenceCreateInfo fenceInfo = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,  // Start signaled so first wait doesn't block
        };
        
        result = vkCreateFence(device, &fenceInfo, nullptr, &encodeFence);
        if (result != VK_SUCCESS) {
            LOGE("Failed to create encode fence: %d", result);
            return false;
        }
        
        // Create semaphore
        VkSemaphoreCreateInfo semaphoreInfo = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };
        
        result = vkCreateSemaphore(device, &semaphoreInfo, nullptr, &encodeSemaphore);
        if (result != VK_SUCCESS) {
            LOGE("Failed to create encode semaphore: %d", result);
            return false;
        }
        
        LOGI("Encode sync objects created");
        return true;
    }
    
    // Record video encode commands into command buffer
    // srcImage: the NV12 multi-planar source image
    // srcView: view of the NV12 image
    // srcQueueFamily: the queue family that released ownership (compute/graphics)
    // If srcQueueFamily differs from video queue family, we need to acquire ownership
    void recordEncodeCommands(VkCommandBuffer cmdBuffer, VkImage srcImage, VkImageView srcView,
                              uint32_t srcQueueFamily = VK_QUEUE_FAMILY_IGNORED) {
        LOGD("\n========== ENCODE FRAME %lu/%lu ==========",
             (unsigned long)decodingOrderFrameNum, (unsigned long)streamFrameNum);
        LOGD("[GOP] Config: gopSize=%u, qp=%u", config.gopSize, config.qp);
        LOGD("[GOP] State: lastRefSlot=%d, lastRefFrameNum=%u, lastRefPOC=%d",
             lastRefSlotIndex, lastRefFrameNum, lastRefPicOrderCnt);
        
        // Apply forced IDR request if present (additional to periodic GOP keyframes)
        bool forcedIDR = forceIDRRequested.exchange(false) && (config.gopSize > 1);
        if (forcedIDR) {
            LOGD("[GOP] Forcing IDR for this frame per user request");
        }
        bool isIDR = forcedIDR || isNextFrameIDR();
        currentFrameIsIDR = isIDR;
        bool isP = (config.gopSize > 1) && !isIDR && decodingOrderFrameNum > 0;
        
        LOGD("[GOP] Frame type determined: IDR=%d, P=%d => %s",
             isIDR, isP, (isIDR ? "IDR" : (isP ? "P-frame" : "I-frame")));
        
        // Determine DPB slot for current reconstructed frame
        // For P-frames, use ping-pong between slot 0 and 1
        int32_t slotIndex = (config.gopSize > 1) ? static_cast<int32_t>(decodingOrderFrameNum % 2) : 0;
        LOGD("[GOP] Using DPB slot: %d", slotIndex);
        
        // DPB slot reference
        DPBSlot& dpbSlot = dpbSlots[slotIndex];
        
        // Determine if we need to acquire ownership from another queue family
        bool needsOwnershipAcquire = (srcQueueFamily != VK_QUEUE_FAMILY_IGNORED) && 
                                      (srcQueueFamily != videoQueueFamilyIndex);
        
        // Build barriers dynamically based on whether we have a reference frame
        // Base barriers: 2 for source image planes + 2 for output DPB slot planes
        // For P-frames: + 2 more for reference DPB slot planes
        std::vector<VkImageMemoryBarrier2> preBarriers;
        preBarriers.reserve(isP && lastRefSlotIndex >= 0 ? 6 : 4);
        
        // Source image plane 0 barrier
        preBarriers.push_back({
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_NONE,  // Already synchronized via semaphore
            .srcAccessMask = VK_ACCESS_2_NONE,
            .dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR,
            .oldLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
            .newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
            .srcQueueFamilyIndex = needsOwnershipAcquire ? srcQueueFamily : VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = needsOwnershipAcquire ? videoQueueFamilyIndex : VK_QUEUE_FAMILY_IGNORED,
            .image = srcImage,
            .subresourceRange = { VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 1, 0, 1 }
        });
        
        // Source image plane 1 barrier
        preBarriers.push_back({
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
            .srcAccessMask = VK_ACCESS_2_NONE,
            .dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR,
            .oldLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
            .newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
            .srcQueueFamilyIndex = needsOwnershipAcquire ? srcQueueFamily : VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = needsOwnershipAcquire ? videoQueueFamilyIndex : VK_QUEUE_FAMILY_IGNORED,
            .image = srcImage,
            .subresourceRange = { VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 1, 0, 1 }
        });
        
        // Output DPB image plane 0 barrier (for reconstructed frame)
        preBarriers.push_back({
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
            .srcAccessMask = VK_ACCESS_2_NONE,
            .dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR | VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = dpbSlot.image,
            .subresourceRange = { VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 1, 0, 1 }
        });
        
        // Output DPB image plane 1 barrier
        preBarriers.push_back({
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
            .srcAccessMask = VK_ACCESS_2_NONE,
            .dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR | VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = dpbSlot.image,
            .subresourceRange = { VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 1, 0, 1 }
        });
        
        // For P-frames, add barriers for the REFERENCE DPB slot (the previous reconstructed frame)
        // This ensures the reference data is properly synchronized for reading
        if (isP && lastRefSlotIndex >= 0) {
            DPBSlot& refDpbSlot = dpbSlots[lastRefSlotIndex];
            LOGD("[GOP] Adding reference DPB barriers for slot %d", lastRefSlotIndex);
            
            // Reference DPB image plane 0 barrier - transition from DPB layout to DPB layout
            // (preserves content, just ensures synchronization)
            preBarriers.push_back({
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
                .srcAccessMask = VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR,
                .dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
                .dstAccessMask = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR,
                .oldLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR,
                .newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = refDpbSlot.image,
                .subresourceRange = { VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 1, 0, 1 }
            });
            
            // Reference DPB image plane 1 barrier
            preBarriers.push_back({
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
                .srcAccessMask = VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR,
                .dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
                .dstAccessMask = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR,
                .oldLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR,
                .newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = refDpbSlot.image,
                .subresourceRange = { VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 1, 0, 1 }
            });
        }
        
        VkDependencyInfo dependencyInfo = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = static_cast<uint32_t>(preBarriers.size()),
            .pImageMemoryBarriers = preBarriers.data(),
        };
        
        vkCmdPipelineBarrier2(cmdBuffer, &dependencyInfo);
        
        // H.264 slice header info
        StdVideoEncodeH264SliceHeader sliceHeader = {};
        sliceHeader.flags.direct_spatial_mv_pred_flag = 0;
        sliceHeader.flags.num_ref_idx_active_override_flag = 0;
        sliceHeader.first_mb_in_slice = 0;  // First macroblock
        sliceHeader.slice_type = isP ? STD_VIDEO_H264_SLICE_TYPE_P : STD_VIDEO_H264_SLICE_TYPE_I;
        sliceHeader.slice_alpha_c0_offset_div2 = 0;
        sliceHeader.slice_beta_offset_div2 = 0;
        sliceHeader.slice_qp_delta = 0;
        // cabac_init_idc is ignored when entropy_coding_mode_flag is 0 (CAVLC)
        sliceHeader.cabac_init_idc = STD_VIDEO_H264_CABAC_INIT_IDC_0;
        sliceHeader.disable_deblocking_filter_idc = STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_DISABLED;
        
        LOGD("[GOP] Slice type: %d(%s-slice)", (int)sliceHeader.slice_type, isP ? "P" : "I");        
        // Build reference lists for P-frames
        StdVideoEncodeH264ReferenceListsInfo refLists = {};
        // Initialize all entries to NO_REFERENCE to satisfy VUID 08339
        for (uint32_t i = 0; i < STD_VIDEO_H264_MAX_NUM_LIST_REF; ++i) {
            refLists.RefPicList0[i] = STD_VIDEO_H264_NO_REFERENCE_PICTURE;
            refLists.RefPicList1[i] = STD_VIDEO_H264_NO_REFERENCE_PICTURE;
        }
        if (isP && lastRefSlotIndex >= 0) {
            // One valid reference in L0 pointing to the input ref slot
            refLists.RefPicList0[0] = static_cast<uint8_t>(lastRefSlotIndex);  // DPB slot index of reference frame
            refLists.num_ref_idx_l0_active_minus1 = 0;  // 1 reference in L0
            refLists.num_ref_idx_l1_active_minus1 = 0;
            LOGD("[GOP] Building reference list: L0[0]=%d, num_ref_l0=1", (int)refLists.RefPicList0[0]);
        } else {
            // No references used; keep lists filled with NO_REFERENCE and set counts to 0
            refLists.num_ref_idx_l0_active_minus1 = 0;
            refLists.num_ref_idx_l1_active_minus1 = 0;
            LOGD("[GOP] No reference list (I-frame or no previous ref)");
        }
        
        StdVideoEncodeH264PictureInfo stdPicInfo = {};
        stdPicInfo.flags.IdrPicFlag = isIDR ? 1 : 0;
        stdPicInfo.flags.is_reference = 1;
        stdPicInfo.flags.no_output_of_prior_pics_flag = 0;
        stdPicInfo.flags.long_term_reference_flag = 0;
        stdPicInfo.flags.adaptive_ref_pic_marking_mode_flag = 0;
        stdPicInfo.seq_parameter_set_id = 0;
        stdPicInfo.pic_parameter_set_id = 0;
        stdPicInfo.idr_pic_id = isIDR ? idrPicId++ : 0;
        // For IDR frames, primary_pic_type must be IDR; for P-frames use P; for I-frames use I
        stdPicInfo.primary_pic_type = isIDR ? STD_VIDEO_H264_PICTURE_TYPE_IDR : (isP ? STD_VIDEO_H264_PICTURE_TYPE_P : STD_VIDEO_H264_PICTURE_TYPE_I);
        /*
        * Frame Numbering and Picture Order Count (POC):
        * 
        * frame_num: A sequential counter used in H.264 to identify frames in decoding order.
        * - Resets to 0 for IDR frames (spec requirement: IDR frames reinitialize the DPB)
        * - Increments by 1 for each subsequent frame
        * - Wraps around at 256 (modulo operation) because frame_num is transmitted in the bitstream
        *   using log2_max_frame_num_minus4, which with our SPS setting yields a maximum of 256
        * - This wrapping is essential for bitstream compactness and decoder compatibility
        * 
        * PicOrderCnt (POC): Defines the display order of frames for proper temporal sequencing.
        * - For POC type 0 (used in this implementation), POC is derived from frame count
        * - Resets to 0 for IDR frames (part of IDR semantics - new sequence starts)
        * - Calculated as (frameCounter * 2) to allow for potential field coding support
        * - Wraps around at 256 due to log2_max_pic_order_cnt_lsb_minus4 setting in SPS
        * - The modulo 256 operation prevents POC from exceeding max_pic_order_cnt_lsb
        * - POC wrapping is mandated by H.264 spec and ensures decoder can correctly compute
        *   display ordering even with the limited bit width used in the bitstream
        * 
        * The 256 wrap value comes from: 2^(log2_max_frame_num_minus4 + 4) where
        * log2_max_frame_num_minus4 = 4 in our SPS configuration, giving 2^8 = 256.
        * 
        * Both values MUST wrap at the same boundary to maintain synchronization between
        * decoding order (frame_num) and display order (POC) in the H.264 bitstream.
        */
        // For IDR frames, frame_num should be 0 since IDR resets the DPB
        stdPicInfo.frame_num = isIDR ? 0 : static_cast<uint32_t>(decodingOrderFrameNum % 256);
        // For POC type 0, PicOrderCnt should wrap around at max_pic_order_cnt_lsb (256)
        // For IDR frames, POC starts at 0
        stdPicInfo.PicOrderCnt = isIDR ? 0 : static_cast<int32_t>((decodingOrderFrameNum * 2) % 256);
        if (isIDR) {
            // if stdPicInfo.frame_num is not set correctly ffplay ignores or skips frames randomly
            decodingOrderFrameNum = 0;  // Reset frame counter after IDR for consistent POC
        }
        stdPicInfo.temporal_id = 0;
        stdPicInfo.pRefLists = (isP && lastRefSlotIndex >= 0) ? &refLists : nullptr;
        
        LOGD("[GOP] Picture info: IdrFlag=%d, primary_pic_type=%d, frame_num=%u, POC=%d, pRefLists=%s",
             (int)stdPicInfo.flags.IdrPicFlag, (int)stdPicInfo.primary_pic_type,
             stdPicInfo.frame_num, stdPicInfo.PicOrderCnt, (stdPicInfo.pRefLists ? "SET" : "NULL"));
        
        // H.264 NALU slice info
        VkVideoEncodeH264NaluSliceInfoKHR sliceInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_NALU_SLICE_INFO_KHR,
            .pNext = nullptr,
            .constantQp = config.useVBR ? 0 : static_cast<int32_t>(config.qp),
            .pStdSliceHeader = &sliceHeader,
        };
        
        // H.264 picture info
        VkVideoEncodeH264PictureInfoKHR h264PicInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PICTURE_INFO_KHR,
            .pNext = nullptr,
            .naluSliceEntryCount = 1,
            .pNaluSliceEntries = &sliceInfo,
            .pStdPictureInfo = &stdPicInfo,
            .generatePrefixNalu = VK_FALSE,
        };
        
        // DPB picture resource
        VkVideoPictureResourceInfoKHR dpbPicResource = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
            .pNext = nullptr,
            .codedOffset = { 0, 0 },
            .codedExtent = { config.width, config.height },
            .baseArrayLayer = 0,
            .imageViewBinding = dpbSlot.view,
        };
        
        // H.264 DPB slot info for current reconstructed frame
        StdVideoEncodeH264ReferenceInfo stdRefInfo = {};
        stdRefInfo.flags.used_for_long_term_reference = 0;
        // Reference info should match the picture type
        stdRefInfo.primary_pic_type = isIDR ? STD_VIDEO_H264_PICTURE_TYPE_IDR : (isP ? STD_VIDEO_H264_PICTURE_TYPE_P : STD_VIDEO_H264_PICTURE_TYPE_I);
        // For IDR frames, frame_num is 0; POC also starts at 0
        stdRefInfo.FrameNum = isIDR ? 0 : static_cast<uint32_t>(decodingOrderFrameNum % 256);
        stdRefInfo.PicOrderCnt = isIDR ? 0 : static_cast<int32_t>((decodingOrderFrameNum * 2) % 256);
        stdRefInfo.long_term_pic_num = 0;
        stdRefInfo.long_term_frame_idx = 0;
        stdRefInfo.temporal_id = 0;
        
        VkVideoEncodeH264DpbSlotInfoKHR h264DpbSlotInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR,
            .pNext = nullptr,
            .pStdReferenceInfo = &stdRefInfo,
        };
        
        // Reference slot for setup (output)
        VkVideoReferenceSlotInfoKHR setupSlot = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
            .pNext = &h264DpbSlotInfo,
            .slotIndex = slotIndex,
            .pPictureResource = &dpbPicResource,
        };
        
        // Source picture resource (input frame)
        VkVideoPictureResourceInfoKHR srcPicResource = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
            .pNext = nullptr,
            .codedOffset = { 0, 0 },
            .codedExtent = { config.width, config.height },
            .baseArrayLayer = 0,
            .imageViewBinding = srcView,  // Use NV12 image view as source
        };
        
        // Setup input reference for P-frames
        VkVideoPictureResourceInfoKHR refPicResource = {};
        StdVideoEncodeH264ReferenceInfo refStdInfo = {};
        VkVideoEncodeH264DpbSlotInfoKHR refH264DpbSlotInfo = {};
        VkVideoReferenceSlotInfoKHR inputRefSlot = {};
        
        if (isP && lastRefSlotIndex >= 0) {
            LOGD("[GOP] Setting up input reference from DPB slot %d", lastRefSlotIndex);            // Get the reference DPB slot
            DPBSlot& refSlot = dpbSlots[lastRefSlotIndex];
            
            // Reference picture resource
            refPicResource = {
                .sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
                .pNext = nullptr,
                .codedOffset = { 0, 0 },
                .codedExtent = { config.width, config.height },
                .baseArrayLayer = 0,
                .imageViewBinding = refSlot.view,
            };
            
            // Reference frame info - use the actual picture type of the reference frame
            refStdInfo.flags.used_for_long_term_reference = 0;
            refStdInfo.primary_pic_type = lastRefPicType;  // Use tracked picture type of reference
            refStdInfo.FrameNum = lastRefFrameNum;
            refStdInfo.PicOrderCnt = lastRefPicOrderCnt;
            refStdInfo.long_term_pic_num = 0;
            refStdInfo.long_term_frame_idx = 0;
            refStdInfo.temporal_id = 0;
            
            LOGD("[GOP] Input ref: FrameNum=%u, POC=%d, picType=%d",
                 refStdInfo.FrameNum, refStdInfo.PicOrderCnt, (int)refStdInfo.primary_pic_type);            
            refH264DpbSlotInfo = {
                .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR,
                .pNext = nullptr,
                .pStdReferenceInfo = &refStdInfo,
            };
            
            // Input reference slot
            inputRefSlot = {
                .sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
                .pNext = &refH264DpbSlotInfo,
                .slotIndex = lastRefSlotIndex,
                .pPictureResource = &refPicResource,
            };
        } else {
            LOGD("[GOP] No input reference (isP=%d, lastRefSlot=%d)", isP, lastRefSlotIndex);
        }
        
        // Encode info
        VkVideoEncodeInfoKHR encodeInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_INFO_KHR,
            .pNext = &h264PicInfo,
            .flags = 0,
            .dstBuffer = bitstreamBuffer,
            .dstBufferOffset = 0,
            .dstBufferRange = bitstreamBufferSize,
            .srcPictureResource = srcPicResource,
            .pSetupReferenceSlot = &setupSlot,
            .referenceSlotCount = (isP && lastRefSlotIndex >= 0) ? 1u : 0u,
            .pReferenceSlots = (isP && lastRefSlotIndex >= 0) ? &inputRefSlot : nullptr,
            .precedingExternallyEncodedBytes = 0,
        };
        
        // Setup reference slots for begin coding
        // For P-frames, we need to include both input reference and output setup slots
        // For I-frames/IDR, we only need a placeholder for tracking
        std::array<VkVideoReferenceSlotInfoKHR, 2> beginSlots;
        uint32_t beginSlotCount = 0;
        
        if (isP && lastRefSlotIndex >= 0) {
            // P-frame: Include input reference slot
            beginSlots[beginSlotCount++] = inputRefSlot;
            LOGD("[GOP] Begin coding: including input ref slot %d", lastRefSlotIndex);
        }
        
        // Output/Setup slot MUST be included in the bound reference slots so validation passes
        // Error 08215: pEncodeInfo->pSetupReferenceSlot->pPictureResource must match one of the bound reference picture resource
        VkVideoReferenceSlotInfoKHR beginSetupSlot = setupSlot;
        if (slotIndex >= 0 && slotIndex < MAX_DPB_SLOTS && !activeSlotsInSession[slotIndex]) {
            beginSetupSlot.slotIndex = -1; // Not yet active, use -1 to avoid VUID-vkCmdBeginVideoCodingKHR-slotIndex-07239
        }
        beginSlots[beginSlotCount++] = beginSetupSlot;
        
        // Reset query pool before beginning video coding (must be outside video coding scope)
        vkCmdResetQueryPool(cmdBuffer, queryPool, 0, 1);
        
        // Rate Control Layer Info (for VBR)
        // We must provide H.264 specific rate control layer info
        VkVideoEncodeH264RateControlLayerInfoKHR h264RateControlLayer = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_LAYER_INFO_KHR,
            .pNext = nullptr,
            .useMinQp = VK_FALSE,
            .minQp = {0, 0, 0},
            .useMaxQp = VK_FALSE,
            .maxQp = {51, 51, 51},
            .useMaxFrameSize = VK_FALSE,
            .maxFrameSize = {0, 0, 0},
        };

        const auto [avgBitrate, maxBitrate] = getCurrentBitrate();
        VkVideoEncodeRateControlLayerInfoKHR rateControlLayerInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_LAYER_INFO_KHR,
            .pNext = &h264RateControlLayer,
            .averageBitrate = avgBitrate,
            .maxBitrate = maxBitrate,
            .frameRateNumerator = (config.maxFrameRate > 0) ? config.maxFrameRate : 60,
            .frameRateDenominator = 1,
        };

        if (config.useVBR) {
            uint32_t currentRate = rateControlLayerInfo.averageBitrate;
            if (currentRate != currentAppliedBitrate) {
                // If it's a new bitrate, we just log here. 
                // We'll apply it using vkCmdControlVideoCodingKHR inside the coding scope.
                if (sessionReset) {
                    LOGD("[Encode] VBR Bitrate Update Detected: %u bps", currentRate);
                }
            }
        }

        // Determine Rate Control Mode
        VkVideoEncodeRateControlModeFlagBitsKHR rcMode = config.useVBR 
            ? VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR 
            : VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR;

        // Rate control info
        // We must also provide H.264 specific rate control info in the pNext chain
        VkVideoEncodeH264RateControlInfoKHR h264RateControlInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_INFO_KHR,
            .pNext = nullptr,
            .flags = VK_VIDEO_ENCODE_H264_RATE_CONTROL_REGULAR_GOP_BIT_KHR | 
                     VK_VIDEO_ENCODE_H264_RATE_CONTROL_REFERENCE_PATTERN_FLAT_BIT_KHR,
            .gopFrameCount = config.gopSize,
            .idrPeriod = config.gopSize,
            .consecutiveBFrameCount = 0,
            .temporalLayerCount = (config.useVBR ? 1u : 0u), // Must be non-zero if VBR
        };

        VkVideoEncodeRateControlInfoKHR rateControlInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_INFO_KHR,
            .pNext = &h264RateControlInfo,
            .flags = 0,
            .rateControlMode = rcMode,
            .layerCount = config.useVBR ? 1u : 0u,
            .pLayers = config.useVBR ? &rateControlLayerInfo : nullptr,
            .virtualBufferSizeInMs = config.useVBR ? 1000u : 0u, // Set leaky bucket size for VBR
            .initialVirtualBufferSizeInMs = 0u, // Must be less than virtualBufferSizeInMs (start with empty buffer)
        };
        
        // Begin video coding
        // - First frame: Don't include rate control (it's still DEFAULT)
        // - Subsequent frames: Include rate control matching current state
        VkVideoBeginCodingInfoKHR beginInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
            .pNext = sessionReset ? &rateControlInfo : nullptr,  // Only after rate control is configured
            .flags = 0,
            .videoSession = videoSession,
            .videoSessionParameters = sessionParams,
            .referenceSlotCount = beginSlotCount,
            .pReferenceSlots = beginSlots.data(),
        };

        // If rate control parameters have changed but we are NOT in the first frame (sessionReset is true),
        // we must not pass the NEW parameters in pNext of BeginCoding because they don't match the CURRENT state.
        // We will update them via ControlVideoCoding inside the block.
        // So, if VBR is active and bitrate changed, we should pass the OLD parameters (or none, if that was allowed, 
        // but for VBR we usually need persistent state).
        // 
        // Actually, the spec says: "if the pNext chain... includes an instance of VkVideoEncodeRateControlInfoKHR... 
        // it must match the rate control state configured... at the time the command is executed."
        //
        // This means if we are about to CHANGE the rate control (because bitrate changed), we must pass the 
        // *CURRENTLY ACTIVE* rate control info to BeginCoding, not the *NEW* target info.
        
        // Create a copy of rate control info reflecting the OLD/CURRENT state for BeginCoding validation
        VkVideoEncodeRateControlLayerInfoKHR currentLayerInfo = rateControlLayerInfo;
        VkVideoEncodeRateControlInfoKHR currentRCInfo = rateControlInfo;
        
        if (sessionReset && config.useVBR && rateControlLayerInfo.averageBitrate != currentAppliedBitrate) {
             // Revert layer info to match currently applied bitrate
             currentLayerInfo.averageBitrate = currentAppliedBitrate;
             currentLayerInfo.maxBitrate = currentAppliedBitrate;
             
             // Point to the reverted layer info
             currentRCInfo.pLayers = &currentLayerInfo;
             
             // Use this "current state" info for BeginCoding validation
             beginInfo.pNext = &currentRCInfo;
        }

        LOGD("[GOP] Begin coding with %u reference slot(s)", beginSlotCount);
        
        fp_vkCmdBeginVideoCodingKHR(cmdBuffer, &beginInfo);
        
        // Reset session and configure rate control on first frame
        if (!sessionReset) {
            // Apply reset and rate control in one command
            // Note: RESET is performed before processing other flags
            VkVideoCodingControlInfoKHR controlInfo = {
                .sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
                .pNext = &rateControlInfo,
                .flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR | VK_VIDEO_CODING_CONTROL_ENCODE_RATE_CONTROL_BIT_KHR,
            };
            fp_vkCmdControlVideoCodingKHR(cmdBuffer, &controlInfo);
            
            currentAppliedBitrate = rateControlLayerInfo.averageBitrate;
            sessionReset = true;
            activeSlotsInSession.reset();
        } else if (config.useVBR && rateControlLayerInfo.averageBitrate != currentAppliedBitrate) {
            // Bitrate changed in an active session - update rate control
            VkVideoCodingControlInfoKHR controlInfo = {
                .sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
                .pNext = &rateControlInfo,
                .flags = VK_VIDEO_CODING_CONTROL_ENCODE_RATE_CONTROL_BIT_KHR,
            };
            fp_vkCmdControlVideoCodingKHR(cmdBuffer, &controlInfo);
            
            LOGD("[Encode] VBR Bitrate Updated to %lu bps", rateControlLayerInfo.averageBitrate);
            currentAppliedBitrate = rateControlLayerInfo.averageBitrate;
        }
        
        // Begin query - use index 0
        // Note: For video encode feedback queries, we use the query within the video coding scope
        LOGD("[Encode] Beginning query...");
        vkCmdBeginQuery(cmdBuffer, queryPool, 0, 0);
        
        // Encode
        LOGD("[Encode] Recording vkCmdEncodeVideoKHR...");
        fp_vkCmdEncodeVideoKHR(cmdBuffer, &encodeInfo);
        LOGD("[Encode] Encode command recorded");

        if (slotIndex >= 0 && slotIndex < MAX_DPB_SLOTS) {
            activeSlotsInSession[slotIndex] = true;
        }
        
        // End query
        vkCmdEndQuery(cmdBuffer, queryPool, 0);
        LOGD("[Encode] Query ended");
        
        // Update reference tracking for next P-frame
        // Store current frame as reference for next frame
        lastRefSlotIndex = slotIndex;
        lastRefFrameNum = stdPicInfo.frame_num;
        lastRefPicOrderCnt = stdPicInfo.PicOrderCnt;
        lastRefPicType = stdPicInfo.primary_pic_type;

        LOGD("[GOP] Updated reference tracking: slot=%d, frameNum=%u, POC=%d, picType=%d",
             lastRefSlotIndex, lastRefFrameNum, lastRefPicOrderCnt, (int)lastRefPicType);
        LOGD("========== END ENCODE FRAME %lu/%lu ==========",
             (unsigned long)decodingOrderFrameNum, (unsigned long)streamFrameNum);
        
        // End video coding
        VkVideoEndCodingInfoKHR endInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR,
            .pNext = nullptr,
            .flags = 0,
        };
        
        fp_vkCmdEndVideoCodingKHR(cmdBuffer, &endInfo);
    }
};

class VulkanExample : public VulkanExampleBase
{
public:
	vkglTF::Model model;

	struct UniformData {
		glm::mat4 projection{};
		glm::mat4 model{};
		glm::mat4 view{};
		int32_t texIndex = 0;
	} uniformData;
	std::array<vks::Buffer, maxConcurrentFrames> uniformBuffers;

	// Feature structures (must persist until device creation)
	VkPhysicalDeviceSynchronization2Features synchronization2Features{};
	VkPhysicalDeviceVideoMaintenance1FeaturesKHR videoMaintenance1Features{};

	VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
	VkPipeline pipeline{ VK_NULL_HANDLE };
	VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
	std::array<VkDescriptorSet, maxConcurrentFrames> descriptorSets{};

	bool screenshotSaved{ false };
	bool recordingEnabled{ false };
	uint64_t encodedFrameCount{ 0 };
	std::chrono::steady_clock::time_point lastEncodeTime{};

	// Video encoding resources
	RGBtoNV12Converter rgbToNv12Converter;
	VulkanH264Encoder h264Encoder;
	
	// Command buffer for RGB to NV12 conversion (runs on graphics queue)
	VkCommandBuffer colorConvertCmdBuffer{ VK_NULL_HANDLE };
	VkFence colorConvertFence{ VK_NULL_HANDLE };
	
	// Cross-queue synchronization semaphores
	VkSemaphore renderCompleteSemaphore{ VK_NULL_HANDLE };      // Graphics -> Compute
	VkSemaphore colorConvertCompleteSemaphore{ VK_NULL_HANDLE }; // Compute -> Video Encode
	
	// Queue family indices for ownership transfers
	uint32_t graphicsQueueFamily{ VK_QUEUE_FAMILY_IGNORED };
	uint32_t videoQueueFamily{ VK_QUEUE_FAMILY_IGNORED };

	// rendering resources for headless mode
	struct HeadlessResources {
		VkImage colorImage{ VK_NULL_HANDLE };
		VkDeviceMemory colorMemory{ VK_NULL_HANDLE };
		VkImageView colorView{ VK_NULL_HANDLE };
		VkFramebuffer framebuffer{ VK_NULL_HANDLE };
		uint32_t width{ 0 };
		uint32_t height{ 0 };
	} offscreen;

	// Headless mode configuration
	static constexpr uint32_t HEADLESS_FRAME_COUNT = 1000;
	uint32_t headlessFramesRendered{ 0 };

	VulkanExample() : VulkanExampleBase(), uniformBuffers{}
	{
		title = "Saving framebuffer to screenshot";
		apiVersion = VK_API_VERSION_1_3;
	    settings.validation = true;
		// Request video encode queue for H.264 encoding
		requestedQueueTypes = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_VIDEO_ENCODE_BIT_KHR;
		camera.type = Camera::CameraType::lookat;
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
		camera.setRotation(glm::vec3(-25.0f, 23.75f, 0.0f));
		camera.setTranslation(glm::vec3(0.0f, 0.0f, -3.0f));
	}

	~VulkanExample() override
	{
		if (device) {
			// Wait for any pending encode operations
			vkDeviceWaitIdle(device);
			
			// Cleanup offscreen resources (headless mode)
			if (offscreen.framebuffer != VK_NULL_HANDLE) {
				vkDestroyFramebuffer(device, offscreen.framebuffer, nullptr);
			}
			if (offscreen.colorView != VK_NULL_HANDLE) {
				vkDestroyImageView(device, offscreen.colorView, nullptr);
			}
			if (offscreen.colorImage != VK_NULL_HANDLE) {
				vkDestroyImage(device, offscreen.colorImage, nullptr);
			}
			if (offscreen.colorMemory != VK_NULL_HANDLE) {
				vkFreeMemory(device, offscreen.colorMemory, nullptr);
			}
			
			// Cleanup color conversion resources
			if (colorConvertFence != VK_NULL_HANDLE) {
				vkDestroyFence(device, colorConvertFence, nullptr);
			}
			if (renderCompleteSemaphore != VK_NULL_HANDLE) {
				vkDestroySemaphore(device, renderCompleteSemaphore, nullptr);
			}
			if (colorConvertCompleteSemaphore != VK_NULL_HANDLE) {
				vkDestroySemaphore(device, colorConvertCompleteSemaphore, nullptr);
			}
			// colorConvertCmdBuffer is freed with the command pool
			
			vkDestroyPipeline(device, pipeline, nullptr);
			vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
			for (auto& buffer : uniformBuffers) {
				buffer.destroy();
			}
		}
	}
    void getEnabledFeatures() override
	{
		// Enable synchronization2 feature for vkCmdPipelineBarrier2
		// We always add this to the pNext chain - if the extension isn't supported,
		// getEnabledExtensions() won't add it and device creation will ignore this
		synchronization2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
		synchronization2Features.pNext = deviceCreatepNextChain;
		synchronization2Features.synchronization2 = VK_TRUE;
		deviceCreatepNextChain = &synchronization2Features;

		// Enable video maintenance1 feature
		videoMaintenance1Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR;
		videoMaintenance1Features.pNext = deviceCreatepNextChain;
		videoMaintenance1Features.videoMaintenance1 = VK_TRUE;
		deviceCreatepNextChain = &videoMaintenance1Features;
	}
    void getEnabledExtensions() override
	{
	    // Check for Vulkan Video extensions
	    bool videoQueue = vulkanDevice->extensionSupported(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);
	    bool encodeQueue = vulkanDevice->extensionSupported(VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME);
		bool sync2 = vulkanDevice->extensionSupported(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
		bool maintenance1 = vulkanDevice->extensionSupported(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME);

	    LOGI("Vulkan Video Encoding Support:");
	    LOGI("  Base Video Queue: %s", videoQueue ? "Yes" : "No");
	    LOGI("  Encode Queue:     %s", encodeQueue ? "Yes" : "No");
		LOGI("  Synchronization2: %s", sync2 ? "Yes" : "No");
		LOGI("  Maintenance1:     %s", maintenance1 ? "Yes" : "No");

	    // Check for specific encoders
        if (videoQueue && encodeQueue && sync2 && maintenance1) {
            enabledDeviceExtensions.push_back(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);
            enabledDeviceExtensions.push_back(VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME);
            enabledDeviceExtensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
            enabledDeviceExtensions.push_back(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME);

            LOGI("  Supported Encoders:");
            bool h264 = vulkanDevice->extensionSupported(VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME);
            bool h265 = vulkanDevice->extensionSupported(VK_KHR_VIDEO_ENCODE_H265_EXTENSION_NAME);
            bool av1 = vulkanDevice->extensionSupported(VK_KHR_VIDEO_ENCODE_AV1_EXTENSION_NAME);
            LOGI("    H.264: %s", h264 ? "Yes" : "No");
            LOGI("    H.265: %s", h265 ? "Yes" : "No");
            LOGI("    AV1:   %s", av1 ? "Yes" : "No");

            if (h264) {
                enabledDeviceExtensions.push_back(VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME);
            }
            if (h265) {
                enabledDeviceExtensions.push_back(VK_KHR_VIDEO_ENCODE_H265_EXTENSION_NAME);
            }
            // Note: AV1 extension not enabled - validation layers don't support it yet
            // Uncomment when validation layer support is added:
            // if (av1) {
            //     enabledDeviceExtensions.push_back(VK_KHR_VIDEO_ENCODE_AV1_EXTENSION_NAME);
            // }
        }
	}

	// Setup offscreen rendering resources for headless mode
	void setupHeadlessResources()
	{
		offscreen.width = width;
		offscreen.height = height;

		// Create color attachment image
		VkImageCreateInfo imageInfo = vks::initializers::imageCreateInfo();
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VK_CHECK_RESULT(vkCreateImage(device, &imageInfo, nullptr, &offscreen.colorImage));

		// Allocate memory
		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(device, offscreen.colorImage, &memReqs);
		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &offscreen.colorMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device, offscreen.colorImage, offscreen.colorMemory, 0));

		// Create image view
		VkImageViewCreateInfo viewInfo = vks::initializers::imageViewCreateInfo();
		viewInfo.image = offscreen.colorImage;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;
		VK_CHECK_RESULT(vkCreateImageView(device, &viewInfo, nullptr, &offscreen.colorView));

		LOGI("Offscreen color attachment created: %ux%u", width, height);
	}

	// Setup render pass for headless mode with TRANSFER_SRC_OPTIMAL final layout
	void setupHeadlessRenderPass()
	{
		std::array<VkAttachmentDescription, 2> attachments = {};
		// Color attachment
		attachments[0].format = VK_FORMAT_B8G8R8A8_UNORM;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

		// Depth attachment
		attachments[1].format = depthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
		VkAttachmentReference depthReference = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

		VkSubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpassDescription.colorAttachmentCount = 1;
		subpassDescription.pColorAttachments = &colorReference;
		subpassDescription.pDepthStencilAttachment = &depthReference;

		std::array<VkSubpassDependency, 2> dependencies;
		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo renderPassInfo = vks::initializers::renderPassCreateInfo();
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDescription;
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();

		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
		LOGI("Headless render pass created");
	}

	// Setup framebuffer for headless mode
	void setupHeadlessFramebuffer()
	{
		std::array<VkImageView, 2> attachments = { offscreen.colorView, depthStencil.view };

		VkFramebufferCreateInfo fbInfo = vks::initializers::framebufferCreateInfo();
		fbInfo.renderPass = renderPass;
		fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		fbInfo.pAttachments = attachments.data();
		fbInfo.width = width;
		fbInfo.height = height;
		fbInfo.layers = 1;

		VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbInfo, nullptr, &offscreen.framebuffer));
		LOGI("Headless framebuffer created: %ux%u", width, height);
	}

	// Headless render loop - renders fixed number of frames and encodes them
	void renderLoopHeadless()
	{
		LOGI("Starting headless rendering of %u frames...", HEADLESS_FRAME_COUNT);
		
		auto startTime = std::chrono::high_resolution_clock::now();
		
		for (headlessFramesRendered = 0; headlessFramesRendered < HEADLESS_FRAME_COUNT; headlessFramesRendered++) {
			// Progress output every 100 frames
			if (headlessFramesRendered % 100 == 0) {
				LOGI("Rendering frame %u/%u", headlessFramesRendered, HEADLESS_FRAME_COUNT);
			}
			
			// Update uniforms (animate camera slightly for visual verification)
			camera.rotate(glm::vec3(0.01f, 0.0f, 0.0f));
			uniformData.projection = camera.matrices.perspective;
			uniformData.view = camera.matrices.view;
			uniformData.model = glm::mat4(1.0f);
			uniformBuffers[0].copyTo(&uniformData, sizeof(UniformData));
			
			// Build and record command buffer
			buildHeadlessCommandBuffer();
			
			// Submit rendering
			VkSubmitInfo submitInfo = vks::initializers::submitInfo();
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &drawCmdBuffers[0];
			
			// Reset fence before submitting (fence is created signaled by base class)
			VK_CHECK_RESULT(vkResetFences(device, 1, &waitFences[0]));
			VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, waitFences[0]));
			VK_CHECK_RESULT(vkWaitForFences(device, 1, &waitFences[0], VK_TRUE, UINT64_MAX));
			
			// Encode frame
			if (h264Encoder.isReady() && rgbToNv12Converter.isReady()) {
				encodeHeadlessFrame();
			}
		}
		
		auto endTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
		
		LOGI("Headless rendering complete!");
		LOGI("  Total frames: %u", HEADLESS_FRAME_COUNT);
		LOGI("  Total time: %ldms", (long)duration);
		LOGI("  Average FPS: %.2f", (HEADLESS_FRAME_COUNT * 1000.0 / duration));
		LOGI("  Encoded frames: %lu", encodedFrameCount);
		
		vkDeviceWaitIdle(device);
	}

	// Build command buffer for headless rendering
	void buildHeadlessCommandBuffer()
	{
		VkCommandBuffer cmdBuffer = drawCmdBuffers[0];
		
		VK_CHECK_RESULT(vkResetCommandBuffer(cmdBuffer, 0));
		
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2]{};
		clearValues[0].color = defaultClearColor;
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;
		renderPassBeginInfo.framebuffer = offscreen.framebuffer;

		VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));
		vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
		VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
		vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);
		VkRect2D scissor = vks::initializers::rect2D(static_cast<int32_t>(width), static_cast<int32_t>(height), 0, 0);
		vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);
		vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[0], 0, nullptr);
		vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		model.draw(cmdBuffer);
		// Note: No UI in headless mode
		vkCmdEndRenderPass(cmdBuffer);
		VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
	}

	// Encode a frame in headless mode
	void encodeHeadlessFrame()
	{
		// Wait for previous color conversion to complete
		vkWaitForFences(device, 1, &colorConvertFence, VK_TRUE, UINT64_MAX);
		vkResetFences(device, 1, &colorConvertFence);
		
		// Record color conversion commands
		vkResetCommandBuffer(colorConvertCmdBuffer, 0);
		
		VkCommandBufferBeginInfo beginInfo = vks::initializers::commandBufferBeginInfo();
		VK_CHECK_RESULT(vkBeginCommandBuffer(colorConvertCmdBuffer, &beginInfo));
		
		// Use TRANSFER_SRC_OPTIMAL for headless mode (set by render pass final layout)
		rgbToNv12Converter.recordCommands(colorConvertCmdBuffer, 0, 
		                                   offscreen.colorImage,
		                                   graphicsQueueFamily, videoQueueFamily,
		                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		
		VK_CHECK_RESULT(vkEndCommandBuffer(colorConvertCmdBuffer));
		
		// Submit color conversion
		VkSubmitInfo submitInfo = vks::initializers::submitInfo();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &colorConvertCmdBuffer;
		
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, colorConvertFence));
		vkWaitForFences(device, 1, &colorConvertFence, VK_TRUE, UINT64_MAX);
		
		// Get NV12 image for encoding (always index 0 in headless mode)
		const auto& nv12Image = rgbToNv12Converter.getNV12Image(0);
		
		// Encode the frame
		if (h264Encoder.encodeFrame(nv12Image.encodeImage, nv12Image.encodeView, queue,
		                            VK_NULL_HANDLE, graphicsQueueFamily)) {
			encodedFrameCount++;
		}
	}

	void loadAssets()
	{
		model.loadFromFile(getAssetPath() + "models/chinesedragon.gltf", vulkanDevice, queue, vkglTF::FileLoadingFlags::PreTransformVertices | vkglTF::FileLoadingFlags::PreMultiplyVertexColors | vkglTF::FileLoadingFlags::FlipY);
	}

	void setupDescriptors()
	{
		// Pool
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, maxConcurrentFrames),
		};
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, maxConcurrentFrames);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

		// Layout
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),		// Binding 0: Vertex shader uniform buffer
		};
		VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

		// Sets per frame, just like the buffers themselves
		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);
		for (auto i = 0; i < uniformBuffers.size(); i++) {
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets[i]));
			std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
				vks::initializers::writeDescriptorSet(descriptorSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers[i].descriptor),
			};
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
		}
	}

	void preparePipelines()
	{
		// Layout
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

		// Pipeline
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
			loadShader(getShadersPath() + "screenshot/mesh.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getShadersPath() + "screenshot/mesh.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
		};

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass, 0);
		pipelineCI.pInputAssemblyState = &inputAssemblyState;
		pipelineCI.pRasterizationState = &rasterizationState;
		pipelineCI.pColorBlendState = &colorBlendState;
		pipelineCI.pMultisampleState = &multisampleState;
		pipelineCI.pViewportState = &viewportState;
		pipelineCI.pDepthStencilState = &depthStencilState;
		pipelineCI.pDynamicState = &dynamicState;
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();
		pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({vkglTF::VertexComponent::Position, vkglTF::VertexComponent::Normal, vkglTF::VertexComponent::Color});
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipeline));
	}

	void prepareUniformBuffers()
	{
		for (auto& buffer : uniformBuffers) {
			VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer, sizeof(UniformData), &uniformData));
			VK_CHECK_RESULT(buffer.map());
		}
	}

	void updateUniformBuffers()
	{
	    camera.rotate(glm::vec3(0.01f, 0.0f, 0.0f));
		uniformData.projection = camera.matrices.perspective;
		uniformData.view = camera.matrices.view;
		uniformData.model = glm::mat4(1.0f);
		uniformBuffers[currentBuffer].copyTo(&uniformData, sizeof(UniformData));
	}

	// Take a screenshot from the current swapchain image
	// This is done using a blit from the swapchain image to a linear image whose memory content is then saved as a ppm image
	// Getting the image date directly from a swapchain image wouldn't work as they're usually stored in an implementation dependent optimal tiling format
	// Note: This requires the swapchain images to be created with the VK_IMAGE_USAGE_TRANSFER_SRC_BIT flag (see VulkanSwapChain::create)
	void saveScreenshot(const char *filename)
	{
		screenshotSaved = false;
		bool supportsBlit = true;

		// Check blit support for source and destination
		VkFormatProperties formatProps;

		// Check if the device supports blitting from optimal images (the swapchain images are in optimal format)
		vkGetPhysicalDeviceFormatProperties(physicalDevice, swapChain.colorFormat, &formatProps);
		if (!(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)) {
			LOGW("Device does not support blitting from optimal tiled images, using copy instead of blit!");
			supportsBlit = false;
		}

		// Check if the device supports blitting to linear images
		vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, &formatProps);
		if (!(formatProps.linearTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
			LOGW("Device does not support blitting to linear tiled images, using copy instead of blit!");
			supportsBlit = false;
		}

		// Source for the copy is the last rendered swapchain image
		VkImage srcImage = swapChain.images[currentBuffer];

		// Create the linear tiled destination image to copy to and to read the memory from
		VkImageCreateInfo imageCreateCI(vks::initializers::imageCreateInfo());
		imageCreateCI.imageType = VK_IMAGE_TYPE_2D;
		// Note that vkCmdBlitImage (if supported) will also do format conversions if the swapchain color format would differ
		imageCreateCI.format = VK_FORMAT_R8G8B8A8_UNORM;
		imageCreateCI.extent.width = width;
		imageCreateCI.extent.height = height;
		imageCreateCI.extent.depth = 1;
		imageCreateCI.arrayLayers = 1;
		imageCreateCI.mipLevels = 1;
		imageCreateCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageCreateCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateCI.tiling = VK_IMAGE_TILING_LINEAR;
		imageCreateCI.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		// Create the image
		VkImage dstImage;
		VK_CHECK_RESULT(vkCreateImage(device, &imageCreateCI, nullptr, &dstImage));
		// Create memory to back up the image
		VkMemoryRequirements memRequirements;
		VkMemoryAllocateInfo memAllocInfo(vks::initializers::memoryAllocateInfo());
		VkDeviceMemory dstImageMemory;
		vkGetImageMemoryRequirements(device, dstImage, &memRequirements);
		memAllocInfo.allocationSize = memRequirements.size;
		// Memory must be host visible to copy from
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAllocInfo, nullptr, &dstImageMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device, dstImage, dstImageMemory, 0));

		// Do the actual blit from the swapchain image to our host visible destination image
		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

		// Transition destination image to transfer destination layout
		vks::tools::insertImageMemoryBarrier(
			copyCmd,
			dstImage,
			0,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

		// Transition swapchain image from present to transfer source layout
		vks::tools::insertImageMemoryBarrier(
			copyCmd,
			srcImage,
			VK_ACCESS_MEMORY_READ_BIT,
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

		// If source and destination support blit we'll blit as this also does automatic format conversion (e.g. from BGR to RGB)
		if (supportsBlit)
		{
			// Define the region to blit (we will blit the whole swapchain image)
			VkOffset3D blitSize;
			blitSize.x = static_cast<int32_t>(width);
			blitSize.y = static_cast<int32_t>(height);
			blitSize.z = 1;
			VkImageBlit imageBlitRegion{};
			imageBlitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBlitRegion.srcSubresource.layerCount = 1;
			imageBlitRegion.srcOffsets[1] = blitSize;
			imageBlitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBlitRegion.dstSubresource.layerCount = 1;
			imageBlitRegion.dstOffsets[1] = blitSize;

			// Issue the blit command
			vkCmdBlitImage(
				copyCmd,
				srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&imageBlitRegion,
				VK_FILTER_NEAREST);
		}
		else
		{
			// Otherwise use image copy (requires us to manually flip components)
			VkImageCopy imageCopyRegion{};
			imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageCopyRegion.srcSubresource.layerCount = 1;
			imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageCopyRegion.dstSubresource.layerCount = 1;
			imageCopyRegion.extent.width = width;
			imageCopyRegion.extent.height = height;
			imageCopyRegion.extent.depth = 1;

			// Issue the copy command
			vkCmdCopyImage(
				copyCmd,
				srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&imageCopyRegion);
		}

		// Transition destination image to general layout, which is the required layout for mapping the image memory later on
		vks::tools::insertImageMemoryBarrier(
			copyCmd,
			dstImage,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_MEMORY_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

		// Transition back the swap chain image after the blit is done
		vks::tools::insertImageMemoryBarrier(
			copyCmd,
			srcImage,
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_ACCESS_MEMORY_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

		vulkanDevice->flushCommandBuffer(copyCmd, queue);

		// Get layout of the image (including row pitch)
		VkImageSubresource subResource { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
		VkSubresourceLayout subResourceLayout;
		vkGetImageSubresourceLayout(device, dstImage, &subResource, &subResourceLayout);

		// Map image memory so we can start copying from it
		const char* data;
		vkMapMemory(device, dstImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
		data += subResourceLayout.offset;

		std::ofstream file(filename, std::ios::out | std::ios::binary);

		// ppm header
		file << "P6\n" << width << "\n" << height << "\n" << 255 << "\n";

		// If source is BGR (destination is always RGB) and we can't use blit (which does automatic conversion), we'll have to manually swizzle color components
		bool colorSwizzle = false;
		// Check if source is BGR
		// Note: Not complete, only contains most common and basic BGR surface formats for demonstration purposes
		if (!supportsBlit)
		{
			std::vector<VkFormat> formatsBGR = { VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SNORM };
			colorSwizzle = (std::find(formatsBGR.begin(), formatsBGR.end(), swapChain.colorFormat) != formatsBGR.end());
		}

		// ppm binary pixel data
		for (uint32_t y = 0; y < height; y++)
		{
			auto *row = (unsigned int*)data;
			for (uint32_t x = 0; x < width; x++)
			{
				if (colorSwizzle)
				{
					file.write((char*)row+2, 1);
					file.write((char*)row+1, 1);
					file.write((char*)row, 1);
				}
				else
				{
					file.write((char*)row, 3);
				}
				row++;
			}
			data += subResourceLayout.rowPitch;
		}
		file.close();

		LOGI("Screenshot saved to disk");

		// Clean up resources
		vkUnmapMemory(device, dstImageMemory);
		vkFreeMemory(device, dstImageMemory, nullptr);
		vkDestroyImage(device, dstImage, nullptr);

		screenshotSaved = true;
	}

	void prepare() override
	{
		if (settings.headless) {
			// Headless mode: skip swapchain, create offscreen resources
			prepareHeadless();
		} else {
			// Normal windowed mode
			VulkanExampleBase::prepare();
			loadAssets();
			prepareUniformBuffers();
			setupDescriptors();
			preparePipelines();
			prepareVideoEncoding();
			prepared = true;
		}
	}

	// Prepare for headless rendering
	void prepareHeadless()
	{
		LOGI("Preparing headless rendering at %ux%u", width, height);
		
		// Create command pool (normally done in base class but we need to do it here)
		VkCommandPoolCreateInfo cmdPoolInfo = vks::initializers::commandPoolCreateInfo();
		cmdPoolInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool));
		
		// Create command buffers
		createCommandBuffers();
		
		// Setup depth stencil (uses base class)
		setupDepthStencil();
		
		// Setup headless-specific render pass
		setupHeadlessRenderPass();
		
		// Setup offscreen color attachment
		setupHeadlessResources();
		
		// Setup framebuffer
		setupHeadlessFramebuffer();
		
		// Create pipeline cache
		VkPipelineCacheCreateInfo pipelineCacheInfo{};
		pipelineCacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		VK_CHECK_RESULT(vkCreatePipelineCache(device, &pipelineCacheInfo, nullptr, &pipelineCache));
		
		// Create synchronization primitives (fences for command buffer submission)
		VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
		VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &waitFences[0]));
		
		// Load assets and prepare rendering resources
		loadAssets();
		prepareUniformBuffers();
		setupDescriptors();
		preparePipelines();
		
		// Prepare video encoding for headless mode
		prepareVideoEncodingHeadless();
		
		prepared = true;
		
		// Auto-start recording in headless mode
		recordingEnabled = true;
		
		// Run the headless render loop
		renderLoopHeadless();
	}

	// Initialize video encoding for headless mode (single offscreen image instead of swapchain)
	void prepareVideoEncodingHeadless()
	{
		// Check if video encoding is supported
		if (!vulkanDevice->extensionSupported(VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME) ||
		    !vulkanDevice->extensionSupported(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME)) {
			LOGW("H.264 video encoding or maintenance1 not supported, skipping encoder setup");
			return;
		}

		// Align resolution to even numbers
		uint32_t alignedWidth = (width % 2 == 0) ? width : width + 1;
		uint32_t alignedHeight = (height % 2 == 0) ? height : height + 1;

		if (alignedWidth != width || alignedHeight != height) {
			LOGI("Note: Aligning video encoding resources from %ux%u to %ux%u (required for 4:2:0 format)",
			     width, height, alignedWidth, alignedHeight);
		}

		// Initialize H264 encoder
		VulkanH264Encoder::EncoderConfig encoderConfig;
		encoderConfig.width = alignedWidth;
		encoderConfig.height = alignedHeight;
		encoderConfig.gopSize = 60;
		encoderConfig.maxFrameRate = 0;  // No frame rate limit in headless mode
		encoderConfig.qp = 23;
		encoderConfig.outputPath = "recording.h264";
		encoderConfig.useVBR = false;  // CQP for headless

		if (!h264Encoder.initialize(vulkanDevice, instance, encoderConfig)) {
			LOGE("Failed to initialize H264 encoder");
			return;
		}

		if (!h264Encoder.setupProfiles()) {
			LOGE("Failed to setup video profiles");
			return;
		}

		// For headless mode, use a single offscreen image
		std::vector<VkImage> offscreenImages = { offscreen.colorImage };

		if (!rgbToNv12Converter.initialize(vulkanDevice, alignedWidth, alignedHeight, 
				VK_FORMAT_B8G8R8A8_UNORM, offscreenImages, getShadersPath(),
				&h264Encoder.getVideoProfileList())) {
			LOGE("Failed to initialize RGB to NV12 converter");
			return;
		}

		if (!h264Encoder.setupVideoSession()) {
			LOGE("Failed to setup video encode session");
			return;
		}

		// Create command buffer for color conversion
		VkCommandBufferAllocateInfo cmdBufAllocInfo = vks::initializers::commandBufferAllocateInfo(cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocInfo, &colorConvertCmdBuffer));
		
		// Create fence for color conversion synchronization
		VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
		VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &colorConvertFence));
		
		// Store queue family indices
		graphicsQueueFamily = vulkanDevice->queueFamilyIndices.graphics;
		videoQueueFamily = h264Encoder.getVideoQueueFamilyIndex();
		
		LOGI("Headless video encoding pipeline initialized");
		LOGI("  Resolution: %ux%u", alignedWidth, alignedHeight);
		LOGI("  Output: %s", encoderConfig.outputPath.c_str());
	}

	// Cleanup video encoding resources (called before reinitialization on resize)
	void cleanupVideoEncoding()
	{
		if (device == VK_NULL_HANDLE) return;
		
		// Wait for any pending encoding operations to complete
		vkDeviceWaitIdle(device);
		
		// Cleanup converter and encoder resources
		rgbToNv12Converter.cleanup();
		h264Encoder.cleanup();
		
		// Reset recording state
		recordingEnabled = false;
		encodedFrameCount = 0;
		
		LOGI("Video encoding resources cleaned up");
	}
	
	// Initialize video encoding pipeline (RGB to NV12 converter + H264 encoder)
	void prepareVideoEncoding()
	{
		// Check if video encoding is supported
		if (!vulkanDevice->extensionSupported(VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME) ||
		    !vulkanDevice->extensionSupported(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME)) {
			LOGW("H.264 video encoding or maintenance1 not supported, skipping encoder setup");
			return;
		}

        // Align resolution to even numbers (required for YUV420 chroma subsampling)
        // If the window size is odd, we'll use a slightly larger even size for the encoder
        // and pad/clamp the input image (handled by sampler CLAMP_TO_EDGE)
        // 
        // NEW: If we want to support cropping via SPS, we might just pass the original 'width' and 'height'
        // to the encoder config, but still align the storage/encoding NV12 images to be sufficient 
        // for the underlying 16x16 macroblocks (or at least 2x2 chroma blocks).
        // 
        // However, usually we want the ENCODE resolution to be the full padded/aligned resolution 
        // if we are relying on SPS cropping to define the display window. 
        // If we configure the encoder with aligned width/height, the SPS generation code 
        // will see config.width/height as the aligned values and won't crop.
        // 
        // So we should pass the ACTUAL (odd/non-divisible) display width/height to the encoder config 
        // so it can calculate the cropping parameters correctly for the SPS.
        // BUT the underlying Vulkan images MUST still be aligned for 4:2:0 format requirements.
        
        // Let's use the actual window dimensions for the encoder configuration (which drives SPS cropping)
        // But keep using aligned dimensions for the resource creation (images, converter).
        
        uint32_t alignedWidth = (width % 2 == 0) ? width : width + 1;
        uint32_t alignedHeight = (height % 2 == 0) ? height : height + 1;

        if (alignedWidth != width || alignedHeight != height) {
            LOGI("Note: Aligning video encoding resources from %ux%u to %ux%u (required for 4:2:0 format)",
                 width, height, alignedWidth, alignedHeight);
        }

		// Initialize H264 encoder first (to get access to video profiles)
		VulkanH264Encoder::EncoderConfig encoderConfig;
		encoderConfig.width = alignedWidth;
		encoderConfig.height = alignedHeight;
		encoderConfig.gopSize = 360;  // All I-frames
	    encoderConfig.maxFrameRate = 60;
		encoderConfig.qp = 23;
		encoderConfig.outputPath = "recording.h264";
		
		// VBR Configuration
		encoderConfig.useVBR = true;
		encoderConfig.averageBitrate = 1000000; // 1 Mbps
		encoderConfig.maxBitrate = 1000000;     // 1 Mbps

		if (!h264Encoder.initialize(vulkanDevice, instance, encoderConfig)) {
			LOGE("Failed to initialize H264 encoder");
			return;
		}

		// Setup video profiles before creating NV12 images that need the profile list
		if (!h264Encoder.setupProfiles()) {
			LOGE("Failed to setup video profiles");
			return;
		}

		// Initialize RGB to NV12 converter with video profile for VIDEO_ENCODE_SRC usage
		std::vector<VkImage> swapchainImages;
		for (uint32_t i = 0; i < swapChain.imageCount; i++) {
			swapchainImages.push_back(swapChain.images[i]);
		}

		// Use aligned width/height for converter to match encoder expectations
		if (!rgbToNv12Converter.initialize(vulkanDevice, alignedWidth, alignedHeight, 
				swapChain.colorFormat, swapchainImages, getShadersPath(),
				&h264Encoder.getVideoProfileList())) {
			LOGE("Failed to initialize RGB to NV12 converter");
			return;
		}

		// Setup video encode session (query capabilities, create session, DPB, bitstream buffer, etc.)
		if (!h264Encoder.setupVideoSession()) {
			LOGE("Failed to setup video encode session");
			return;
		}

		// Create command buffer for color conversion
		VkCommandBufferAllocateInfo cmdBufAllocInfo = vks::initializers::commandBufferAllocateInfo(cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocInfo, &colorConvertCmdBuffer));
		
		// Create fence for color conversion synchronization
		VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
		VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &colorConvertFence));
		
		// Create semaphores for cross-queue synchronization
		VkSemaphoreCreateInfo semaphoreInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderCompleteSemaphore));
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &colorConvertCompleteSemaphore));
		
		// Store queue family indices for potential ownership transfers
		graphicsQueueFamily = vulkanDevice->queueFamilyIndices.graphics;
		videoQueueFamily = h264Encoder.getVideoQueueFamilyIndex();
		
		bool sameQueueFamily = (graphicsQueueFamily == videoQueueFamily);
		LOGI("Video encoding pipeline initialized successfully");
		LOGI("  Resolution: %ux%u (aligned from %ux%u)",
		     alignedWidth, alignedHeight, width, height);
		LOGI("  Output: %s", encoderConfig.outputPath.c_str());
		LOGI("  Graphics queue family: %u", graphicsQueueFamily);
		LOGI("  Video queue family: %u", videoQueueFamily);
		LOGI("  Cross-queue transfer needed: %s", sameQueueFamily ? "No" : "Yes");
		LOGI("  Press 'R' to start/stop recording");
	}
	
	// Handle window resize by reinitializing video encoder with new dimensions
	void windowResized() override
	{
		// Check if encoder was initialized before resize
		bool wasEncoderInitialized = h264Encoder.isReady();
		
		if (wasEncoderInitialized) {
			LOGI("Window resized to %ux%u, reinitializing video encoder...", width, height);
			
			// Clean up existing encoder resources
			cleanupVideoEncoding();
			
			// Reinitialize video encoding with new dimensions
			// This will generate new SPS/PPS and start from an I-frame
			prepareVideoEncoding();

            // Transition all swapchain images to PRESENT_SRC_KHR layout
            // This ensures that they are in the expected layout for the first frame after resize,
            // avoiding validation errors if descriptors or render passes expect PRESENT_SRC_KHR.
            VkCommandBuffer layoutCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
            for (uint32_t i = 0; i < swapChain.imageCount; i++) {
                vks::tools::insertImageMemoryBarrier(
                    layoutCmd,
                    swapChain.images[i],
                    0,
                    VK_ACCESS_MEMORY_READ_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });
            }
            vulkanDevice->flushCommandBuffer(layoutCmd, queue);
			
			if (h264Encoder.isReady()) {
				LOGI("Video encoder reinitialized successfully at %ux%u", width, height);
			} else {
				LOGE("Failed to reinitialize video encoder after resize");
			}
		}
	}

	void buildCommandBuffer()
	{
		VkCommandBuffer cmdBuffer = drawCmdBuffers[currentBuffer];
				
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2]{};
		clearValues[0].color = defaultClearColor;
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;
		renderPassBeginInfo.framebuffer = frameBuffers[currentImageIndex];

		VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));
		vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
		VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
		vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);
		VkRect2D scissor = vks::initializers::rect2D(static_cast<int32_t>(width), static_cast<int32_t>(height), 0, 0);
		vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);
		vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentBuffer], 0, nullptr);
		vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		model.draw(cmdBuffer);
		drawUI(cmdBuffer);
		vkCmdEndRenderPass(cmdBuffer);
		VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
	}

	void render() override
	{
		if (!prepared)
			return;
		VulkanExampleBase::prepareFrame();
		updateUniformBuffers();
		buildCommandBuffer();
		VulkanExampleBase::submitFrame();
		
	    if (!recordingEnabled)
	        recordingEnabled = true;  // Auto-start recording for demonstration purposes
		// Encode frame if recording is enabled
		if (recordingEnabled && h264Encoder.isReady() && rgbToNv12Converter.isReady()) {
			encodeCurrentFrame();
		}
	}
	
	// Encode the current frame to H.264
	void encodeCurrentFrame()
	{
		// Check frame rate limiter
		const auto& config = h264Encoder.getConfig();
		if (config.maxFrameRate > 0) {
			auto now = std::chrono::steady_clock::now();
			// Skip the first frame check (lastEncodeTime is default initialized)
			if (lastEncodeTime != std::chrono::steady_clock::time_point{}) {
				auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastEncodeTime);
				auto minInterval = std::chrono::milliseconds(1000 / config.maxFrameRate);
				if (elapsed < minInterval) {
					// std::cout << "[Frame] Skipping frame (rate limit: " << elapsed.count() 
					//           << "ms < " << minInterval.count() << "ms)" << std::endl;
					return;
				}
			}
			lastEncodeTime = now;
		}
		
		LOGD("[Frame] Starting encode of frame %lu", (unsigned long)encodedFrameCount);
		
		// Wait for previous color conversion to complete
		LOGD("[Frame] Waiting for color convert fence...");
		vkWaitForFences(device, 1, &colorConvertFence, VK_TRUE, UINT64_MAX);
		LOGD("[Frame] Color convert fence signaled");
		vkResetFences(device, 1, &colorConvertFence);
		
		// Record color conversion commands
		vkResetCommandBuffer(colorConvertCmdBuffer, 0);
		
		VkCommandBufferBeginInfo beginInfo = vks::initializers::commandBufferBeginInfo();
		VK_CHECK_RESULT(vkBeginCommandBuffer(colorConvertCmdBuffer, &beginInfo));
		
		// Dispatch RGB to NV12 conversion
		// Pass queue family info for ownership transfer if needed
		LOGD("[Frame] Recording color conversion commands...");
		rgbToNv12Converter.recordCommands(colorConvertCmdBuffer, currentImageIndex, 
		                                   swapChain.images[currentImageIndex],
		                                   graphicsQueueFamily, videoQueueFamily);
		
		VK_CHECK_RESULT(vkEndCommandBuffer(colorConvertCmdBuffer));
		LOGD("[Frame] Color conversion command buffer recorded");
		
		// Submit color conversion to graphics queue
		// Note: Since we wait on the fence before encoding, we don't need semaphore signaling
		VkSubmitInfo submitInfo = vks::initializers::submitInfo();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &colorConvertCmdBuffer;
		// No semaphore signaling - we use fence-based synchronization
		submitInfo.signalSemaphoreCount = 0;
		submitInfo.pSignalSemaphores = nullptr;
		
		LOGD("[Frame] Submitting color conversion to graphics queue...");
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, colorConvertFence));
		LOGD("[Frame] Color conversion submitted, semaphore will be signaled");
		
		// Wait for color conversion to actually complete before encoding
		// This ensures the data is in the encode image before we try to encode
		LOGD("[Frame] Waiting for color conversion to complete on GPU...");
		vkWaitForFences(device, 1, &colorConvertFence, VK_TRUE, UINT64_MAX);
		LOGD("[Frame] Color conversion complete");
		
		// Get NV12 images for encoding
		const auto& nv12Image = rgbToNv12Converter.getNV12Image(currentImageIndex);
		
		// Encode the frame - don't pass semaphore since we waited for fence
		// Pass graphicsQueueFamily for ownership acquire on video queue
		LOGD("[Frame] Calling encodeFrame...");
		if (h264Encoder.encodeFrame(nv12Image.encodeImage, nv12Image.encodeView, queue,
		                            VK_NULL_HANDLE,  // No semaphore, we waited on fence
		                            graphicsQueueFamily)) {
			encodedFrameCount++;
			LOGD("[Frame] Frame encoded successfully, total: %lu", (unsigned long)encodedFrameCount);
		} else {
			LOGE("[Frame] Frame encoding failed!");
		}
	}

	void OnUpdateUIOverlay(vks::UIOverlay *overlay) override
	{
		if (overlay->header("Functions")) {
            if (overlay->button("Resize Window")) {
                if (width != 1201 || height != 710)
                    setSize(1201, 710); // Odd dimensions to test alignment handling
                else
                    setSize(1280, 720);
            }
			if (overlay->button("Take screenshot")) {
				saveScreenshot("screenshot.ppm");
			}
			if (screenshotSaved) {
				overlay->text("Screenshot saved as screenshot.ppm");
			}
			
			// Video recording controls
			if (h264Encoder.isReady()) {
				if (overlay->button(recordingEnabled ? "Stop Recording" : "Start Recording")) {
					recordingEnabled = !recordingEnabled;
					if (recordingEnabled) {
						LOGI("Recording started...");
					} else {
						LOGI("Recording stopped. Encoded %lu frames.", encodedFrameCount);
					}
				}
                // Force a one-off IDR on next frame
                if (overlay->button("Force I-Frame")) {
                    h264Encoder.requestKeyframe();
                }
				if (recordingEnabled) {
					overlay->text("Recording: %lu frames", encodedFrameCount);
				}
			}
		}
	}
	
	void keyPressed(uint32_t key) override
	{
		VulkanExampleBase::keyPressed(key);
		
		// Toggle recording with 'R' key (key code 82 = 'R')
		if (key == 82 && h264Encoder.isReady()) {
			recordingEnabled = !recordingEnabled;
			if (recordingEnabled) {
				LOGI("Recording started...");
			} else {
				LOGI("Recording stopped. Encoded %lu frames.", encodedFrameCount);
			}
		}
	}

};

VULKAN_EXAMPLE_MAIN()