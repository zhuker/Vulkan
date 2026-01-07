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
#include <thread>
#include <chrono>

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
        std::cout << "RGB to NV12 converter initialized (" << width << "x" << height 
                  << ", swizzle=" << (needsSwizzle ? "yes" : "no") << ")" << std::endl;
        return true;
    }

    // Record compute dispatch commands for color conversion
    // If srcQueueFamily != dstQueueFamily, we need to release ownership to dstQueueFamily (video encode)
    void recordCommands(VkCommandBuffer cmdBuffer, uint32_t imageIndex, VkImage srcImage,
                        uint32_t srcQueueFamily = VK_QUEUE_FAMILY_IGNORED, 
                        uint32_t dstQueueFamily = VK_QUEUE_FAMILY_IGNORED) {
        if (!isInitialized || imageIndex >= nv12Images.size()) return;

        NV12Image& nv12 = nv12Images[imageIndex];
        
        // Determine if cross-queue ownership transfer is needed
        bool needsOwnershipTransfer = (srcQueueFamily != VK_QUEUE_FAMILY_IGNORED) && 
                                       (dstQueueFamily != VK_QUEUE_FAMILY_IGNORED) &&
                                       (srcQueueFamily != dstQueueFamily);
        
        bool hasEncodeImage = (nv12.encodeImage != VK_NULL_HANDLE);

        // Transition source (swapchain) image to SHADER_READ_ONLY_OPTIMAL for sampled read
        VkImageMemoryBarrier srcBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
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
            std::cout << "hasEncodeImage" << std::endl;
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
            VkImageMemoryBarrier2 encodeBarriers[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
                    .dstAccessMask = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR,
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
                    .dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
                    .dstAccessMask = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR,
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

        // Transition swapchain image back to present
        VkImageMemoryBarrier srcPostBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
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
                std::cerr << "Failed to create swapchain image view " << i << std::endl;
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
                std::cerr << "Failed to create Y plane image " << i << ": " << result << std::endl;
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
                std::cerr << "Failed to allocate Y plane memory " << i << std::endl;
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
                std::cerr << "Failed to create UV plane image " << i << ": " << result << std::endl;
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
                std::cerr << "Failed to allocate UV plane memory " << i << std::endl;
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
                std::cerr << "Failed to create Y plane view " << i << std::endl;
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
                std::cerr << "Failed to create UV plane view " << i << std::endl;
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
                    std::cerr << "Failed to create encode NV12 image " << i << ": " << result << std::endl;
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
                    std::cerr << "Failed to allocate encode image memory " << i << std::endl;
                    return false;
                }

                // Bind memory (regular binding for non-disjoint image)
                result = vkBindImageMemory(device, img.encodeImage, img.encodeMemory, 0);
                if (result != VK_SUCCESS) {
                    std::cerr << "Failed to bind encode image memory " << i << ": " << result << std::endl;
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
                    std::cerr << "Failed to create encode image view " << i << ": " << result << std::endl;
                    return false;
                }
            }
        }

        std::cout << "Created " << count << " Y+UV image pairs (" << width << "x" << height << ")"
                  << (videoProfileList ? " with NV12 encode images" : "") << std::endl;
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
            std::cerr << "Failed to create input sampler" << std::endl;
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
            std::cerr << "Failed to create descriptor set layout" << std::endl;
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
            std::cerr << "Failed to create pipeline layout" << std::endl;
            return false;
        }

        // Load compute shader
        std::string shaderFile = shaderPath + "screenshot/rgb_to_nv12.comp.spv";
        std::ifstream file(shaderFile, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open shader file: " << shaderFile << std::endl;
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
            std::cerr << "Failed to create shader module" << std::endl;
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
            std::cerr << "Failed to create compute pipeline" << std::endl;
            return false;
        }

        std::cout << "RGB to NV12 compute pipeline created" << std::endl;
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
            std::cerr << "Failed to create descriptor pool" << std::endl;
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
            std::cerr << "Failed to allocate descriptor sets" << std::endl;
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

        std::cout << "Created " << count << " descriptor sets for RGB to NV12 conversion" << std::endl;
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
        std::string outputPath = "recording.h264";
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
    uint64_t frameCounter = 0;
    uint64_t lastIDRFrame = 0;
    uint16_t idrPicId = 0;
    
    // Reference frame tracking for P-frames
    int32_t lastRefSlotIndex = -1;      // DPB slot index of last reconstructed frame
    uint32_t lastRefFrameNum = 0;       // frame_num of last reference
    int32_t lastRefPicOrderCnt = 0;     // PicOrderCnt of last reference
    StdVideoH264PictureType lastRefPicType = STD_VIDEO_H264_PICTURE_TYPE_IDR;  // Picture type of last reference
    
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
            std::cerr << "Failed to load Vulkan Video extension functions" << std::endl;
            return false;
        }
        
        // Open output file
        outputFile.open(config.outputPath, std::ios::binary | std::ios::trunc);
        if (!outputFile.is_open()) {
            std::cerr << "Failed to open output file: " << config.outputPath << std::endl;
            return false;
        }
        
        isInitialized = true;
        return true;
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
            std::cerr << "Failed to query video capabilities: " << result << std::endl;
            return false;
        }
        
        std::cout << "H.264 Encode Capabilities:" << std::endl;
        std::cout << "  Max coded extent: " << videoCapabilities.maxCodedExtent.width 
                  << "x" << videoCapabilities.maxCodedExtent.height << std::endl;
        std::cout << "  Min coded extent: " << videoCapabilities.minCodedExtent.width 
                  << "x" << videoCapabilities.minCodedExtent.height << std::endl;
        std::cout << "  Max DPB slots: " << videoCapabilities.maxDpbSlots << std::endl;
        std::cout << "  Max active refs: " << videoCapabilities.maxActiveReferencePictures << std::endl;
        std::cout << "  Min bitstream alignment: " << videoCapabilities.minBitstreamBufferSizeAlignment << std::endl;
        std::cout << "  Supported encode feedback flags: 0x" << std::hex << encodeCapabilities.supportedEncodeFeedbackFlags << std::dec << std::endl;
        std::cout << "  Rate control modes: 0x" << std::hex << encodeCapabilities.rateControlModes << std::dec << std::endl;
        std::cout << "  H.264 capabilities:" << std::endl;
        std::cout << "    Max level: " << h264Capabilities.maxLevelIdc << std::endl;
        std::cout << "    Max slice count: " << h264Capabilities.maxSliceCount << std::endl;
        std::cout << "    Max PPicture L0 ref count: " << h264Capabilities.maxPPictureL0ReferenceCount << std::endl;
        std::cout << "    Max BPicture L0 ref count: " << h264Capabilities.maxBPictureL0ReferenceCount << std::endl;
        std::cout << "    Max L1 ref count: " << h264Capabilities.maxL1ReferenceCount << std::endl;
        std::cout << "    Max temporal layer count: " << h264Capabilities.maxTemporalLayerCount << std::endl;
        std::cout << "    Preferred max L0 ref count: " << h264Capabilities.maxQp << std::endl;
        std::cout << "    Flags: 0x" << std::hex << h264Capabilities.flags << std::dec << std::endl;
        
        return true;
    }
    
    // Create the video session
    bool createVideoSession() {
        if (!device || !isInitialized) return false;
        
        // Use the video encode queue family index from the base device
        // This ensures we use the queue family that was actually created
        videoQueueFamilyIndex = vulkanDevice->queueFamilyIndices.videoEncode;
        
        std::cout << "Using video encode queue family index: " << videoQueueFamilyIndex << std::endl;
        
        if (videoQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED || videoQueueFamilyIndex == 0xFFFFFFFF) {
            std::cerr << "No video encode queue family found in device" << std::endl;
            return false;
        }
        
        // Get video queue
        vkGetDeviceQueue(device, videoQueueFamilyIndex, 0, &videoQueue);
        
        if (videoQueue == VK_NULL_HANDLE) {
            std::cerr << "Failed to get video encode queue" << std::endl;
            return false;
        }
        
        std::cout << "Got video encode queue: " << videoQueue << std::endl;
        
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
            std::cerr << "Failed to create video session: " << result << std::endl;
            return false;
        }
        
        // Bind memory to video session
        if (!bindVideoSessionMemory()) {
            return false;
        }
        
        std::cout << "Video session created successfully" << std::endl;
        return true;
    }
    
    // Create session parameters with SPS/PPS
    bool createSessionParameters() {
        if (!videoSession) return false;
        
        // H.264 SPS (Sequence Parameter Set)
        StdVideoH264SequenceParameterSet sps = {};
        sps.flags.constraint_set0_flag = 0;
        sps.flags.constraint_set1_flag = 0;
        sps.flags.constraint_set2_flag = 0;
        sps.flags.constraint_set3_flag = 0;
        sps.flags.constraint_set4_flag = 0;
        sps.flags.constraint_set5_flag = 0;
        sps.flags.direct_8x8_inference_flag = 1;
        sps.flags.frame_mbs_only_flag = 1;
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
        sps.pic_width_in_mbs_minus1 = (config.width + 15) / 16 - 1;
        sps.pic_height_in_map_units_minus1 = (config.height + 15) / 16 - 1;
        
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
            std::cerr << "Failed to create session parameters: " << result << std::endl;
            return false;
        }
        
        std::cout << "Session parameters created successfully" << std::endl;
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
            std::cerr << "Failed to create bitstream buffer: " << result << std::endl;
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
            std::cerr << "Failed to allocate bitstream memory: " << result << std::endl;
            return false;
        }
        
        vkBindBufferMemory(device, bitstreamBuffer, bitstreamMemory, 0);
        vkMapMemory(device, bitstreamMemory, 0, size, 0, &bitstreamMappedPtr);
        
        bitstreamBufferSize = size;
        std::cout << "Bitstream buffer created: " << size << " bytes" << std::endl;
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
            std::cerr << "Failed to create query pool: " << result << std::endl;
            return false;
        }
        
        std::cout << "Query pool created" << std::endl;
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
            std::cerr << "Failed to get SPS/PPS size: " << result << std::endl;
            return false;
        }
        
        if (dataSize == 0) {
            std::cerr << "SPS/PPS data size is 0" << std::endl;
            return false;
        }
        
        // Allocate buffer and retrieve data
        std::vector<uint8_t> paramData(dataSize);
        result = fp_vkGetEncodedVideoSessionParametersKHR(device, &getInfo, &feedback, &dataSize, paramData.data());
        if (result != VK_SUCCESS) {
            std::cerr << "Failed to get SPS/PPS data: " << result << std::endl;
            return false;
        }
        
        // Write to file - data already includes start codes (0x00 0x00 0x00 0x01)
        outputFile.write(reinterpret_cast<const char*>(paramData.data()), dataSize);
        
        std::cout << "Wrote SPS/PPS to file: " << dataSize << " bytes" << std::endl;
        std::cout << "  SPS written: " << (h264Feedback.hasStdSPSOverrides ? "with overrides" : "as-is") << std::endl;
        std::cout << "  PPS written: " << (h264Feedback.hasStdPPSOverrides ? "with overrides" : "as-is") << std::endl;
        
        spsPpsWritten = true;
        return true;
    }
    
    // Check if SPS/PPS has been written
    bool isSpsPpsWritten() const { return spsPpsWritten; }
    
    // Get current frame number
    uint64_t getFrameCount() const { return frameCounter; }
    
    // Check if next frame should be IDR
    bool isNextFrameIDR() const {
        bool result = (frameCounter == 0) || (config.gopSize > 0 && (frameCounter % config.gopSize == 0));
        std::cout << "[GOP] Frame " << frameCounter << ": isNextFrameIDR = " << result 
                  << " (gopSize=" << config.gopSize << ", mod=" << (frameCounter % config.gopSize) << ")" << std::endl;
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
                std::cerr << "Failed to create DPB image " << i << ": " << result << std::endl;
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
                std::cerr << "Failed to allocate DPB image memory " << i << std::endl;
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
                std::cerr << "Failed to create DPB image view " << i << std::endl;
                return false;
            }
            
            slot.inUse = false;
        }
        
        std::cout << "Created " << numDPBSlots << " DPB images" << std::endl;
        return true;
    }
    
    // Full initialization sequence - call after initialize()
    bool setupVideoSession() {
        if (!isInitialized) {
            std::cerr << "Encoder not initialized" << std::endl;
            return false;
        }
        
        // Step 1: Query capabilities
        if (!queryCapabilities()) {
            std::cerr << "Failed to query video capabilities" << std::endl;
            return false;
        }
        
        // Step 2: Create video session
        if (!createVideoSession()) {
            std::cerr << "Failed to create video session" << std::endl;
            return false;
        }
        
        // Step 3: Create session parameters (SPS/PPS)
        if (!createSessionParameters()) {
            std::cerr << "Failed to create session parameters" << std::endl;
            return false;
        }
        
        // Step 4: Create DPB images
        if (!createDPBImages()) {
            std::cerr << "Failed to create DPB images" << std::endl;
            return false;
        }
        
        // Step 5: Create bitstream buffer
        if (!createBitstreamBuffer()) {
            std::cerr << "Failed to create bitstream buffer" << std::endl;
            return false;
        }
        
        // Step 6: Create query pool
        if (!createQueryPool()) {
            std::cerr << "Failed to create query pool" << std::endl;
            return false;
        }
        
        // Step 7: Create command pool for video queue
        if (!createVideoCommandPool()) {
            std::cerr << "Failed to create video command pool" << std::endl;
            return false;
        }
        
        // Step 8: Create encode command buffer and sync objects
        if (!createEncodeSyncObjects()) {
            std::cerr << "Failed to create encode sync objects" << std::endl;
            return false;
        }
        
        std::cout << "Video encode session fully initialized" << std::endl;
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
        
        std::cout << "[Encode] Starting frame " << frameCounter << std::endl;
        
        // Wait for previous encode to complete
        std::cout << "[Encode] Waiting for previous encode fence..." << std::endl;
        vkWaitForFences(device, 1, &encodeFence, VK_TRUE, UINT64_MAX);
        std::cout << "[Encode] Previous fence signaled, resetting..." << std::endl;
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
        std::cout << "[Encode] Recording encode commands..." << std::endl;
        recordEncodeCommands(encodeCommandBuffer, srcImage, srcView, srcQueueFamily);
        
        VK_CHECK_RESULT(vkEndCommandBuffer(encodeCommandBuffer));
        std::cout << "[Encode] Command buffer recorded" << std::endl;
        
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
        
        std::cout << "[Encode] Submitting to video queue (waitSemaphore=" 
                  << (waitSemaphore != VK_NULL_HANDLE ? "yes" : "no") << ")..." << std::endl;
        VkResult result = vkQueueSubmit(videoQueue, 1, &submitInfo, encodeFence);
        if (result != VK_SUCCESS) {
            std::cerr << "Failed to submit encode command buffer: " << result << std::endl;
            return false;
        }
        std::cout << "[Encode] Submitted successfully" << std::endl;
        
        // Wait for encode to complete and read back results
        std::cout << "[Encode] Waiting for encode fence..." << std::endl;
        result = vkWaitForFences(device, 1, &encodeFence, VK_TRUE, UINT64_MAX);
        std::cout << "[Encode] Encode fence signaled (result=" << result << ")" << std::endl;
        
        // Also wait on the queue to ensure all work is done
        std::cout << "[Encode] Waiting for video queue idle..." << std::endl;
        result = vkQueueWaitIdle(videoQueue);
        std::cout << "[Encode] Video queue idle (result=" << result << ")" << std::endl;
        
        // Query encode results
        // Try reading raw bytes first to see what the driver actually writes
        uint8_t rawData[64] = {0};
        
        // First try with just the status bit to see if query is available at all
        result = vkGetQueryPoolResults(device, queryPool, 0, 1, sizeof(rawData), rawData,
            sizeof(rawData), VK_QUERY_RESULT_WITH_STATUS_BIT_KHR);
        
        std::cout << "[Encode] Raw query result: " << result << std::endl;
        std::cout << "[Encode] Raw bytes: ";
        for (int i = 0; i < 32; i++) {
            printf("%02x ", rawData[i]);
        }
        std::cout << std::endl;
        
        if (result == VK_NOT_READY) {
            // Try with 64-bit flag
            result = vkGetQueryPoolResults(device, queryPool, 0, 1, sizeof(rawData), rawData,
                sizeof(rawData), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_STATUS_BIT_KHR);
            std::cout << "[Encode] With 64-bit: result=" << result << std::endl;
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
        std::cout << "[Encode] As 32-bit: offset=" << fb32->offset << ", bytes=" << fb32->bytesWritten 
                  << ", status=" << fb32->status << std::endl;
        
        // Also try 64-bit interpretation
        struct EncodeFeedback64 {
            uint64_t offset;
            uint64_t bytesWritten;
            int64_t status;
        };
        EncodeFeedback64* fb64 = reinterpret_cast<EncodeFeedback64*>(rawData);
        std::cout << "[Encode] As 64-bit: offset=" << fb64->offset << ", bytes=" << fb64->bytesWritten 
                  << ", status=" << fb64->status << std::endl;
        
        // If query not ready, something went wrong with the encode
        if (result == VK_NOT_READY) {
            std::cerr << "[Encode] Query not ready after GPU completed - encode may have been skipped" << std::endl;
            // Check if maybe the status field has useful info
            std::cerr << "[Encode] Status from raw data: 32-bit=" << fb32->status 
                      << ", 64-bit=" << fb64->status << std::endl;
            return false;
        }
        
        // Check status from 32-bit interpretation (without VK_QUERY_RESULT_64_BIT)
        if (fb32->status != VK_QUERY_RESULT_STATUS_COMPLETE_KHR) {
            std::cerr << "[Encode] Encode status not complete: " << fb32->status 
                      << " (COMPLETE=" << VK_QUERY_RESULT_STATUS_COMPLETE_KHR 
                      << ", ERROR=" << VK_QUERY_RESULT_STATUS_ERROR_KHR << ")" << std::endl;
            // Still continue to see what data we got
        }
        
        // Write encoded data to file using 32-bit values
        if (result == VK_SUCCESS && fb32->bytesWritten > 0 && bitstreamMappedPtr) {
            // For IDR frames, ensure SPS/PPS is written first
            // This makes the stream self-contained and decodable from any IDR
            bool isIDR = (frameCounter == 0) || (config.gopSize > 0 && (frameCounter % config.gopSize == 0));
            if (isIDR && !spsPpsWritten) {
                writeSpsPps();
            }
            
            std::cout << "[Encode] Writing " << fb32->bytesWritten << " bytes at offset " << fb32->offset;
            if (isIDR) std::cout << " (IDR frame)";
            std::cout << std::endl;
            
            const uint8_t* data = static_cast<const uint8_t*>(bitstreamMappedPtr) + fb32->offset;
            writeNALUnit(data, static_cast<size_t>(fb32->bytesWritten));
        }
        
        frameCounter++;
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
        
        isInitialized = false;
    }

private:
    // Helper to determine if current frame should be encoded as P-frame
    bool isPFrame() const {
        bool notIDR = !isNextFrameIDR();
        bool result = config.gopSize > 1 && notIDR && frameCounter > 0;
        std::cout << "[GOP] Frame " << frameCounter << ": isPFrame = " << result 
                  << " (gopSize>1: " << (config.gopSize > 1) << ", notIDR: " << notIDR 
                  << ", frameCounter>0: " << (frameCounter > 0) 
                  << ", lastRefSlot: " << lastRefSlotIndex << ")" << std::endl;
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
                std::cerr << "Failed to allocate video session memory: " << result << std::endl;
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
            std::cerr << "Failed to bind video session memory: " << result << std::endl;
            return false;
        }
        
        std::cout << "Video session memory bound (" << memReqCount << " allocations)" << std::endl;
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
            std::cerr << "Failed to create video command pool: " << result << std::endl;
            return false;
        }
        
        std::cout << "Video command pool created" << std::endl;
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
            std::cerr << "Failed to allocate encode command buffer: " << result << std::endl;
            return false;
        }
        
        // Create fence
        VkFenceCreateInfo fenceInfo = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,  // Start signaled so first wait doesn't block
        };
        
        result = vkCreateFence(device, &fenceInfo, nullptr, &encodeFence);
        if (result != VK_SUCCESS) {
            std::cerr << "Failed to create encode fence: " << result << std::endl;
            return false;
        }
        
        // Create semaphore
        VkSemaphoreCreateInfo semaphoreInfo = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };
        
        result = vkCreateSemaphore(device, &semaphoreInfo, nullptr, &encodeSemaphore);
        if (result != VK_SUCCESS) {
            std::cerr << "Failed to create encode semaphore: " << result << std::endl;
            return false;
        }
        
        std::cout << "Encode sync objects created" << std::endl;
        return true;
    }
    
    // Record video encode commands into command buffer
    // srcImage: the NV12 multi-planar source image
    // srcView: view of the NV12 image
    // srcQueueFamily: the queue family that released ownership (compute/graphics)
    // If srcQueueFamily differs from video queue family, we need to acquire ownership
    void recordEncodeCommands(VkCommandBuffer cmdBuffer, VkImage srcImage, VkImageView srcView,
                              uint32_t srcQueueFamily = VK_QUEUE_FAMILY_IGNORED) {
        std::cout << "\n========== ENCODE FRAME " << frameCounter << " ==========" << std::endl;
        std::cout << "[GOP] Config: gopSize=" << config.gopSize << ", qp=" << config.qp << std::endl;
        std::cout << "[GOP] State: lastRefSlot=" << lastRefSlotIndex 
                  << ", lastRefFrameNum=" << lastRefFrameNum 
                  << ", lastRefPOC=" << lastRefPicOrderCnt << std::endl;
        
        bool isIDR = isNextFrameIDR();
        bool isP = isPFrame();
        
        std::cout << "[GOP] Frame type determined: IDR=" << isIDR << ", P=" << isP 
                  << " => " << (isIDR ? "IDR" : (isP ? "P-frame" : "I-frame")) << std::endl;
        
        // Determine DPB slot for current reconstructed frame
        // For P-frames, use ping-pong between slot 0 and 1
        int32_t slotIndex = (config.gopSize > 1) ? static_cast<int32_t>(frameCounter % 2) : 0;
        std::cout << "[GOP] Using DPB slot: " << slotIndex << std::endl;
        
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
            std::cout << "[GOP] Adding reference DPB barriers for slot " << lastRefSlotIndex << std::endl;
            
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
        
        std::cout << "[GOP] Slice type: " << (int)sliceHeader.slice_type << "(" << (isP ? "P" : "I") << "-slice)" << std::endl;        
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
            std::cout << "[GOP] Building reference list: L0[0]=" << (int)refLists.RefPicList0[0] << ", num_ref_l0=1" << std::endl;
        } else {
            // No references used; keep lists filled with NO_REFERENCE and set counts to 0
            refLists.num_ref_idx_l0_active_minus1 = 0;
            refLists.num_ref_idx_l1_active_minus1 = 0;
            std::cout << "[GOP] No reference list (I-frame or no previous ref)" << std::endl;
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
        // For IDR frames, frame_num should be 0 since IDR resets the DPB
        stdPicInfo.frame_num = isIDR ? 0 : static_cast<uint32_t>(frameCounter % 256);
        // For POC type 0, PicOrderCnt should wrap around at max_pic_order_cnt_lsb (256)
        // For IDR frames, POC starts at 0
        stdPicInfo.PicOrderCnt = isIDR ? 0 : static_cast<int32_t>((frameCounter * 2) % 256);
        if (isIDR) {
            // if stdPicInfo.frame_num is not set correctly ffplay ignores or skips frames randly 
            frameCounter = 0;  // Reset frame counter after IDR for consistent POC
        }
        stdPicInfo.temporal_id = 0;
        stdPicInfo.pRefLists = (isP && lastRefSlotIndex >= 0) ? &refLists : nullptr;
        
        std::cout << "[GOP] Picture info: IdrFlag=" << (int)stdPicInfo.flags.IdrPicFlag << ", primary_pic_type=" << (int)stdPicInfo.primary_pic_type                   << ", frame_num=" << stdPicInfo.frame_num
                          << ", POC=" << stdPicInfo.PicOrderCnt                  << ", pRefLists=" << (stdPicInfo.pRefLists ? "SET" : "NULL") << std::endl;
        
        // H.264 NALU slice info
        VkVideoEncodeH264NaluSliceInfoKHR sliceInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_NALU_SLICE_INFO_KHR,
            .pNext = nullptr,
            .constantQp = static_cast<int32_t>(config.qp),
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
        stdRefInfo.FrameNum = isIDR ? 0 : static_cast<uint32_t>(frameCounter % 256);
        stdRefInfo.PicOrderCnt = isIDR ? 0 : static_cast<int32_t>((frameCounter * 2) % 256);
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
            std::cout << "[GOP] Setting up input reference from DPB slot " << lastRefSlotIndex << std::endl;            // Get the reference DPB slot
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
            
            std::cout << "[GOP] Input ref: FrameNum=" << refStdInfo.FrameNum 
                      << ", POC=" << refStdInfo.PicOrderCnt 
                      << ", picType=" << (int)refStdInfo.primary_pic_type << std::endl;            
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
            std::cout << "[GOP] No input reference (isP=" << isP << ", lastRefSlot=" << lastRefSlotIndex << ")" <<
                std::endl;
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
            std::cout << "[GOP] Begin coding: including input ref slot " << lastRefSlotIndex << std::endl;
        }
        
        // Always include a placeholder for tracking (will be assigned during encode)
        VkVideoReferenceSlotInfoKHR placeholderSlot = setupSlot;
        placeholderSlot.slotIndex = -1;  // Mark as not yet assigned
        beginSlots[beginSlotCount++] = placeholderSlot;
        
        // Reset query pool before beginning video coding (must be outside video coding scope)
        vkCmdResetQueryPool(cmdBuffer, queryPool, 0, 1);
        
        // Rate control info for DISABLED mode (constant QP)
        VkVideoEncodeRateControlInfoKHR rateControlInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_INFO_KHR,
            .pNext = nullptr,
            .flags = 0,
            .rateControlMode = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR,
            .layerCount = 0,
            .pLayers = nullptr,
            .virtualBufferSizeInMs = 0,
            .initialVirtualBufferSizeInMs = 0,
        };
        
        // Begin video coding
        // - First frame: Don't include rate control (it's still DEFAULT)
        // - Subsequent frames: Include rate control matching current state (DISABLED)
        VkVideoBeginCodingInfoKHR beginInfo = {
            .sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
            .pNext = sessionReset ? &rateControlInfo : nullptr,  // Only after rate control is configured
            .flags = 0,
            .videoSession = videoSession,
            .videoSessionParameters = sessionParams,
            .referenceSlotCount = beginSlotCount,
            .pReferenceSlots = beginSlots.data(),
        };
        
        std::cout << "[GOP] Begin coding with " << beginSlotCount << " reference slot(s)" << std::endl;
        
        fp_vkCmdBeginVideoCodingKHR(cmdBuffer, &beginInfo);
        
        // Reset session and configure rate control on first frame
        if (!sessionReset) {
            // First reset the session
            VkVideoCodingControlInfoKHR controlInfo = {
                .sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
                .pNext = nullptr,
                .flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR,
            };
            fp_vkCmdControlVideoCodingKHR(cmdBuffer, &controlInfo);
            
            // Then set rate control mode to DISABLED for constant QP encoding
            VkVideoCodingControlInfoKHR rateControlCommand = {
                .sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
                .pNext = &rateControlInfo,
                .flags = VK_VIDEO_CODING_CONTROL_ENCODE_RATE_CONTROL_BIT_KHR,
            };
            fp_vkCmdControlVideoCodingKHR(cmdBuffer, &rateControlCommand);
            
            sessionReset = true;
        }
        
        // Begin query - use index 0
        // Note: For video encode feedback queries, we use the query within the video coding scope
        std::cout << "[Encode] Beginning query..." << std::endl;
        vkCmdBeginQuery(cmdBuffer, queryPool, 0, 0);
        
        // Encode
        std::cout << "[Encode] Recording vkCmdEncodeVideoKHR..." << std::endl;
        fp_vkCmdEncodeVideoKHR(cmdBuffer, &encodeInfo);
        std::cout << "[Encode] Encode command recorded" << std::endl;
        
        // End query
        vkCmdEndQuery(cmdBuffer, queryPool, 0);
        std::cout << "[Encode] Query ended" << std::endl;
        
        // Update reference tracking for next P-frame
        // Store current frame as reference for next frame
        lastRefSlotIndex = slotIndex;
        lastRefFrameNum = stdPicInfo.frame_num;
        lastRefPicOrderCnt = stdPicInfo.PicOrderCnt;
        lastRefPicType = stdPicInfo.primary_pic_type;

        std::cout << "[GOP] Updated reference tracking: slot=" << lastRefSlotIndex << ", frameNum=" << lastRefFrameNum
            << ", POC=" << lastRefPicOrderCnt << ", picType=" << (int)lastRefPicType << std::endl;
        std::cout << "========== END ENCODE FRAME " << frameCounter << " ==========" << std::endl;
        
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

	    std::cout << "Vulkan Video Encoding Support:" << std::endl;
	    std::cout << "  Base Video Queue: " << (videoQueue ? "Yes" : "No") << std::endl;
	    std::cout << "  Encode Queue:     " << (encodeQueue ? "Yes" : "No") << std::endl;
		std::cout << "  Synchronization2: " << (sync2 ? "Yes" : "No") << std::endl;
		std::cout << "  Maintenance1:     " << (maintenance1 ? "Yes" : "No") << std::endl;

	    // Check for specific encoders
        if (videoQueue && encodeQueue && sync2 && maintenance1) {
            enabledDeviceExtensions.push_back(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);
            enabledDeviceExtensions.push_back(VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME);
            enabledDeviceExtensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
            enabledDeviceExtensions.push_back(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME);

            std::cout << "  Supported Encoders:" << std::endl;
            bool h264 = vulkanDevice->extensionSupported(VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME);
            bool h265 = vulkanDevice->extensionSupported(VK_KHR_VIDEO_ENCODE_H265_EXTENSION_NAME);
            bool av1 = vulkanDevice->extensionSupported(VK_KHR_VIDEO_ENCODE_AV1_EXTENSION_NAME);
            std::cout << "    H.264: " << (h264 ? "Yes" : "No") << std::endl;
            std::cout << "    H.265: " << (h265 ? "Yes" : "No") << std::endl;
            std::cout << "    AV1:   " << (av1 ? "Yes" : "No") << std::endl;

            if (h264) {
                enabledDeviceExtensions.push_back(VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME);
            }
            if (h265) {
                enabledDeviceExtensions.push_back(VK_KHR_VIDEO_ENCODE_H265_EXTENSION_NAME);
            }
            if (av1) {
                enabledDeviceExtensions.push_back(VK_KHR_VIDEO_ENCODE_AV1_EXTENSION_NAME);
            }
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
			std::cerr << "Device does not support blitting from optimal tiled images, using copy instead of blit!" << std::endl;
			supportsBlit = false;
		}

		// Check if the device supports blitting to linear images
		vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, &formatProps);
		if (!(formatProps.linearTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
			std::cerr << "Device does not support blitting to linear tiled images, using copy instead of blit!" << std::endl;
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

		std::cout << "Screenshot saved to disk" << std::endl;

		// Clean up resources
		vkUnmapMemory(device, dstImageMemory);
		vkFreeMemory(device, dstImageMemory, nullptr);
		vkDestroyImage(device, dstImage, nullptr);

		screenshotSaved = true;
	}

	void prepare() override
	{
		VulkanExampleBase::prepare();
		loadAssets();
		prepareUniformBuffers();
		setupDescriptors();
		preparePipelines();
		prepareVideoEncoding();
		prepared = true;
	}

	// Initialize video encoding pipeline (RGB to NV12 converter + H264 encoder)
	void prepareVideoEncoding()
	{
		// Check if video encoding is supported
		if (!vulkanDevice->extensionSupported(VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME) ||
		    !vulkanDevice->extensionSupported(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME)) {
			std::cout << "H.264 video encoding or maintenance1 not supported, skipping encoder setup" << std::endl;
			return;
		}

		// Initialize H264 encoder first (to get access to video profiles)
		VulkanH264Encoder::EncoderConfig encoderConfig;
		encoderConfig.width = width;
		encoderConfig.height = height;
		encoderConfig.gopSize = 80;  // All I-frames
		encoderConfig.qp = 23;
		encoderConfig.outputPath = "recording.h264";

		if (!h264Encoder.initialize(vulkanDevice, instance, encoderConfig)) {
			std::cerr << "Failed to initialize H264 encoder" << std::endl;
			return;
		}

		// Setup video profiles before creating NV12 images that need the profile list
		if (!h264Encoder.setupProfiles()) {
			std::cerr << "Failed to setup video profiles" << std::endl;
			return;
		}

		// Initialize RGB to NV12 converter with video profile for VIDEO_ENCODE_SRC usage
		std::vector<VkImage> swapchainImages;
		for (uint32_t i = 0; i < swapChain.imageCount; i++) {
			swapchainImages.push_back(swapChain.images[i]);
		}

		if (!rgbToNv12Converter.initialize(vulkanDevice, width, height, 
				swapChain.colorFormat, swapchainImages, getShadersPath(),
				&h264Encoder.getVideoProfileList())) {
			std::cerr << "Failed to initialize RGB to NV12 converter" << std::endl;
			return;
		}

		// Setup video encode session (query capabilities, create session, DPB, bitstream buffer, etc.)
		if (!h264Encoder.setupVideoSession()) {
			std::cerr << "Failed to setup video encode session" << std::endl;
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
		std::cout << "Video encoding pipeline initialized successfully" << std::endl;
		std::cout << "  Resolution: " << width << "x" << height << std::endl;
		std::cout << "  Output: " << encoderConfig.outputPath << std::endl;
		std::cout << "  Graphics queue family: " << graphicsQueueFamily << std::endl;
		std::cout << "  Video queue family: " << videoQueueFamily << std::endl;
		std::cout << "  Cross-queue transfer needed: " << (sameQueueFamily ? "No" : "Yes") << std::endl;
		std::cout << "  Press 'R' to start/stop recording" << std::endl;
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
		std::cout << "[Frame] Starting encode of frame " << encodedFrameCount << std::endl;
		
		// Wait for previous color conversion to complete
		std::cout << "[Frame] Waiting for color convert fence..." << std::endl;
		vkWaitForFences(device, 1, &colorConvertFence, VK_TRUE, UINT64_MAX);
		std::cout << "[Frame] Color convert fence signaled" << std::endl;
		vkResetFences(device, 1, &colorConvertFence);
		
		// Record color conversion commands
		vkResetCommandBuffer(colorConvertCmdBuffer, 0);
		
		VkCommandBufferBeginInfo beginInfo = vks::initializers::commandBufferBeginInfo();
		VK_CHECK_RESULT(vkBeginCommandBuffer(colorConvertCmdBuffer, &beginInfo));
		
		// Dispatch RGB to NV12 conversion
		// Pass queue family info for ownership transfer if needed
		std::cout << "[Frame] Recording color conversion commands..." << std::endl;
		rgbToNv12Converter.recordCommands(colorConvertCmdBuffer, currentImageIndex, 
		                                   swapChain.images[currentImageIndex],
		                                   graphicsQueueFamily, videoQueueFamily);
		
		VK_CHECK_RESULT(vkEndCommandBuffer(colorConvertCmdBuffer));
		std::cout << "[Frame] Color conversion command buffer recorded" << std::endl;
		
		// Submit color conversion to graphics queue
		// Note: Since we wait on the fence before encoding, we don't need semaphore signaling
		VkSubmitInfo submitInfo = vks::initializers::submitInfo();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &colorConvertCmdBuffer;
		// No semaphore signaling - we use fence-based synchronization
		submitInfo.signalSemaphoreCount = 0;
		submitInfo.pSignalSemaphores = nullptr;
		
		std::cout << "[Frame] Submitting color conversion to graphics queue..." << std::endl;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, colorConvertFence));
		std::cout << "[Frame] Color conversion submitted, semaphore will be signaled" << std::endl;
		
		// Wait for color conversion to actually complete before encoding
		// This ensures the data is in the encode image before we try to encode
		std::cout << "[Frame] Waiting for color conversion to complete on GPU..." << std::endl;
		vkWaitForFences(device, 1, &colorConvertFence, VK_TRUE, UINT64_MAX);
		std::cout << "[Frame] Color conversion complete" << std::endl;
		
		// Get NV12 images for encoding
		const auto& nv12Image = rgbToNv12Converter.getNV12Image(currentImageIndex);
		
		// Encode the frame - don't pass semaphore since we waited for fence
		// Pass graphicsQueueFamily for ownership acquire on video queue
		std::cout << "[Frame] Calling encodeFrame..." << std::endl;
		if (h264Encoder.encodeFrame(nv12Image.encodeImage, nv12Image.encodeView, queue,
		                            VK_NULL_HANDLE,  // No semaphore, we waited on fence
		                            graphicsQueueFamily)) {
			encodedFrameCount++;
			std::cout << "[Frame] Frame encoded successfully, total: " << encodedFrameCount << std::endl;
		} else {
			std::cerr << "[Frame] Frame encoding failed!" << std::endl;
		}
	}

	void OnUpdateUIOverlay(vks::UIOverlay *overlay) override
	{
		if (overlay->header("Functions")) {
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
						std::cout << "Recording started..." << std::endl;
					} else {
						std::cout << "Recording stopped. Encoded " << encodedFrameCount << " frames." << std::endl;
					}
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
				std::cout << "Recording started..." << std::endl;
			} else {
				std::cout << "Recording stopped. Encoded " << encodedFrameCount << " frames." << std::endl;
			}
		}
	}

};

VULKAN_EXAMPLE_MAIN()