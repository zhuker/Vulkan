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

// RGB to NV12 Color Conversion Pipeline
// Uses a compute shader to convert RGB swapchain images to NV12 format for video encoding
class RGBtoNV12Converter {
public:
    struct NV12Image {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView viewY = VK_NULL_HANDLE;      // Y plane view (full resolution)
        VkImageView viewUV = VK_NULL_HANDLE;     // UV plane view (half resolution)
        VkImageView viewFull = VK_NULL_HANDLE;   // Full image view for video encode
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

    // Push constant for shader
    struct PushConstants {
        int32_t swizzle;
    };

public:
    RGBtoNV12Converter() = default;

    ~RGBtoNV12Converter() {
        cleanup();
    }

    // Initialize the converter
    bool initialize(vks::VulkanDevice* vulkanDevice, uint32_t width, uint32_t height, 
                   VkFormat swapchainFormat, const std::vector<VkImage>& swapchainImages,
                   const std::string& shaderPath) {
        this->vulkanDevice = vulkanDevice;
        this->device = vulkanDevice->logicalDevice;
        this->physicalDevice = vulkanDevice->physicalDevice;
        this->width = width;
        this->height = height;

        // Check if swapchain format is BGR and needs swizzle
        std::vector<VkFormat> formatsBGR = { 
            VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SNORM 
        };
        needsSwizzle = (std::find(formatsBGR.begin(), formatsBGR.end(), swapchainFormat) != formatsBGR.end());

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
    void recordCommands(VkCommandBuffer cmdBuffer, uint32_t imageIndex, VkImage srcImage) {
        if (!isInitialized || imageIndex >= nv12Images.size()) return;

        NV12Image& nv12 = nv12Images[imageIndex];

        // Transition source (swapchain) image to GENERAL for compute read
        VkImageMemoryBarrier srcBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = srcImage,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
        };

        // Transition NV12 image to GENERAL for compute write
        VkImageMemoryBarrier dstBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = nv12.image,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
        };

        VkImageMemoryBarrier barriers[] = { srcBarrier, dstBarrier };
        vkCmdPipelineBarrier(cmdBuffer,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 2, barriers);

        // Bind compute pipeline and descriptor set
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, 
            pipelineLayout, 0, 1, &descriptorSets[imageIndex], 0, nullptr);

        // Push swizzle constant
        PushConstants pushConstants = { needsSwizzle ? 1 : 0 };
        vkCmdPushConstants(cmdBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 
            0, sizeof(PushConstants), &pushConstants);

        // Dispatch compute shader (16x16 workgroups)
        uint32_t groupCountX = (width + 15) / 16;
        uint32_t groupCountY = (height + 15) / 16;
        vkCmdDispatch(cmdBuffer, groupCountX, groupCountY, 1);

        // Barrier: compute write -> video encode read
        VkImageMemoryBarrier postBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = nv12.image,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
        };

        vkCmdPipelineBarrier(cmdBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0, 0, nullptr, 0, nullptr, 1, &postBarrier);

        // Transition swapchain image back to present
        VkImageMemoryBarrier srcPostBarrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = srcImage,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
        };

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

        // Destroy NV12 images
        for (auto& img : nv12Images) {
            if (img.viewFull != VK_NULL_HANDLE) {
                vkDestroyImageView(device, img.viewFull, nullptr);
            }
            if (img.viewUV != VK_NULL_HANDLE) {
                vkDestroyImageView(device, img.viewUV, nullptr);
            }
            if (img.viewY != VK_NULL_HANDLE) {
                vkDestroyImageView(device, img.viewY, nullptr);
            }
            if (img.image != VK_NULL_HANDLE) {
                vkDestroyImage(device, img.image, nullptr);
            }
            if (img.memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, img.memory, nullptr);
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

    // Create NV12 format images for color-converted output
    bool createNV12Images(uint32_t count) {
        nv12Images.resize(count);

        for (uint32_t i = 0; i < count; i++) {
            NV12Image& img = nv12Images[i];
            img.width = width;
            img.height = height;

            // Create NV12 image (2-plane YUV 4:2:0)
            VkImageCreateInfo imageInfo = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                .extent = { width, height, 1 },
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            };

            VkResult result = vkCreateImage(device, &imageInfo, nullptr, &img.image);
            if (result != VK_SUCCESS) {
                std::cerr << "Failed to create NV12 image " << i << ": " << result << std::endl;
                return false;
            }

            // Allocate memory
            VkMemoryRequirements memReqs;
            vkGetImageMemoryRequirements(device, img.image, &memReqs);

            VkMemoryAllocateInfo allocInfo = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = memReqs.size,
                .memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, 
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
            };

            result = vkAllocateMemory(device, &allocInfo, nullptr, &img.memory);
            if (result != VK_SUCCESS) {
                std::cerr << "Failed to allocate NV12 image memory " << i << std::endl;
                return false;
            }

            vkBindImageMemory(device, img.image, img.memory, 0);

            // Create Y plane view (plane 0)
            VkImageViewCreateInfo yViewInfo = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = img.image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = VK_FORMAT_R8_UNORM,
                .subresourceRange = { VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 1, 0, 1 }
            };

            result = vkCreateImageView(device, &yViewInfo, nullptr, &img.viewY);
            if (result != VK_SUCCESS) {
                std::cerr << "Failed to create Y plane view " << i << std::endl;
                return false;
            }

            // Create UV plane view (plane 1) - half resolution
            VkImageViewCreateInfo uvViewInfo = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = img.image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = VK_FORMAT_R8G8_UNORM,
                .subresourceRange = { VK_IMAGE_ASPECT_PLANE_1_BIT, 0, 1, 0, 1 }
            };

            result = vkCreateImageView(device, &uvViewInfo, nullptr, &img.viewUV);
            if (result != VK_SUCCESS) {
                std::cerr << "Failed to create UV plane view " << i << std::endl;
                return false;
            }

            // Create full image view for video encode (color aspect)
            VkImageViewCreateInfo fullViewInfo = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = img.image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
            };

            result = vkCreateImageView(device, &fullViewInfo, nullptr, &img.viewFull);
            if (result != VK_SUCCESS) {
                std::cerr << "Failed to create full NV12 view " << i << std::endl;
                return false;
            }
        }

        std::cout << "Created " << count << " NV12 images (" << width << "x" << height << ")" << std::endl;
        return true;
    }

    // Create compute pipeline for RGB to NV12 conversion
    bool createComputePipeline(const std::string& shaderPath) {
        // Descriptor set layout: binding 0 = input image, binding 1 = Y output, binding 2 = UV output
        std::array<VkDescriptorSetLayoutBinding, 3> bindings = {{
            {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
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

        VkResult result = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
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
        // Descriptor pool
        VkDescriptorPoolSize poolSize = {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = count * 3,  // 3 images per set
        };

        VkDescriptorPoolCreateInfo poolInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = count,
            .poolSizeCount = 1,
            .pPoolSizes = &poolSize,
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
                .imageView = swapchainImageViews[i],
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
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
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
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
        uint32_t frameRate = 60;
        uint32_t gopSize = 1;        // All I-frames for simplicity
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
    
    // Configuration
    EncoderConfig config{};
    
    // Output file
    std::ofstream outputFile;
    bool isInitialized = false;
    
    // Video queue family
    uint32_t videoQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    VkQueue videoQueue = VK_NULL_HANDLE;

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

        if (!fp_vkCreateVideoSessionKHR || !fp_vkDestroyVideoSessionKHR || !fp_vkGetVideoSessionMemoryRequirementsKHR ||
            !fp_vkBindVideoSessionMemoryKHR || !fp_vkCreateVideoSessionParametersKHR || !fp_vkDestroyVideoSessionParametersKHR ||
            !fp_vkGetPhysicalDeviceVideoCapabilitiesKHR) {
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
    
    // Query video encode capabilities
    bool queryCapabilities() {
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
        
        return true;
    }
    
    // Create the video session
    bool createVideoSession() {
        if (!device || !isInitialized) return false;
        
        // Find video encode queue family
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
        
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR) {
                videoQueueFamilyIndex = i;
                break;
            }
        }
        
        if (videoQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED) {
            std::cerr << "No video encode queue family found" << std::endl;
            return false;
        }
        
        // Get video queue
        vkGetDeviceQueue(device, videoQueueFamilyIndex, 0, &videoQueue);
        
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
        sps.pic_order_cnt_type = STD_VIDEO_H264_POC_TYPE_2;
        sps.log2_max_pic_order_cnt_lsb_minus4 = 0;
        sps.max_num_ref_frames = 1;
        sps.pic_width_in_mbs_minus1 = (config.width + 15) / 16 - 1;
        sps.pic_height_in_map_units_minus1 = (config.height + 15) / 16 - 1;
        
        // H.264 PPS (Picture Parameter Set)
        StdVideoH264PictureParameterSet pps = {};
        pps.flags.entropy_coding_mode_flag = 1;  // CABAC
        // pps.flags.pic_order_present_flag = 0;
        pps.flags.weighted_pred_flag = 0;
        pps.flags.deblocking_filter_control_present_flag = 1;
        pps.flags.constrained_intra_pred_flag = 0;
        pps.flags.redundant_pic_cnt_present_flag = 0;
        pps.flags.transform_8x8_mode_flag = 0;
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
        VkQueryPoolCreateInfo queryPoolInfo = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = &videoProfileList,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR,
            .queryCount = 2,  // Double-buffering
        };
        
        VkQueryPoolVideoEncodeFeedbackCreateInfoKHR feedbackInfo = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_VIDEO_ENCODE_FEEDBACK_CREATE_INFO_KHR,
            .pNext = nullptr,
            .encodeFeedbackFlags = VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BUFFER_OFFSET_BIT_KHR |
                                   VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BYTES_WRITTEN_BIT_KHR,
        };
        queryPoolInfo.pNext = &feedbackInfo;
        
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
        static const uint8_t startCode[] = { 0x00, 0x00, 0x00, 0x01 };
        outputFile.write(reinterpret_cast<const char*>(startCode), sizeof(startCode));
        outputFile.write(reinterpret_cast<const char*>(data), size);
    }
    
    // Get current frame number
    uint64_t getFrameCount() const { return frameCounter; }
    
    // Check if next frame should be IDR
    bool isNextFrameIDR() const {
        return (frameCounter == 0) || (config.gopSize > 0 && (frameCounter % config.gopSize == 0));
    }
    
    // Increment frame counter
    void nextFrame() { 
        frameCounter++; 
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
        
        for (auto& mem : sessionMemory) {
            if (mem != VK_NULL_HANDLE) {
                vkFreeMemory(device, mem, nullptr);
            }
        }
        sessionMemory.clear();
        
        isInitialized = false;
    }

private:
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
            VkMemoryAllocateInfo allocInfo = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = memReqs[i].memoryRequirements.size,
                .memoryTypeIndex = vulkanDevice->getMemoryType(
                    memReqs[i].memoryRequirements.memoryTypeBits,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
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
};

class VulkanExample : public VulkanExampleBase
{
public:
	vkglTF::Model model;

	struct UniformData {
		glm::mat4 projection;
		glm::mat4 model;
		glm::mat4 view;
		int32_t texIndex = 0;
	} uniformData;
	std::array<vks::Buffer, maxConcurrentFrames> uniformBuffers;

	VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
	VkPipeline pipeline{ VK_NULL_HANDLE };
	VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
	std::array<VkDescriptorSet, maxConcurrentFrames> descriptorSets{};

	bool screenshotSaved{ false };

	// Video encoding resources
	RGBtoNV12Converter rgbToNv12Converter;
	VulkanH264Encoder h264Encoder;

	VulkanExample() : VulkanExampleBase(), uniformBuffers{}
	{
		title = "Saving framebuffer to screenshot";
		apiVersion = VK_API_VERSION_1_1;
	    settings.validation = true;
		camera.type = Camera::CameraType::lookat;
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
		camera.setRotation(glm::vec3(-25.0f, 23.75f, 0.0f));
		camera.setTranslation(glm::vec3(0.0f, 0.0f, -3.0f));
	}

	~VulkanExample() override
	{
		if (device) {
			vkDestroyPipeline(device, pipeline, nullptr);
			vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
			for (auto& buffer : uniformBuffers) {
				buffer.destroy();
			}
		}
	}
    void getEnabledExtensions() override
	{
	    // Check for Vulkan Video extensions
	    bool videoQueue = vulkanDevice->extensionSupported(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);
	    bool encodeQueue = vulkanDevice->extensionSupported(VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME);
		bool sync2 = vulkanDevice->extensionSupported(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);

	    std::cout << "Vulkan Video Encoding Support:" << std::endl;
	    std::cout << "  Base Video Queue: " << (videoQueue ? "Yes" : "No") << std::endl;
	    std::cout << "  Encode Queue:     " << (encodeQueue ? "Yes" : "No") << std::endl;
		std::cout << "  Synchronization2: " << (sync2 ? "Yes" : "No") << std::endl;

	    // Check for specific encoders
        if (videoQueue && encodeQueue && sync2) {
            enabledDeviceExtensions.push_back(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);
            enabledDeviceExtensions.push_back(VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME);
            enabledDeviceExtensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);

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
		if (!vulkanDevice->extensionSupported(VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME)) {
			std::cout << "H.264 video encoding not supported, skipping encoder setup" << std::endl;
			return;
		}

		// Initialize RGB to NV12 converter
		std::vector<VkImage> swapchainImages;
		for (uint32_t i = 0; i < swapChain.imageCount; i++) {
			swapchainImages.push_back(swapChain.images[i]);
		}

		if (!rgbToNv12Converter.initialize(vulkanDevice, width, height, 
				swapChain.colorFormat, swapchainImages, getShadersPath())) {
			std::cerr << "Failed to initialize RGB to NV12 converter" << std::endl;
			return;
		}

		// Initialize H264 encoder
		VulkanH264Encoder::EncoderConfig encoderConfig;
		encoderConfig.width = width;
		encoderConfig.height = height;
		encoderConfig.frameRate = 60;
		encoderConfig.gopSize = 1;  // All I-frames
		encoderConfig.qp = 23;
		encoderConfig.outputPath = "recording.h264";

		if (!h264Encoder.initialize(vulkanDevice, instance, encoderConfig)) {
			std::cerr << "Failed to initialize H264 encoder" << std::endl;
			return;
		}

		std::cout << "Video encoding pipeline initialized successfully" << std::endl;
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
		}
	}

};

VULKAN_EXAMPLE_MAIN()