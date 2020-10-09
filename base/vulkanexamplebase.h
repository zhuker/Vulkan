/*
* Vulkan Example base class
*
* Copyright (C) by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once
#include <xcb/xcb.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <numeric>
#include <random>
#include <sys/stat.h>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <numeric>
#include <string>

#include "vulkan/vulkan.h"

#include "VulkanBuffer.h"
#include "VulkanDevice.h"
#include "VulkanSwapChain.h"
#include "VulkanTexture.h"
#include "VulkanTools.h"

#include "VulkanInitializers.hpp"

class VulkanExampleBase
{
  private:
	std::string getWindowTitle();
	uint32_t    destWidth;
	uint32_t    destHeight;
	void        createPipelineCache();
	void        createCommandPool();
	void        createSynchronizationPrimitives();
	void        initSwapchain();
	void        setupSwapChain();
	std::string shaderDir = "glsl";

  protected:
	// Returns the path to the root of the glsl or hlsl shader directory.
	std::string getShadersPath() const;

	// Frame counter to display fps
	uint32_t                                                    frameCounter = 0;
	std::chrono::time_point<std::chrono::high_resolution_clock> lastTimestamp;
	// Vulkan instance, stores all per-application states
	VkInstance               instance;
	std::vector<std::string> supportedInstanceExtensions;
	// Physical device (GPU) that Vulkan will use
	VkPhysicalDevice physicalDevice;
	// Stores physical device properties (for e.g. checking device limits)
	VkPhysicalDeviceProperties deviceProperties;
	// Stores the features available on the selected physical device (for e.g. checking if a feature is available)
	VkPhysicalDeviceFeatures deviceFeatures;
	// Stores all available memory (type) properties for the physical device
	VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
	/** @brief Set of physical device features to be enabled for this example (must be set in the derived constructor) */
	VkPhysicalDeviceFeatures enabledFeatures{};
	/** @brief Set of device extensions to be enabled for this example (must be set in the derived constructor) */
	std::vector<const char *> enabledDeviceExtensions;
	std::vector<const char *> enabledInstanceExtensions;
	/** @brief Optional pNext structure for passing extension structures to device creation */
	void *deviceCreatepNextChain = nullptr;
	/** @brief Logical device, application's view of the physical device (GPU) */
	VkDevice device;
	// Handle to the device graphics queue that command buffers are submitted to
	VkQueue queue;
	// Depth buffer format (selected during Vulkan initialization)
	VkFormat depthFormat;
	// Command buffer pool
	VkCommandPool cmdPool;
	/** @brief Pipeline stages used to wait at for graphics queue submissions */
	VkPipelineStageFlags submitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	// Contains command buffers and semaphores to be presented to the queue
	VkSubmitInfo submitInfo;
	// Command buffers used for rendering
	std::vector<VkCommandBuffer> drawCmdBuffers;
	// Global render pass for frame buffer writes
	VkRenderPass renderPass;
	// List of available frame buffers (same as number of swap chain images)
	std::vector<VkFramebuffer> frameBuffers;
	// Active frame buffer index
	uint32_t currentBuffer = 0;
	// Descriptor set pool
	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
	// List of shader modules created (stored for cleanup)
	std::vector<VkShaderModule> shaderModules;
	// Pipeline cache object
	VkPipelineCache pipelineCache;
	// Wraps the swap chain to present images (framebuffers) to the windowing system
	VulkanSwapChain swapChain;
	// Synchronization semaphores
	struct
	{
		// Swap chain image presentation
		VkSemaphore presentComplete;
		// Command buffer submission and execution
		VkSemaphore renderComplete;
	} semaphores;
	std::vector<VkFence> waitFences;
	void                 destroyCommandBuffers();
	void                 createCommandBuffers();

  public:
	bool     prepared = false;
	uint32_t width    = 1280;
	uint32_t height   = 720;

	/** @brief Encapsulated physical and logical vulkan device */
	vks::VulkanDevice *vulkanDevice;

	std::string title      = "Vulkan Example";
	std::string name       = "vulkanExample";
	uint32_t    apiVersion = VK_API_VERSION_1_0;

	struct
	{
		VkImage        image;
		VkDeviceMemory mem;
		VkImageView    view;
	} depthStencil;

	// OS specific
	bool                     quit = false;
	xcb_connection_t *       connection;
	xcb_screen_t *           screen;
	xcb_window_t             window;
	xcb_intern_atom_reply_t *atom_wm_delete_window;

	explicit VulkanExampleBase();
	virtual ~VulkanExampleBase();
	/** @brief Setup the vulkan instance, enable required extensions and connect to the physical device (GPU) */
	bool initVulkan();

	xcb_window_t setupWindow();
	void         initxcbConnection();
	void         handleEvent(const xcb_generic_event_t *event);

	/** @brief (Virtual) Creates the application wide Vulkan instance */
	virtual VkResult createInstance();
	/** @brief (Pure virtual) Render function to be implemented by the sample application */
	virtual void render() = 0;
	/** @brief (Virtual) Setup default depth and stencil views */
	virtual void setupDepthStencil();
	/** @brief (Virtual) Setup default framebuffers for all requested swapchain images */
	virtual void setupFrameBuffer();
	/** @brief (Virtual) Setup a default renderpass */
	virtual void setupRenderPass();

	/** @brief Prepares all Vulkan resources and functions required to run the sample */
	virtual void prepare();

	/** @brief Loads a SPIR-V shader file for the given shader stage */
	VkPipelineShaderStageCreateInfo loadShader(const std::string &fileName, VkShaderStageFlagBits stage);

	/** @brief Entry point for the main render loop */
	void renderLoop();

	/** Prepare the next frame for workload submission by acquiring the next swap chain image */
	void prepareFrame();
	/** @brief Presents the current image to the swap chain */
	void submitFrame();
	void createDevice();
};

