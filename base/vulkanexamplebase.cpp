/*
* Vulkan Example base class
*
* Copyright (C) by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"

VkResult VulkanExampleBase::createInstance()
{
	VkApplicationInfo appInfo = {};
	appInfo.sType             = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName  = name.c_str();
	appInfo.pEngineName       = name.c_str();
	appInfo.apiVersion        = apiVersion;

	std::vector<const char *> instanceExtensions = {VK_KHR_SURFACE_EXTENSION_NAME};

	// Enable surface extensions depending on os
	instanceExtensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);

	// Get extensions supported by the instance and store for later use
	uint32_t extCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
	if (extCount > 0)
	{
		std::vector<VkExtensionProperties> extensions(extCount);
		if (vkEnumerateInstanceExtensionProperties(nullptr, &extCount, &extensions.front()) == VK_SUCCESS)
		{
			for (const auto &extension : extensions)
			{
				supportedInstanceExtensions.emplace_back(extension.extensionName);
			}
		}
	}

	// Enabled requested instance extensions
	if (!enabledInstanceExtensions.empty())
	{
		for (const char *enabledExtension : enabledInstanceExtensions)
		{
			// Output message if requested extension is not available
			if (std::find(supportedInstanceExtensions.begin(), supportedInstanceExtensions.end(), enabledExtension) == supportedInstanceExtensions.end())
			{
				std::cerr << "Enabled instance extension \"" << enabledExtension << "\" is not present at instance level\n";
			}
			instanceExtensions.push_back(enabledExtension);
		}
	}

	VkInstanceCreateInfo instanceCreateInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	instanceCreateInfo.pNext                = nullptr;
	instanceCreateInfo.pApplicationInfo     = &appInfo;
	if (!instanceExtensions.empty())
	{
		instanceCreateInfo.enabledExtensionCount   = (uint32_t) instanceExtensions.size();
		instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
	}
	return vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
}

std::string VulkanExampleBase::getWindowTitle()
{
	return title + " - " + deviceProperties.deviceName;
}

void VulkanExampleBase::createCommandBuffers()
{
	// Create one command buffer for each swap chain image and reuse for rendering
	drawCmdBuffers.resize(swapChain.imageCount);

	VkCommandBufferAllocateInfo cmdBufAllocateInfo =
	    vks::initializers::commandBufferAllocateInfo(
	        cmdPool,
	        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
	        static_cast<uint32_t>(drawCmdBuffers.size()));

	VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, drawCmdBuffers.data()));
}

void VulkanExampleBase::destroyCommandBuffers()
{
	vkFreeCommandBuffers(device, cmdPool, static_cast<uint32_t>(drawCmdBuffers.size()), drawCmdBuffers.data());
}

std::string VulkanExampleBase::getShadersPath() const
{
	return getAssetPath() + "shaders/" + shaderDir + "/";
}

void VulkanExampleBase::createPipelineCache()
{
	VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
	pipelineCacheCreateInfo.sType                     = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	VK_CHECK_RESULT(vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache));
}

void VulkanExampleBase::prepare()
{
	initSwapchain();
	createCommandPool();
	setupSwapChain();
	createCommandBuffers();
	createSynchronizationPrimitives();
	setupDepthStencil();
	setupRenderPass();
	createPipelineCache();
	setupFrameBuffer();
}

VkPipelineShaderStageCreateInfo VulkanExampleBase::loadShader(const std::string &fileName, VkShaderStageFlagBits stage)
{
	VkPipelineShaderStageCreateInfo shaderStage = {};
	shaderStage.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStage.stage                           = stage;
	shaderStage.module                          = vks::tools::loadShader(fileName.c_str(), device);
	shaderStage.pName                           = "main";
	assert(shaderStage.module != VK_NULL_HANDLE);
	shaderModules.push_back(shaderStage.module);
	return shaderStage;
}

void VulkanExampleBase::renderLoop()
{
	destWidth     = width;
	destHeight    = height;
	lastTimestamp = std::chrono::high_resolution_clock::now();
	xcb_flush(connection);
	while (!quit)
	{
		auto                 tStart = std::chrono::high_resolution_clock::now();
		xcb_generic_event_t *event;
		while ((event = xcb_poll_for_event(connection)))
		{
			handleEvent(event);
			free(event);
		}
		render();
		frameCounter++;
		auto tEnd = std::chrono::high_resolution_clock::now();
		// Convert to clamped timer value
		float fpsTimer = std::chrono::duration<double, std::milli>(tEnd - lastTimestamp).count();
		if (fpsTimer > 1000.0f)
		{
			frameCounter  = 0;
			lastTimestamp = tEnd;
		}
	}
	// Flush device to make sure all resources can be freed
	if (device != VK_NULL_HANDLE)
	{
		vkDeviceWaitIdle(device);
	}
}

void VulkanExampleBase::prepareFrame()
{
	// Acquire the next image from the swap chain
	VkResult result = swapChain.acquireNextImage(semaphores.presentComplete, &currentBuffer);
	// Recreate the swapchain if it's no longer compatible with the surface (OUT_OF_DATE) or no longer optimal for presentation (SUBOPTIMAL)
	if ((result == VK_ERROR_OUT_OF_DATE_KHR) || (result == VK_SUBOPTIMAL_KHR))
	{
	}
	else
	{
		VK_CHECK_RESULT(result);
	}
}

void VulkanExampleBase::submitFrame()
{
	VkResult result = swapChain.queuePresent(queue, currentBuffer, semaphores.renderComplete);
	if (!((result == VK_SUCCESS) || (result == VK_SUBOPTIMAL_KHR)))
	{
		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			// Swap chain is no longer compatible with the surface and needs to be recreated
			return;
		}
		else
		{
			VK_CHECK_RESULT(result);
		}
	}
	VK_CHECK_RESULT(vkQueueWaitIdle(queue));
}

VulkanExampleBase::VulkanExampleBase()
{
	// Check for a valid asset path
	struct stat info;
	if (stat(getAssetPath().c_str(), &info) != 0)
	{
		std::cerr << "Error: Could not find asset path in " << getAssetPath() << "\n";
		exit(-1);
	}

	initxcbConnection();
}

VulkanExampleBase::~VulkanExampleBase()
{
	// Clean up Vulkan resources
	swapChain.cleanup();
	if (descriptorPool != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
	}
	destroyCommandBuffers();
	vkDestroyRenderPass(device, renderPass, nullptr);
	for (auto &frameBuffer : frameBuffers)
	{
		vkDestroyFramebuffer(device, frameBuffer, nullptr);
	}

	for (auto &shaderModule : shaderModules)
	{
		vkDestroyShaderModule(device, shaderModule, nullptr);
	}
	vkDestroyImageView(device, depthStencil.view, nullptr);
	vkDestroyImage(device, depthStencil.image, nullptr);
	vkFreeMemory(device, depthStencil.mem, nullptr);

	vkDestroyPipelineCache(device, pipelineCache, nullptr);

	vkDestroyCommandPool(device, cmdPool, nullptr);

	vkDestroySemaphore(device, semaphores.presentComplete, nullptr);
	vkDestroySemaphore(device, semaphores.renderComplete, nullptr);
	for (auto &fence : waitFences)
	{
		vkDestroyFence(device, fence, nullptr);
	}

	delete vulkanDevice;

	vkDestroyInstance(instance, nullptr);
	xcb_destroy_window(connection, window);
	xcb_disconnect(connection);
}

bool VulkanExampleBase::initVulkan()
{
	// Vulkan instance
	auto err = createInstance();
	if (err)
		vks::tools::exitFatal("Could not create Vulkan instance : \n" + vks::tools::errorString(err), err);

	createDevice();

	// Get a graphics queue from the device
	vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.graphics, 0, &queue);

	// Find a suitable depth format
	VkBool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &depthFormat);
	assert(validDepthFormat);

	swapChain.connect(instance, physicalDevice, device);

	// Create synchronization objects
	VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
	// Create a semaphore used to synchronize image presentation
	// Ensures that the image is displayed before we start submitting new commands to the queue
	VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphores.presentComplete));
	// Create a semaphore used to synchronize command submission
	// Ensures that the image is not presented until all commands have been submitted and executed
	VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphores.renderComplete));

	// Set up submit info structure
	// Semaphores will stay the same during application lifetime
	// Command buffer submission info is set by each example
	submitInfo                      = vks::initializers::submitInfo();
	submitInfo.pWaitDstStageMask    = &submitPipelineStages;
	submitInfo.waitSemaphoreCount   = 1;
	submitInfo.pWaitSemaphores      = &semaphores.presentComplete;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores    = &semaphores.renderComplete;

	return true;
}

void VulkanExampleBase::createDevice()
{
	VkResult err;

	// Vulkan instance
	err = createInstance();
	if (err)
		vks::tools::exitFatal("Could not create Vulkan instance : \n" + vks::tools::errorString(err), err);

	// Physical device
	uint32_t gpuCount = 0;
	// Get number of available physical devices
	err = vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr);
	if (err || gpuCount <= 0)
		vks::tools::exitFatal("GPU not found : " + vks::tools::errorString(err), err);

	// Enumerate devices
	std::vector<VkPhysicalDevice> physicalDevices(gpuCount);
	err = vkEnumeratePhysicalDevices(instance, &gpuCount, physicalDevices.data());
	if (err)
	{
		vks::tools::exitFatal("Could not enumerate physical devices : \n" + vks::tools::errorString(err), err);
	}

	for (const auto &dev : physicalDevices)
	{
		VkPhysicalDeviceProperties props{};
		vkGetPhysicalDeviceProperties(dev, &props);
		std::cout << "Device : " << props.deviceName << std::endl;
		std::cout << "  Type : " << vks::tools::physicalDeviceTypeString(props.deviceType) << "\n";
		std::cout << "   API : " << (props.apiVersion >> 22) << "." << ((props.apiVersion >> 12) & 0x3ff) << "." << (props.apiVersion & 0xfff) << "\n";
	}

	// Select physical device to be used for the Vulkan example
	// Defaults to the first device unless specified by command line
	uint32_t selectedDevice = 0;

	physicalDevice = physicalDevices[selectedDevice];

	// Store properties (including limits), features and memory properties of the physical device (so that examples can check against them)
	vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
	vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);

	// Vulkan device creation
	// This is handled by a separate class that gets a logical device representation
	// and encapsulates functions related to a device
	vulkanDevice = new vks::VulkanDevice(physicalDevice);
	err          = vulkanDevice->createLogicalDevice(enabledFeatures, enabledDeviceExtensions, deviceCreatepNextChain);
	if (err != VK_SUCCESS)
		vks::tools::exitFatal("Could not create Vulkan device: \n" + vks::tools::errorString(err), err);
	device = vulkanDevice->logicalDevice;
}

static inline xcb_intern_atom_reply_t *intern_atom_helper(xcb_connection_t *conn, bool only_if_exists, const char *str)
{
	xcb_intern_atom_cookie_t cookie = xcb_intern_atom(conn, only_if_exists, strlen(str), str);
	return xcb_intern_atom_reply(conn, cookie, NULL);
}

// Set up a window using XCB and request event types
xcb_window_t VulkanExampleBase::setupWindow()
{
	uint32_t value_mask, value_list[32];

	window = xcb_generate_id(connection);

	value_mask    = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
	value_list[0] = screen->black_pixel;
	value_list[1] =
	    XCB_EVENT_MASK_KEY_RELEASE |
	    XCB_EVENT_MASK_KEY_PRESS |
	    XCB_EVENT_MASK_EXPOSURE |
	    XCB_EVENT_MASK_STRUCTURE_NOTIFY |
	    XCB_EVENT_MASK_POINTER_MOTION |
	    XCB_EVENT_MASK_BUTTON_PRESS |
	    XCB_EVENT_MASK_BUTTON_RELEASE;

	xcb_create_window(connection,
	                  XCB_COPY_FROM_PARENT,
	                  window, screen->root,
	                  0, 0, width, height, 0,
	                  XCB_WINDOW_CLASS_INPUT_OUTPUT,
	                  screen->root_visual,
	                  value_mask, value_list);

	/* Magic code that will send notification when window is destroyed */
	xcb_intern_atom_reply_t *reply = intern_atom_helper(connection, true, "WM_PROTOCOLS");
	atom_wm_delete_window          = intern_atom_helper(connection, false, "WM_DELETE_WINDOW");

	xcb_change_property(connection, XCB_PROP_MODE_REPLACE,
	                    window, (*reply).atom, 4, 32, 1,
                        &(*atom_wm_delete_window).atom);

	std::string windowTitle = getWindowTitle();
	xcb_change_property(connection, XCB_PROP_MODE_REPLACE,
	                    window, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8,
	                    title.size(), windowTitle.c_str());

	free(reply);

	/**
	 * Set the WM_CLASS property to display
	 * title in dash tooltip and application menu
	 * on GNOME and other desktop environments
	 */
	std::string wm_class;
	wm_class = wm_class.insert(0, name);
	wm_class = wm_class.insert(name.size(), 1, '\0');
	wm_class = wm_class.insert(name.size() + 1, title);
	wm_class = wm_class.insert(wm_class.size(), 1, '\0');
	xcb_change_property(connection, XCB_PROP_MODE_REPLACE, window, XCB_ATOM_WM_CLASS, XCB_ATOM_STRING, 8, wm_class.size() + 2, wm_class.c_str());

	xcb_map_window(connection, window);

	return (window);
}

// Initialize XCB connection
void VulkanExampleBase::initxcbConnection()
{
	const xcb_setup_t *   setup;
	xcb_screen_iterator_t iter;
	int                   scr;

	// xcb_connect always returns a non-NULL pointer to a xcb_connection_t,
	// even on failure. Callers need to use xcb_connection_has_error() to
	// check for failure. When finished, use xcb_disconnect() to close the
	// connection and free the structure.
	connection = xcb_connect(NULL, &scr);
	assert(connection);
	if (xcb_connection_has_error(connection))
	{
		printf("Could not find a compatible Vulkan ICD!\n");
		fflush(stdout);
		exit(1);
	}

	setup = xcb_get_setup(connection);
	iter  = xcb_setup_roots_iterator(setup);
	while (scr-- > 0)
		xcb_screen_next(&iter);
	screen = iter.data;
}

void VulkanExampleBase::handleEvent(const xcb_generic_event_t *event)
{
	switch (event->response_type & 0x7f)
	{
		case XCB_CLIENT_MESSAGE:
			if ((*(xcb_client_message_event_t *) event).data.data32[0] ==
			    (*atom_wm_delete_window).atom)
			{
				quit = true;
			}
			break;
		case XCB_DESTROY_NOTIFY:
			quit = true;
			break;
		case XCB_CONFIGURE_NOTIFY: {
			const auto *cfgEvent = (const xcb_configure_notify_event_t *) event;
			if ((prepared) && ((cfgEvent->width != width) || (cfgEvent->height != height)))
			{
				destWidth  = cfgEvent->width;
				destHeight = cfgEvent->height;
				if ((destWidth > 0) && (destHeight > 0))
				{
				}
			}
		}
		break;
		default:
			break;
	}
}

void VulkanExampleBase::createSynchronizationPrimitives()
{
	// Wait fences to sync command buffer access
	VkFenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
	waitFences.resize(drawCmdBuffers.size());
	for (auto &fence : waitFences)
	{
		VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));
	}
}

void VulkanExampleBase::createCommandPool()
{
	VkCommandPoolCreateInfo cmdPoolInfo = {};
	cmdPoolInfo.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cmdPoolInfo.queueFamilyIndex        = swapChain.queueNodeIndex;
	cmdPoolInfo.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool));
}

void VulkanExampleBase::setupDepthStencil()
{
	VkImageCreateInfo imageCI{};
	imageCI.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageCI.imageType   = VK_IMAGE_TYPE_2D;
	imageCI.format      = depthFormat;
	imageCI.extent      = {width, height, 1};
	imageCI.mipLevels   = 1;
	imageCI.arrayLayers = 1;
	imageCI.samples     = VK_SAMPLE_COUNT_1_BIT;
	imageCI.tiling      = VK_IMAGE_TILING_OPTIMAL;
	imageCI.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

	VK_CHECK_RESULT(vkCreateImage(device, &imageCI, nullptr, &depthStencil.image));
	VkMemoryRequirements memReqs{};
	vkGetImageMemoryRequirements(device, depthStencil.image, &memReqs);

	VkMemoryAllocateInfo memAllloc{};
	memAllloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memAllloc.allocationSize  = memReqs.size;
	memAllloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	VK_CHECK_RESULT(vkAllocateMemory(device, &memAllloc, nullptr, &depthStencil.mem));
	VK_CHECK_RESULT(vkBindImageMemory(device, depthStencil.image, depthStencil.mem, 0));

	VkImageViewCreateInfo imageViewCI{};
	imageViewCI.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	imageViewCI.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
	imageViewCI.image                           = depthStencil.image;
	imageViewCI.format                          = depthFormat;
	imageViewCI.subresourceRange.baseMipLevel   = 0;
	imageViewCI.subresourceRange.levelCount     = 1;
	imageViewCI.subresourceRange.baseArrayLayer = 0;
	imageViewCI.subresourceRange.layerCount     = 1;
	imageViewCI.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
	// Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
	if (depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT)
	{
		imageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
	}
	VK_CHECK_RESULT(vkCreateImageView(device, &imageViewCI, nullptr, &depthStencil.view));
}

void VulkanExampleBase::setupFrameBuffer()
{
	VkImageView attachments[2];

	// Depth/Stencil attachment is the same for all frame buffers
	attachments[1] = depthStencil.view;

	VkFramebufferCreateInfo frameBufferCreateInfo = {};
	frameBufferCreateInfo.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	frameBufferCreateInfo.pNext                   = nullptr;
	frameBufferCreateInfo.renderPass              = renderPass;
	frameBufferCreateInfo.attachmentCount         = 2;
	frameBufferCreateInfo.pAttachments            = attachments;
	frameBufferCreateInfo.width                   = width;
	frameBufferCreateInfo.height                  = height;
	frameBufferCreateInfo.layers                  = 1;

	// Create frame buffers for every swap chain image
	frameBuffers.resize(swapChain.imageCount);
	for (uint32_t i = 0; i < frameBuffers.size(); i++)
	{
		attachments[0] = swapChain.buffers[i].view;
		VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCreateInfo, nullptr, &frameBuffers[i]));
	}
}

void VulkanExampleBase::setupRenderPass()
{
	std::array<VkAttachmentDescription, 2> attachments = {};
	// Color attachment
	attachments[0].format         = swapChain.colorFormat;
	attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[0].finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	// Depth attachment
	attachments[1].format         = depthFormat;
	attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference colorReference = {};
	colorReference.attachment            = 0;
	colorReference.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthReference = {};
	depthReference.attachment            = 1;
	depthReference.layout                = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpassDescription    = {};
	subpassDescription.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpassDescription.colorAttachmentCount    = 1;
	subpassDescription.pColorAttachments       = &colorReference;
	subpassDescription.pDepthStencilAttachment = &depthReference;
	subpassDescription.inputAttachmentCount    = 0;
	subpassDescription.pInputAttachments       = nullptr;
	subpassDescription.preserveAttachmentCount = 0;
	subpassDescription.pPreserveAttachments    = nullptr;
	subpassDescription.pResolveAttachments     = nullptr;

	// Subpass dependencies for layout transitions
	std::array<VkSubpassDependency, 2> dependencies{};

	dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass      = 0;
	dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	dependencies[1].srcSubpass      = 0;
	dependencies[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[1].dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount        = static_cast<uint32_t>(attachments.size());
	renderPassInfo.pAttachments           = attachments.data();
	renderPassInfo.subpassCount           = 1;
	renderPassInfo.pSubpasses             = &subpassDescription;
	renderPassInfo.dependencyCount        = static_cast<uint32_t>(dependencies.size());
	renderPassInfo.pDependencies          = dependencies.data();

	VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
}

void VulkanExampleBase::initSwapchain()
{
	swapChain.initSurface(connection, window);
}

void VulkanExampleBase::setupSwapChain()
{
	swapChain.create(&width, &height, false);
}