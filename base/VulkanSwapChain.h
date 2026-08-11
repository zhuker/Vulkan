/*
* Class wrapping access to the swap chain
* 
* A swap chain is a collection of framebuffers used for rendering and presentation to the windowing system
*
* Copyright (C) 2016-2025 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <stdlib.h>
#include <string>
#include <assert.h>
#include <stdio.h>
#include <vector>

#include <vulkan/vulkan.h>
#include "VulkanTools.h"

#ifdef __ANDROID__
#include "VulkanAndroid.h"
#endif

#ifdef __APPLE__
#include <sys/utsname.h>
#endif

class VulkanSwapChain
{
private: 
	VkInstance instance{ VK_NULL_HANDLE };
	VkDevice device{ VK_NULL_HANDLE };
	VkPhysicalDevice physicalDevice{ VK_NULL_HANDLE };
	VkQueue queue{ VK_NULL_HANDLE };
	VkSurfaceKHR surface{ VK_NULL_HANDLE };
	// Offscreen mode only: memory backing the images that would otherwise be owned by the swapchain
	std::vector<VkDeviceMemory> imageMemory{};
	// Offscreen mode only: dimensions of the images and command pool used for storing them to disk
	VkExtent2D imageExtent{};
	VkCommandPool commandPool{ VK_NULL_HANDLE };
	// Offscreen mode only: host visible image that the images are copied into before they are stored to disk
	VkImage stagingImage{ VK_NULL_HANDLE };
	VkDeviceMemory stagingImageMemory{ VK_NULL_HANDLE };
	VkSubresourceLayout stagingImageLayout{};
	const char* stagingImageData{ nullptr };
	// Offscreen mode only: set if the images can be blitted (which also converts them to the format stored to disk)
	bool supportsBlit{ false };
	// Offscreen mode only: index of the image returned by the next call to acquireNextImage
	uint32_t nextImageIndex{ 0 };
	/* Create the images usually owned by the swapchain ourselves, used if no presentation engine is available */
	void createOffscreen(uint32_t width, uint32_t height);
	/* Create the host visible image that images are copied into for storing them to disk */
	void createStagingImage();
	/* Destroy the host visible image used for storing images to disk */
	void destroyStagingImage();
	/* Destroy the image views (and in offscreen mode also the images and their memory) */
	void destroyImages();
	/* Empty submission used to signal and/or wait for semaphores that would be handled by the presentation engine */
	VkResult submitEmpty(VkSemaphore waitSemaphore, VkSemaphore signalSemaphore);
	/* Get the index of a memory type that matches the given requirements */
	uint32_t getMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags properties);
	/* Store the contents of one of the images to a ppm file, used instead of presenting in offscreen mode */
	void saveImage(uint32_t imageIndex);
public:
	VkFormat colorFormat{};
	VkColorSpaceKHR colorSpace{};
	VkSwapchainKHR swapChain{ VK_NULL_HANDLE };
	std::vector<VkImage> images{};
	std::vector<VkImageView> imageViews{};
	uint32_t queueNodeIndex{ UINT32_MAX };
	uint32_t imageCount{ 0 };
	/** @brief Render without a window and a presentation engine, must be set before creating the swapchain */
	bool offscreen{ false };
	/** @brief Name of the file that images are stored to instead of being presented in offscreen mode */
	std::string offscreenFilename{ "offscreen.ppm" };

#if defined(VK_USE_PLATFORM_WIN32_KHR)
	void initSurface(void* platformHandle, void* platformWindow);
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
	void initSurface(ANativeWindow* window);
#elif defined(VK_USE_PLATFORM_DIRECTFB_EXT)
	void initSurface(IDirectFB* dfb, IDirectFBSurface* window);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
	void initSurface(wl_display* display, wl_surface* window);
#elif defined(VK_USE_PLATFORM_XCB_KHR)
	void initSurface(xcb_connection_t* connection, xcb_window_t window);
#elif (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK))
	void initSurface(void* view);
#elif defined(VK_USE_PLATFORM_METAL_EXT)
	void initSurface(CAMetalLayer* metalLayer);
#elif (defined(_DIRECT2DISPLAY) || defined(VK_USE_PLATFORM_HEADLESS_EXT))
	void initSurface(uint32_t width, uint32_t height);
#if defined(_DIRECT2DISPLAY)
	void createDirect2DisplaySurface(uint32_t width, uint32_t height);
#endif
#elif defined(VK_USE_PLATFORM_SCREEN_QNX)
	void initSurface(screen_context_t screen_context, screen_window_t screen_window);
#endif
	/* Select the queue family and color format for offscreen rendering, replaces initSurface if no presentation engine is available */
	void initOffscreen();
	/* Set the Vulkan objects required for swapchain creation and management, must be called before swapchain creation */
	void setContext(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device, VkQueue queue = VK_NULL_HANDLE);
	/**
	* Create the swapchain and get its images with given width and height
	* 
	* @param width Pointer to the width of the swapchain (may be adjusted to fit the requirements of the swapchain)
	* @param height Pointer to the height of the swapchain (may be adjusted to fit the requirements of the swapchain)
	* @param vsync (Optional, default = false) Can be used to force vsync-ed rendering (by using VK_PRESENT_MODE_FIFO_KHR as presentation mode)
	*/
	void create(uint32_t& width, uint32_t& height, bool vsync = false, bool fullscreen = false);
	/**
	* Acquires the next image in the swap chain
	* 
	* @param presentCompleteSemaphore (Optional) Semaphore that is signaled when the image is ready for use
	* @param imageIndex Pointer to the image index that will be increased if the next image could be acquired
	* 
	* @note The function will always wait until the next image has been acquired by setting timeout to UINT64_MAX
	* 
	* @return VkResult of the image acquisition
	*/
	VkResult acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t& imageIndex);
	/**
	* Queues an image for presentation
	*
	* @param waitSemaphore Semaphore that is waited on before the image is presented
	* @param imageIndex Index of the swapchain image to present
	*
	* @return VkResult of the presentation
	*/
	VkResult queuePresent(VkSemaphore waitSemaphore, uint32_t imageIndex);
	/* Free all Vulkan resources acquired by the swapchain */
	void cleanup();
};
