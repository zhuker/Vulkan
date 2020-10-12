/*
* Vulkan Example - Basic example for ray tracing using VK_NV_ray_tracing
*
* Copyright (C) by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <sys/stat.h>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <set>
#include <xcb/xcb.h>

#include "vulkan/vulkan.h"

#include "VulkanBuffer.h"
#include "VulkanDevice.h"
#include "VulkanSwapChain.h"
#include "VulkanTools.h"

#include "VulkanInitializers.hpp"

static constexpr uint32_t NUM_TIME_STEPS = 10;
static glm::mat3x4        transform_at_time(const std::vector<glm::mat3x4> &transforms, const float ray_time)
{
	assert(ray_time >= 0.0f && ray_time <= 1.0f);
	if (ray_time == 0.0f)
		return transforms[0];

	float time_per_step = 1.0f / (NUM_TIME_STEPS - 1);

	for (int32_t step = 0; step < NUM_TIME_STEPS; step++)
	{
		int32_t next_step = step + 1;
		float   t         = time_per_step * step;
		float   tnext     = time_per_step * next_step;
		if (tnext >= ray_time && ray_time > t)
		{
			const glm::mat3x4 &transform      = transforms[step];
			const glm::mat3x4 &next_transform = transforms[next_step];

			float x      = transform[0][3];
			float y      = transform[1][3];
			float z      = transform[2][3];
			float next_x = next_transform[0][3];
			float next_y = next_transform[1][3];
			float next_z = next_transform[2][3];

			float d  = (ray_time - t) / time_per_step;
			float x_ = x * (1.0f - d) + next_x * d;
			float y_ = y * (1.0f - d) + next_y * d;
			float z_ = z * (1.0f - d) + next_z * d;
			printf("\t%f (%f, %f, %f)\n", d, x_, y_, z_);
			glm::mat3x4 copy = transform;
			copy[0][3]       = x_;
			copy[1][3]       = y_;
			copy[2][3]       = z_;
			return copy;
		}
	}
	return transforms[0];
}

static glm::mat3x4 transform_at_time(const glm::mat3x4 &initial_transform, const glm::vec3 &isovelocity, const float ray_time)
{
	assert(ray_time >= 0.0f && ray_time <= 1.0f);
	if (ray_time == 0.0f)
		return initial_transform;
	if (isovelocity == glm::vec3{0})
		return initial_transform;

	float time_per_step = 1.0f / (NUM_TIME_STEPS - 1);

	for (int32_t step = 0; step < NUM_TIME_STEPS; step++)
	{
		int32_t next_step = step + 1;
		float   t         = time_per_step * step;
		float   tnext     = time_per_step * next_step;
		if (tnext >= ray_time && ray_time > t)
		{
			float x = initial_transform[0][3] + (isovelocity[0] * 0.1f * step / NUM_TIME_STEPS);
			float y = initial_transform[1][3] + (isovelocity[1] * 0.1f * step / NUM_TIME_STEPS);
			float z = initial_transform[2][3] + (isovelocity[2] * 0.1f * step / NUM_TIME_STEPS);

			float next_x = initial_transform[0][3] + (isovelocity[0] * 0.1f * next_step / NUM_TIME_STEPS);
			float next_y = initial_transform[1][3] + (isovelocity[1] * 0.1f * next_step / NUM_TIME_STEPS);
			float next_z = initial_transform[2][3] + (isovelocity[2] * 0.1f * next_step / NUM_TIME_STEPS);

			float d  = (ray_time - t) / time_per_step;
			float x_ = x * (1.0f - d) + next_x * d;
			float y_ = y * (1.0f - d) + next_y * d;
			float z_ = z * (1.0f - d) + next_z * d;
			printf("\t%f (%f, %f, %f)\n", d, x_, y_, z_);
			glm::mat3x4 copy = initial_transform;
			copy[0][3]       = x_;
			copy[1][3]       = y_;
			copy[2][3]       = z_;
			return copy;
		}
	}
	return initial_transform;
}

static int64_t current_time_msec()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static inline xcb_intern_atom_reply_t *intern_atom_helper(xcb_connection_t *conn, bool only_if_exists, const char *str)
{
	xcb_intern_atom_cookie_t cookie = xcb_intern_atom(conn, only_if_exists, strlen(str), str);
	return xcb_intern_atom_reply(conn, cookie, nullptr);
}

// Ray tracing acceleration structure
struct AccelerationStructure
{
	VkDeviceMemory                memory                = nullptr;
	VkAccelerationStructureNV     accelerationStructure = nullptr;
	uint64_t                      handle                = 0;
	VkAccelerationStructureInfoNV buildInfo{};
};

struct Vertex
{
	float pos[4];
};

// clang-format off
std::vector<Vertex> vertices0 = {
    {{0.0f, 0.0f, 0.0f, 0.0f}},
    {{0.0f, 1.0f, 0.0f, 0.0f}},
    {{1.0f, 0.0f, 0.0f, 0.0f}},
    {{1.0f, 1.0f, 0.0f, 0.0f}},
};

std::vector<Vertex> vertices1 = {
    {{ 0.5f, -0.5f,  0.5f, 0.0f}},
    {{ 0.5f, -0.5f, -0.5f, 0.0f}},
    {{-0.5f, -0.5f,  0.5f, 0.0f}},
    {{ 0.5f,  0.5f,  0.5f, 0.0f}},
    {{-0.5f,  0.5f,  0.5f, 0.0f}},
    {{-0.5f,  0.5f, -0.5f, 0.0f}},
    {{-0.5f, -0.5f, -0.5f, 0.0f}},
    {{ 0.5f,  0.5f, -0.5f, 0.0f}},
};

std::vector<Vertex> vertices2 = {
    {{2.0f, 0.0f, -5.0f, 0.0f}},
    {{2.0f, 1.0f, -5.0f, 0.0f}},
    {{3.0f, 0.0f, -5.0f, 0.0f}},
    {{3.0f, 1.0f, -5.0f, 0.0f}},
};

// clang-format on

std::vector<uint32_t> indices0 = {0, 1, 2, 2, 3, 0};
std::vector<uint32_t> indices1 = {0, 1, 3, 3, 1, 7, 2, 6, 0, 0, 6, 1, 4, 5, 2, 2, 5, 6,
                                  3, 7, 4, 4, 7, 5, 6, 5, 1, 1, 5, 7, 4, 2, 3, 3, 2, 0};
std::vector<uint32_t> indices2 = {0, 1, 2, 2, 3, 0};

// Ray tracing geometry instance
struct GeometryInstance
{
	glm::mat3x4 transform;
	uint32_t    instanceId : 24;
	uint32_t    mask : 8;
	uint32_t    instanceOffset : 24;
	uint32_t    flags : 8;
	uint64_t    accelerationStructureHandle;
};

//RayPy and HitPy have the same memory layout this is to use same buffer for input and output
struct Ray
{
	glm::vec4 origin{};                    //point in Hit
	glm::vec4 direction{};                 //normal in Hit
	uint      unused0{0};                  //valid in Hit
	float     max_range_m = 120.0f;        //distance in Hit
	float     time        = 0.0f;          //bary_u in Hit
	float     unused1{};
	uint      unused2{};
	uint      unused3{};
	uint      lidar_id = 0;        //lidar_id in HitPy
	uint      padding{};           // makes structure 64bytes in size
};

static Ray ray(const glm::vec3 origin, const glm::vec3 direction, const float time = 0.0f)
{
	Ray r{};
	r.origin    = {origin, 0.f};
	r.direction = {direction, 0.0f};
	r.time      = time;
	return r;
}

struct Hit
{
	glm::vec4 point;           // The point in 3D space that the ray hit.
	glm::vec4 normal;          // The normalized geometry normal
	uint      valid;           // true if ray hit a vertex
	float     distance;        // The distance measured from the ray origin to this hit.
	float     bary_u;          // The u component of barycentric coordinate of this hit.
	float     bary_v;          // The v component of barycentric coordinate of this hit.
	uint      instID;          // The instance ID of the object in the scene
	uint      primID;          // The index of the primitive of the mesh hit
	uint      lidar_id;        // The lidar id of the ray
	uint      padding;         // makes structure 64bytes in size
};

struct HitPy
{
	glm::vec3    point;           // The point in 3D space that the ray hit.
	glm::vec3    normal;          // The normalized geometry normal
	float        distance;        // The distance measured from the ray origin to this hit.
	float        bary_u;          // The u component of barycentric coordinate of this hit.
	float        bary_v;          // The v component of barycentric coordinate of this hit.
	unsigned int geomID;          // The geometry ID of object in the scene (ignore for now)
	unsigned int instID;          // The instance ID of the object in the scene
	unsigned int primID;          // The index of the primitive of the mesh hit
	unsigned int lidar_id;        // The lidar id of the ray
	bool         valid;
};
static std::string to_string(const Hit &hit)
{
	char str[256] = {0};
	snprintf(str, 256, "{{%f, %f, %f}, {%f, %f, %f}, %f, %f, %f, 0, %d, %d, %d, %d}",
	         hit.point.x, hit.point.y, hit.point.z,
	         hit.normal.x, hit.normal.y, hit.normal.z,
	         hit.distance,
	         hit.bary_u, hit.bary_v,
	         hit.instID, hit.primID,
	         hit.lidar_id,
	         hit.valid);
	std::string res(str, strlen(str));
	return res;
}

static std::string to_string(const HitPy &hit)
{
	char str[256] = {0};
	snprintf(str, 256, "{{%f, %f, %f}, {%f, %f, %f}, %f, %f, %f, 0, %d, %d, %d, %d}",
	         hit.point.x, hit.point.y, hit.point.z,
	         hit.normal.x, hit.normal.y, hit.normal.z,
	         hit.distance,
	         hit.bary_u, hit.bary_v,
	         hit.instID, hit.primID,
	         hit.lidar_id,
	         hit.valid);
	std::string res(str, strlen(str));
	return res;
}

using Vector3f = glm::vec3;

std::vector<Ray> rays2 = {
    ray(Vector3f(0.000001, 0.0, 2.0), Vector3f(0.0, 0.0, -1.0)),
    ray(Vector3f(0.000001, 2.0, 0.0), Vector3f(0.0, -1.0, 0.0)),
    ray(Vector3f(0.0, 0.0, 0.000001), Vector3f(1.0, 0.0, 0.0)),
    ray(Vector3f(0.499999, 0.5, -1.0), Vector3f(0.0, 0.0, 1.0)),
    ray(Vector3f(0.0, 0.0, 0.000001), Vector3f(1.0, 0.0, 0.0)),
    ray(Vector3f(0.000001, 0.0, 0.0), Vector3f(0.0, 1.0, 0.0)),
    ray(Vector3f(0.0, 0.000001, 0.0), Vector3f(0.0, 0.0, 1.0)),
    ray(Vector3f(1.5, 0.5, -1.0), Vector3f(0.0, 0.0, 1.0), 0.95f)        // motion blur ray
};

std::vector<Ray> rays = {
    ray(Vector3f(0.000001, 0.0, 2.0), Vector3f(0.0, 0.0, -1.0)),
    ray(Vector3f(0.000001, 2.0, 0.0), Vector3f(0.0, -1.0, 0.0)),
    ray(Vector3f(0.0, 0.0, 0.000001), Vector3f(1.0, 0.0, 0.0)),
    ray(Vector3f(0.499999, 0.5, -1.0), Vector3f(0.0, 0.0, 1.0)),
    ray(Vector3f(0.0, 0.0, 0.000001), Vector3f(1.0, 0.0, 0.0)),
    ray(Vector3f(0.000001, 0.0, 0.0), Vector3f(0.0, 1.0, 0.0)),
    ray(Vector3f(0.0, 0.000001, 0.0), Vector3f(0.0, 0.0, 1.0))};

static inline void ASSERT_NEAR(float expected, float actual, float epsilon)
{
	float absdiff = abs(expected - actual);
	bool  ok      = absdiff < epsilon;
	if (!ok)
	{
		printf("expected %f actual %f absdiff %f\n", expected, actual, absdiff);
	}
	assert(ok);
}
#define ASSERT_EQ(expected, actual) assert(expected == actual)

static void assert_near(const HitPy &expected, const Hit &actual)
{
	printf("e: %s\na: %s\n", to_string(expected).c_str(), to_string(actual).c_str());
	const float eps = 0.0001f;
	ASSERT_NEAR(expected.distance, actual.distance, eps);
	ASSERT_NEAR(expected.point.x, actual.point.x, eps);
	ASSERT_NEAR(expected.point.y, actual.point.y, eps);
	ASSERT_NEAR(expected.point.z, actual.point.z, eps);
	ASSERT_NEAR(expected.normal.x, actual.normal.x, eps);
	ASSERT_NEAR(expected.normal.y, actual.normal.y, eps);
	ASSERT_NEAR(expected.normal.z, actual.normal.z, eps);
	ASSERT_NEAR(expected.bary_u, actual.bary_u, eps);
	ASSERT_NEAR(expected.bary_v, actual.bary_v, eps);
	ASSERT_EQ(expected.instID, actual.instID);
	ASSERT_EQ(expected.primID, actual.primID);
	ASSERT_EQ(expected.lidar_id, actual.lidar_id);
}

static void assert_near(const std::vector<HitPy> &expecteds, const std::vector<Hit> &actuals)
{
	if (expecteds.size() != actuals.size())
	{
		for (const auto &expected : expecteds)
			printf("e: %s\n", to_string(expected).c_str());

		for (const auto &actual : actuals)
			printf("a: %s\n", to_string(actual).c_str());
	}

	ASSERT_EQ(expecteds.size(), actuals.size());
	for (size_t i = 0; i < expecteds.size(); i++)
	{
		const auto &expected = expecteds[i];
		const auto &actual   = actuals[i];
		printf("e: %s\n", to_string(expected).c_str());
		printf("a: %s\n", to_string(actual).c_str());
		printf("\n");
	}
	printf("\n");

	for (size_t i = 0; i < actuals.size(); i++)
	{
		assert_near(expecteds[i], actuals[i]);
	}
}

HitPy from_hit(Hit *pHit)
{
	HitPy hit{};
	if (pHit == nullptr)
		return hit;
	hit.point    = glm::vec3(pHit->point);
	hit.normal   = glm::vec3(pHit->normal);
	hit.distance = pHit->distance;
	hit.bary_u   = pHit->bary_u;
	hit.bary_v   = pHit->bary_v;
	hit.geomID   = 0;
	hit.primID   = pHit->primID;
	hit.instID   = pHit->instID;
	hit.valid    = pHit->valid;
	hit.lidar_id = pHit->lidar_id;
	return hit;
}

// Indices for the different ray tracing shader types used in this example
#define INDEX_RAYGEN 0
#define INDEX_MISS 1
#define INDEX_CLOSEST_HIT 2

#define NUM_SHADER_GROUPS 3

struct ObjModel
{
	uint32_t    nbIndices{0};
	uint32_t    nbVertices{0};
	vks::Buffer vertexBuffer{};        // Device buffer of all 'Vertex'
	vks::Buffer indexBuffer{};         // Device buffer of the indices forming triangles
};

// Instance of the OBJ
struct ObjInstance
{
	std::vector<glm::mat3x4> transforms;        // Position of the instance at specified time
};

struct MyObj
{
	ObjModel              model{};
	ObjInstance           instance{};
	AccelerationStructure blas{};
	VkGeometryNV          geom{};
};

const int ignore = 0;

class VulkanExample final
{
	void createPipelineCache()
	{
		VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
		pipelineCacheCreateInfo.sType                     = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		VK_CHECK_RESULT(vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache));
	}

	void createCommandPool()
	{
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex        = swapChain.queueNodeIndex;
		cmdPoolInfo.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool));
	}

	void createSynchronizationPrimitives()
	{
		// Wait fences to sync command buffer access
		VkFenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
		waitFences.resize(drawCmdBuffers.size());
		for (auto &fence : waitFences)
		{
			VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));
		}
	}
	void initSwapchain()
	{
		swapChain.initSurface(connection, window);
	}
	void setupSwapChain()
	{
		swapChain.create(&width, &height, false);
	}
	std::string shaderDir = "glsl";

	// Returns the path to the root of the glsl or hlsl shader directory.
	std::string getShadersPath() const
	{
		return getAssetPath() + "shaders/" + shaderDir + "/";
	}
	VkInstance                            instance{};        // Vulkan instance, stores all per-application states
	std::vector<std::string>              supportedInstanceExtensions;
	VkPhysicalDevice                      physicalDevice{};                // Physical device (GPU) that Vulkan will use
	VkPhysicalDeviceProperties            deviceProperties{};              // Stores physical device properties (for e.g. checking device limits)
	VkPhysicalDeviceFeatures              deviceFeatures{};                // Stores the features available on the selected physical device (for e.g. checking if a feature is available)
	VkPhysicalDeviceMemoryProperties      deviceMemoryProperties{};        // Stores all available memory (type) properties for the physical device
	VkPhysicalDeviceFeatures              enabledFeatures{};               /** @brief Set of physical device features to be enabled for this example (must be set in the derived constructor) */
	std::vector<const char *>             enabledDeviceExtensions;         /** @brief Set of device extensions to be enabled for this example (must be set in the derived constructor) */
	std::vector<const char *>             enabledInstanceExtensions;
	void *                                deviceCreatepNextChain = nullptr;                                     /** @brief Optional pNext structure for passing extension structures to device creation */
	VkDevice                              device{};                                                             /** @brief Logical device, application's view of the physical device (GPU) */
	VkQueue                               queue{};                                                              // Handle to the device graphics queue that command buffers are submitted to
	VkFormat                              depthFormat = VK_FORMAT_UNDEFINED;                                    // Depth buffer format (selected during Vulkan initialization)
	VkCommandPool                         cmdPool{};                                                            // Command buffer pool
	VkPipelineStageFlags                  submitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; /** @brief Pipeline stages used to wait at for graphics queue submissions */
	VkSubmitInfo                          submitInfo{};                                                         // Contains command buffers and semaphores to be presented to the queue
	std::vector<VkCommandBuffer>          drawCmdBuffers;                                                       // Command buffers used for rendering
	VkRenderPass                          renderPass{};                                                         // Global render pass for frame buffer writes
	std::vector<VkFramebuffer>            frameBuffers;                                                         // List of available frame buffers (same as number of swap chain images)
	uint32_t                              currentBuffer  = 0;                                                   // Active frame buffer index
	VkDescriptorPool                      descriptorPool = VK_NULL_HANDLE;                                      // Descriptor set pool
	std::map<std::string, VkShaderModule> shaderModules;                                                        // List of shader modules created (stored for cleanup)
	VkPipelineCache                       pipelineCache{};                                                      // Pipeline cache object
	VulkanSwapChain                       swapChain;                                                            // Wraps the swap chain to present images (framebuffers) to the windowing system
	// Synchronization semaphores
	struct
	{
		VkSemaphore presentComplete;        // Swap chain image presentation
		VkSemaphore renderComplete;         // Command buffer submission and execution
	} semaphores{};
	std::vector<VkFence> waitFences;
	uint32_t             width  = 1280;
	uint32_t             height = 720;
	vks::VulkanDevice *  vulkanDevice{}; /** @brief Encapsulated physical and logical vulkan device */
	std::string          title      = "VK_NV_ray_tracing";
	std::string          name       = "vulkanExample";
	uint32_t             apiVersion = VK_API_VERSION_1_0;

	struct
	{
		VkImage        image;
		VkDeviceMemory mem;
		VkImageView    view;
	} depthStencil{};

	// OS specific
	bool                     quit = false;
	xcb_connection_t *       connection{};
	xcb_screen_t *           screen{};
	xcb_window_t             window{};
	xcb_intern_atom_reply_t *atom_wm_delete_window{};

	void destroyCommandBuffers()
	{
		vkFreeCommandBuffers(device, cmdPool, static_cast<uint32_t>(drawCmdBuffers.size()), drawCmdBuffers.data());
	}

	void createCommandBuffers()
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

	void initxcbConnection()
	{
		const xcb_setup_t *   setup;
		xcb_screen_iterator_t iter;
		int                   scr;

		// xcb_connect always returns a non-NULL pointer to a xcb_connection_t,
		// even on failure. Callers need to use xcb_connection_has_error() to
		// check for failure. When finished, use xcb_disconnect() to close the
		// connection and free the structure.
		connection = xcb_connect(nullptr, &scr);
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

	/** @brief (Virtual) Creates the application wide Vulkan instance */
	virtual VkResult createInstance()
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
	/** @brief (Virtual) Setup default depth and stencil views */
	virtual void setupDepthStencil()
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

	/** @brief (Virtual) Setup default framebuffers for all requested swapchain images */
	virtual void setupFrameBuffer()
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

	/** @brief (Virtual) Setup a default renderpass */
	virtual void setupRenderPass()
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

	/** @brief Loads a SPIR-V shader file for the given shader stage */
	VkPipelineShaderStageCreateInfo loadShader(const std::string &fileName, VkShaderStageFlagBits stage)
	{
		if (shaderModules.count(fileName) == 0)
		{
			VkShaderModule module = vks::tools::loadShader(fileName.c_str(), device);
			assert(module != VK_NULL_HANDLE);
			shaderModules.emplace(fileName, module);
		}
		VkPipelineShaderStageCreateInfo shaderStage = {};
		shaderStage.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStage.stage                           = stage;
		shaderStage.module                          = shaderModules.at(fileName);
		shaderStage.pName                           = "main";
		return shaderStage;
	}

	/** Prepare the next frame for workload submission by acquiring the next swap chain image */
	void prepareFrame()
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
	/** @brief Presents the current image to the swap chain */
	void submitFrame()
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

	void createDevice()
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

	PFN_vkCreateAccelerationStructureNV                vkCreateAccelerationStructureNV{};
	PFN_vkDestroyAccelerationStructureNV               vkDestroyAccelerationStructureNV{};
	PFN_vkBindAccelerationStructureMemoryNV            vkBindAccelerationStructureMemoryNV{};
	PFN_vkGetAccelerationStructureHandleNV             vkGetAccelerationStructureHandleNV{};
	PFN_vkCmdBuildAccelerationStructureNV              vkCmdBuildAccelerationStructureNV{};
	PFN_vkGetAccelerationStructureMemoryRequirementsNV vkGetAccelerationStructureMemoryRequirementsNV{};
	PFN_vkCreateRayTracingPipelinesNV                  vkCreateRayTracingPipelinesNV{};
	PFN_vkGetRayTracingShaderGroupHandlesNV            vkGetRayTracingShaderGroupHandlesNV{};
	PFN_vkCmdTraceRaysNV                               vkCmdTraceRaysNV{};

	VkPhysicalDeviceRayTracingPropertiesNV rayTracingProperties{};

	std::map<uint32_t, MyObj> objects;

	AccelerationStructure topLevelAS{};

	vks::Buffer shaderBindingTable;

	struct StorageImage
	{
		VkDeviceMemory memory;
		VkImage        image;
		VkImageView    view;
		VkFormat       format;
	} storageImage{};

	vks::Buffer deviceBuffer{}, hostBuffer{};

	struct UniformData
	{
		float time = 0.0f;
	} uniformData;
	vks::Buffer ubo{};

	VkPipeline            pipeline            = VK_NULL_HANDLE;
	VkPipelineLayout      pipelineLayout      = VK_NULL_HANDLE;
	VkDescriptorSet       descriptorSet       = VK_NULL_HANDLE;
	VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;

	void destroyObjectInternal(MyObj &obj)
	{
		vkFreeMemory(device, obj.blas.memory, nullptr);
		vkDestroyAccelerationStructureNV(device, obj.blas.accelerationStructure, nullptr);
		obj.blas = {};
		obj.model.vertexBuffer.destroy();
		obj.model.indexBuffer.destroy();
	}

	/*
		Set up a storage image that the ray generation shader will be writing to
	*/
	void createStorageImage()
	{
		VkImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType         = VK_IMAGE_TYPE_2D;
		image.format            = swapChain.colorFormat;
		image.extent.width      = width;
		image.extent.height     = height;
		image.extent.depth      = 1;
		image.mipLevels         = 1;
		image.arrayLayers       = 1;
		image.samples           = VK_SAMPLE_COUNT_1_BIT;
		image.tiling            = VK_IMAGE_TILING_OPTIMAL;
		image.usage             = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
		image.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &storageImage.image));

		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(device, storageImage.image, &memReqs);
		VkMemoryAllocateInfo memoryAllocateInfo = vks::initializers::memoryAllocateInfo();
		memoryAllocateInfo.allocationSize       = memReqs.size;
		memoryAllocateInfo.memoryTypeIndex      = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &storageImage.memory));
		VK_CHECK_RESULT(vkBindImageMemory(device, storageImage.image, storageImage.memory, 0));

		VkImageViewCreateInfo colorImageView           = vks::initializers::imageViewCreateInfo();
		colorImageView.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
		colorImageView.format                          = swapChain.colorFormat;
		colorImageView.subresourceRange                = {};
		colorImageView.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		colorImageView.subresourceRange.baseMipLevel   = 0;
		colorImageView.subresourceRange.levelCount     = 1;
		colorImageView.subresourceRange.baseArrayLayer = 0;
		colorImageView.subresourceRange.layerCount     = 1;
		colorImageView.image                           = storageImage.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &colorImageView, nullptr, &storageImage.view));

		VkCommandBuffer cmdBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		vks::tools::setImageLayout(cmdBuffer, storageImage.image,
		                           VK_IMAGE_LAYOUT_UNDEFINED,
		                           VK_IMAGE_LAYOUT_GENERAL,
		                           {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
		vulkanDevice->flushCommandBuffer(cmdBuffer, queue);
	}

	void createStorageBuffer()
	{
		const VkDeviceSize bufferSize = width * height * sizeof(Ray);

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
		    &hostBuffer,
		    bufferSize,
		    nullptr));

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    &deviceBuffer,
		    bufferSize));
	}
	/*
		The bottom level acceleration structure contains the scene's geometry (vertices, triangles)
	*/
	void createBottomLevelAccelerationStructure(AccelerationStructure &blas, const VkGeometryNV *geometries)
	{
		blas.buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
		blas.buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
		blas.buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
		blas.buildInfo.instanceCount = 0;
		blas.buildInfo.geometryCount = 1;
		blas.buildInfo.pGeometries   = geometries;

		VkAccelerationStructureCreateInfoNV accelerationStructureCI{};
		accelerationStructureCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
		accelerationStructureCI.info  = blas.buildInfo;
		VK_CHECK_RESULT(vkCreateAccelerationStructureNV(device, &accelerationStructureCI, nullptr, &blas.accelerationStructure));

		VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo{};
		memoryRequirementsInfo.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
		memoryRequirementsInfo.type                  = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
		memoryRequirementsInfo.accelerationStructure = blas.accelerationStructure;

		VkMemoryRequirements2 memoryRequirements2{};
		vkGetAccelerationStructureMemoryRequirementsNV(device, &memoryRequirementsInfo, &memoryRequirements2);

		VkMemoryAllocateInfo memoryAllocateInfo = vks::initializers::memoryAllocateInfo();
		memoryAllocateInfo.allocationSize       = memoryRequirements2.memoryRequirements.size;
		memoryAllocateInfo.memoryTypeIndex      = vulkanDevice->getMemoryType(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &blas.memory));

		VkBindAccelerationStructureMemoryInfoNV accelerationStructureMemoryInfo{};
		accelerationStructureMemoryInfo.sType                 = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
		accelerationStructureMemoryInfo.accelerationStructure = blas.accelerationStructure;
		accelerationStructureMemoryInfo.memory                = blas.memory;
		VK_CHECK_RESULT(vkBindAccelerationStructureMemoryNV(device, 1, &accelerationStructureMemoryInfo));

		VK_CHECK_RESULT(vkGetAccelerationStructureHandleNV(device, blas.accelerationStructure, sizeof(uint64_t), &blas.handle));
	}

	/*
		The top level acceleration structure contains the scene's object instances
	*/
	void createTopLevelAccelerationStructure(uint32_t instanceCount)
	{
		topLevelAS.buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
		topLevelAS.buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
		topLevelAS.buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
		topLevelAS.buildInfo.instanceCount = instanceCount;
		topLevelAS.buildInfo.geometryCount = 0;

		VkAccelerationStructureCreateInfoNV accelerationStructureCI{};
		accelerationStructureCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
		accelerationStructureCI.info  = topLevelAS.buildInfo;
		VK_CHECK_RESULT(vkCreateAccelerationStructureNV(device, &accelerationStructureCI, nullptr, &topLevelAS.accelerationStructure));

		VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo{};
		memoryRequirementsInfo.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
		memoryRequirementsInfo.type                  = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
		memoryRequirementsInfo.accelerationStructure = topLevelAS.accelerationStructure;

		VkMemoryRequirements2 memoryRequirements2{};
		vkGetAccelerationStructureMemoryRequirementsNV(device, &memoryRequirementsInfo, &memoryRequirements2);
		printf("createTopLevelAccelerationStructure mem %ld\n", memoryRequirements2.memoryRequirements.size);

		VkMemoryAllocateInfo memoryAllocateInfo = vks::initializers::memoryAllocateInfo();
		memoryAllocateInfo.allocationSize       = memoryRequirements2.memoryRequirements.size;
		memoryAllocateInfo.memoryTypeIndex      = vulkanDevice->getMemoryType(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &topLevelAS.memory));

		VkBindAccelerationStructureMemoryInfoNV accelerationStructureMemoryInfo{};
		accelerationStructureMemoryInfo.sType                 = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
		accelerationStructureMemoryInfo.accelerationStructure = topLevelAS.accelerationStructure;
		accelerationStructureMemoryInfo.memory                = topLevelAS.memory;
		VK_CHECK_RESULT(vkBindAccelerationStructureMemoryNV(device, 1, &accelerationStructureMemoryInfo));

		VK_CHECK_RESULT(vkGetAccelerationStructureHandleNV(device, topLevelAS.accelerationStructure, sizeof(uint64_t), &topLevelAS.handle));
	}

	static VkGeometryNV createVkGeometryNV(const ObjModel &object)
	{
		VkGeometryNV geometry{};
		geometry.sType                              = VK_STRUCTURE_TYPE_GEOMETRY_NV;
		geometry.geometryType                       = VK_GEOMETRY_TYPE_TRIANGLES_NV;
		geometry.geometry.triangles.sType           = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV;
		geometry.geometry.triangles.vertexData      = object.vertexBuffer.buffer;
		geometry.geometry.triangles.vertexOffset    = 0;
		geometry.geometry.triangles.vertexCount     = object.nbVertices;
		geometry.geometry.triangles.vertexStride    = sizeof(Vertex);
		geometry.geometry.triangles.vertexFormat    = VK_FORMAT_R32G32B32_SFLOAT;
		geometry.geometry.triangles.indexData       = object.indexBuffer.buffer;
		geometry.geometry.triangles.indexOffset     = 0;
		geometry.geometry.triangles.indexCount      = object.nbIndices;
		geometry.geometry.triangles.indexType       = VK_INDEX_TYPE_UINT32;
		geometry.geometry.triangles.transformData   = VK_NULL_HANDLE;
		geometry.geometry.triangles.transformOffset = 0;
		geometry.geometry.aabbs                     = {};
		geometry.geometry.aabbs.sType               = {VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV};
		geometry.flags                              = VK_GEOMETRY_OPAQUE_BIT_NV;
		return geometry;
	}

	VkDeviceSize getScratchSize(AccelerationStructure &as, bool update = false)
	{
		// Estimate the amount of scratch memory required to build the BLAS, and update the
		// size of the scratch buffer that will be allocated to sequentially build all BLASes
		VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV};
		memoryRequirementsInfo.type                  = update ? VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_UPDATE_SCRATCH_NV : VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
		memoryRequirementsInfo.accelerationStructure = as.accelerationStructure;

		VkMemoryRequirements2 reqMem{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
		vkGetAccelerationStructureMemoryRequirementsNV(device, &memoryRequirementsInfo, &reqMem);
		return reqMem.memoryRequirements.size;
	}

	static void rwBarrierAS(VkCommandBuffer cmdBuffer)
	{
		VkMemoryBarrier memoryBarrier = vks::initializers::memoryBarrier();
		memoryBarrier.srcAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
		memoryBarrier.dstAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
		vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
	}

	bool culling_enabled = false;

	GeometryInstance createGeometryInstance(const std::pair<uint32_t, MyObj> &obj, const float ray_time = 0.0f) const
	{
		printf("createGeometryInstance %d\n", obj.first);
		GeometryInstance geometryInstance{};
		geometryInstance.transform                   = transform_at_time(obj.second.instance.transforms, ray_time);
		geometryInstance.instanceId                  = obj.first;
		geometryInstance.mask                        = 0xff;
		geometryInstance.instanceOffset              = 0;
		geometryInstance.flags                       = culling_enabled ? 0 : VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV;
		geometryInstance.accelerationStructureHandle = obj.second.blas.handle;
		return geometryInstance;
	}

	void createOrUpdateBlas(AccelerationStructure &blas, VkGeometryNV &geom, bool update = false)
	{
		if (update)
			blas.buildInfo.pGeometries = &geom;
		else
			createBottomLevelAccelerationStructure(blas, &geom);

		const auto  blasScratchSize = getScratchSize(blas, update);
		vks::Buffer scratchBuffer;
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    &scratchBuffer,
		    blasScratchSize));

		VkCommandBuffer cmdBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

		if (update)
		{
			// Update the acceleration structure. Note the VK_TRUE parameter to trigger the update,
			// and the existing BLAS being passed and updated in place
			vkCmdBuildAccelerationStructureNV(cmdBuffer, &blas.buildInfo, VK_NULL_HANDLE, 0, VK_TRUE, blas.accelerationStructure, blas.accelerationStructure,
			                                  scratchBuffer.buffer, 0);
		}
		else
		{
			vkCmdBuildAccelerationStructureNV(
			    cmdBuffer,
			    &blas.buildInfo,
			    VK_NULL_HANDLE,
			    0,
			    VK_FALSE,
			    blas.accelerationStructure,
			    VK_NULL_HANDLE,
			    scratchBuffer.buffer,
			    0);
		}

		rwBarrierAS(cmdBuffer);
		vulkanDevice->flushCommandBuffer(cmdBuffer, queue);
		scratchBuffer.destroy();
	}

	void updateTlas(const float ray_time = 0.0f)
	{
		buildTlas(true, ray_time);
	}

	void buildTlas(bool update = false, const float ray_time = 0.0f)
	{
		vks::Buffer instanceBuffer;

		std::vector<GeometryInstance> geometryInstances;
		std::transform(objects.begin(), objects.end(), std::back_inserter(geometryInstances), [this, ray_time](const std::pair<uint32_t, MyObj> &obj) {
			return createGeometryInstance(obj, ray_time);
		});

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		    &instanceBuffer,
		    geometryInstances.size() * sizeof(GeometryInstance),
		    geometryInstances.data()));

		if (!update)
			createTopLevelAccelerationStructure(geometryInstances.size());

		const auto  tlasScratchSize = getScratchSize(topLevelAS, update);
		vks::Buffer scratchBuffer;
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    &scratchBuffer,
		    tlasScratchSize));
		VkCommandBuffer cmdBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		rwBarrierAS(cmdBuffer);

		if (update)
		{
			// Update the acceleration structure. Note the VK_TRUE parameter to trigger the update,
			// and the existing TLAS being passed and updated in place
			vkCmdBuildAccelerationStructureNV(cmdBuffer, &topLevelAS.buildInfo, instanceBuffer.buffer, 0, VK_TRUE, topLevelAS.accelerationStructure,
			                                  topLevelAS.accelerationStructure, scratchBuffer.buffer, 0);
		}
		else
		{
			vkCmdBuildAccelerationStructureNV(
			    cmdBuffer,
			    &topLevelAS.buildInfo,
			    instanceBuffer.buffer,
			    0,
			    VK_FALSE,
			    topLevelAS.accelerationStructure,
			    VK_NULL_HANDLE,
			    scratchBuffer.buffer,
			    0);
		}

		vulkanDevice->flushCommandBuffer(cmdBuffer, queue);

		scratchBuffer.destroy();
		instanceBuffer.destroy();
	}

	VkDeviceSize copyShaderIdentifier(uint8_t *data, const uint8_t *shaderHandleStorage, uint32_t groupIndex) const
	{
		const uint32_t shaderGroupHandleSize = rayTracingProperties.shaderGroupHandleSize;
		memcpy(data, shaderHandleStorage + groupIndex * shaderGroupHandleSize, shaderGroupHandleSize);
		return shaderGroupHandleSize;
	}

	/*
		Create the Shader Binding Table that binds the programs and top-level acceleration structure
	*/
	void createShaderBindingTable()
	{
		// Create buffer for the shader binding table
		const uint32_t sbtSize = rayTracingProperties.shaderGroupHandleSize * NUM_SHADER_GROUPS;
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
		    &shaderBindingTable,
		    sbtSize));
		shaderBindingTable.map();

		auto shaderHandleStorage = new uint8_t[sbtSize];
		// Get shader identifiers
		VK_CHECK_RESULT(vkGetRayTracingShaderGroupHandlesNV(device, pipeline, 0, NUM_SHADER_GROUPS, sbtSize, shaderHandleStorage));
		auto *data = static_cast<uint8_t *>(shaderBindingTable.mapped);
		// Copy the shader identifiers to the shader binding table
		data += copyShaderIdentifier(data, shaderHandleStorage, INDEX_RAYGEN);
		data += copyShaderIdentifier(data, shaderHandleStorage, INDEX_MISS);
		data += copyShaderIdentifier(data, shaderHandleStorage, INDEX_CLOSEST_HIT);
		shaderBindingTable.unmap();
	}

	/*
		Create the descriptor sets used for the ray tracing dispatch
	*/
	void createDescriptorSets()
	{
		std::vector<VkDescriptorPoolSize> poolSizes = {
		    {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV, 1},
		    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
		    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
		    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<uint32_t>(1 + objects.size() + objects.size())}};
		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 1);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool));

		VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));

		VkWriteDescriptorSetAccelerationStructureNV descriptorAccelerationStructureInfo{};
		descriptorAccelerationStructureInfo.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV;
		descriptorAccelerationStructureInfo.accelerationStructureCount = 1;
		descriptorAccelerationStructureInfo.pAccelerationStructures    = &topLevelAS.accelerationStructure;

		VkWriteDescriptorSet accelerationStructureWrite{};
		accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		// The specialized acceleration structure descriptor has to be chained
		accelerationStructureWrite.pNext           = &descriptorAccelerationStructureInfo;
		accelerationStructureWrite.dstSet          = descriptorSet;
		accelerationStructureWrite.dstBinding      = 0;
		accelerationStructureWrite.descriptorCount = 1;
		accelerationStructureWrite.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;

		VkDescriptorImageInfo storageImageDescriptor{};
		storageImageDescriptor.imageView   = storageImage.view;
		storageImageDescriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		VkWriteDescriptorSet resultImageWrite   = vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &storageImageDescriptor);
		VkWriteDescriptorSet uniformBufferWrite = vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2, &ubo.descriptor);

		VkDescriptorBufferInfo bufferDescriptor     = {deviceBuffer.buffer, 0, VK_WHOLE_SIZE};
		VkWriteDescriptorSet   storageBufferDescSet = vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &bufferDescriptor);

		std::vector<VkDescriptorBufferInfo> vertexBufferDescInfos{};
		std::vector<VkDescriptorBufferInfo> indexBufferDescInfos{};

		for (auto &obj : objects)
		{
			VkDescriptorBufferInfo vertexBufferDesc{obj.second.model.vertexBuffer.buffer, 0, VK_WHOLE_SIZE};
			VkDescriptorBufferInfo indexBufferDesc{obj.second.model.indexBuffer.buffer, 0, VK_WHOLE_SIZE};
			vertexBufferDescInfos.emplace_back(vertexBufferDesc);
			indexBufferDescInfos.emplace_back(indexBufferDesc);
		}

		VkWriteDescriptorSet vertexBufferWrite = vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, vertexBufferDescInfos.data(), vertexBufferDescInfos.size());
		VkWriteDescriptorSet indexBufferWrite  = vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5, indexBufferDescInfos.data(), indexBufferDescInfos.size());

		std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
		    accelerationStructureWrite,
		    resultImageWrite,
		    uniformBufferWrite,
		    storageBufferDescSet,
		    vertexBufferWrite,
		    indexBufferWrite};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, VK_NULL_HANDLE);
	}

	/*
		Create our ray tracing pipeline
	*/
	void createRayTracingPipeline()
	{
		auto objectCount = static_cast<uint32_t>(objects.size());

		VkDescriptorSetLayoutBinding accelerationStructureLB{0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV, 1, VK_SHADER_STAGE_RAYGEN_BIT_NV};
		VkDescriptorSetLayoutBinding resultImageLB{1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_NV};
		VkDescriptorSetLayoutBinding uniformBufferLB{2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_NV};
		VkDescriptorSetLayoutBinding storageBufferLB{3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_NV};
		VkDescriptorSetLayoutBinding vertexBufferLB{4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, objectCount, VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV};
		VkDescriptorSetLayoutBinding indexBufferLB{5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, objectCount, VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV};

		std::vector<VkDescriptorSetLayoutBinding> bindings({accelerationStructureLB,
		                                                    resultImageLB,
		                                                    uniformBufferLB,
		                                                    storageBufferLB,
		                                                    vertexBufferLB,
		                                                    indexBufferLB});

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings    = bindings.data();
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout));

		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
		pipelineLayoutCreateInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutCreateInfo.setLayoutCount = 1;
		pipelineLayoutCreateInfo.pSetLayouts    = &descriptorSetLayout;

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

		const uint32_t shaderIndexRaygen     = 0;
		const uint32_t shaderIndexMiss       = 1;
		const uint32_t shaderIndexClosestHit = 2;

		std::array<VkPipelineShaderStageCreateInfo, 3> shaderStages{};
		shaderStages[shaderIndexRaygen]     = loadShader(getShadersPath() + "nv_ray_tracing_basic/raygen.rgen.spv", VK_SHADER_STAGE_RAYGEN_BIT_NV);
		shaderStages[shaderIndexMiss]       = loadShader(getShadersPath() + "nv_ray_tracing_basic/miss.rmiss.spv", VK_SHADER_STAGE_MISS_BIT_NV);
		shaderStages[shaderIndexClosestHit] = loadShader(getShadersPath() + "nv_ray_tracing_basic/closesthit.rchit.spv", VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV);

		/*
			Setup ray tracing shader groups
		*/
		std::array<VkRayTracingShaderGroupCreateInfoNV, NUM_SHADER_GROUPS> groups{};
		for (auto &group : groups)
		{
			// Init all groups with some default values
			group.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
			group.generalShader      = VK_SHADER_UNUSED_NV;
			group.closestHitShader   = VK_SHADER_UNUSED_NV;
			group.anyHitShader       = VK_SHADER_UNUSED_NV;
			group.intersectionShader = VK_SHADER_UNUSED_NV;
		}

		// Links shaders and types to ray tracing shader groups
		groups[INDEX_RAYGEN].type                  = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
		groups[INDEX_RAYGEN].generalShader         = shaderIndexRaygen;
		groups[INDEX_MISS].type                    = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
		groups[INDEX_MISS].generalShader           = shaderIndexMiss;
		groups[INDEX_CLOSEST_HIT].type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_NV;
		groups[INDEX_CLOSEST_HIT].generalShader    = VK_SHADER_UNUSED_NV;
		groups[INDEX_CLOSEST_HIT].closestHitShader = shaderIndexClosestHit;

		VkRayTracingPipelineCreateInfoNV rayPipelineInfo{};
		rayPipelineInfo.sType             = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV;
		rayPipelineInfo.stageCount        = static_cast<uint32_t>(shaderStages.size());
		rayPipelineInfo.pStages           = shaderStages.data();
		rayPipelineInfo.groupCount        = static_cast<uint32_t>(groups.size());
		rayPipelineInfo.pGroups           = groups.data();
		rayPipelineInfo.maxRecursionDepth = 1;
		rayPipelineInfo.layout            = pipelineLayout;
		VK_CHECK_RESULT(vkCreateRayTracingPipelinesNV(device, VK_NULL_HANDLE, 1, &rayPipelineInfo, nullptr, &pipeline));
	}

	/*
		Create the uniform buffer used to pass matrices to the ray tracing ray generation shader
	*/
	void createUniformBuffer()
	{
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		    &ubo,
		    sizeof(uniformData),
		    &uniformData));
		VK_CHECK_RESULT(ubo.map());
	}

	/*
		Command buffer generation
	*/
	void buildCommandBuffers()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkImageSubresourceRange subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			/*
				Dispatch the ray tracing commands
			*/
			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipeline);
			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

			// Calculate shader binding offsets, which is pretty straight forward in our example
			VkDeviceSize bindingOffsetRayGenShader = rayTracingProperties.shaderGroupHandleSize * INDEX_RAYGEN;
			VkDeviceSize bindingOffsetMissShader   = rayTracingProperties.shaderGroupHandleSize * INDEX_MISS;
			VkDeviceSize bindingOffsetHitShader    = rayTracingProperties.shaderGroupHandleSize * INDEX_CLOSEST_HIT;
			VkDeviceSize bindingStride             = rayTracingProperties.shaderGroupHandleSize;

			vkCmdTraceRaysNV(drawCmdBuffers[i],
			                 shaderBindingTable.buffer, bindingOffsetRayGenShader,
			                 shaderBindingTable.buffer, bindingOffsetMissShader, bindingStride,
			                 shaderBindingTable.buffer, bindingOffsetHitShader, bindingStride,
			                 VK_NULL_HANDLE, 0, 0,
			                 width, height, 1);

			/*
				Copy raytracing output to swap chain image
			*/

			// Prepare current swapchain image as transfer destination
			vks::tools::setImageLayout(
			    drawCmdBuffers[i],
			    swapChain.images[i],
			    VK_IMAGE_LAYOUT_UNDEFINED,
			    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			    subresourceRange);

			// Prepare ray tracing output image as transfer source
			vks::tools::setImageLayout(
			    drawCmdBuffers[i],
			    storageImage.image,
			    VK_IMAGE_LAYOUT_GENERAL,
			    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			    subresourceRange);

			VkImageCopy copyRegion{};
			copyRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
			copyRegion.srcOffset      = {0, 0, 0};
			copyRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
			copyRegion.dstOffset      = {0, 0, 0};
			copyRegion.extent         = {width, height, 1};
			vkCmdCopyImage(drawCmdBuffers[i], storageImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapChain.images[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

			// Transition swap chain image back for presentation
			vks::tools::setImageLayout(
			    drawCmdBuffers[i],
			    swapChain.images[i],
			    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			    subresourceRange);

			// Transition ray tracing output image back to general layout
			vks::tools::setImageLayout(
			    drawCmdBuffers[i],
			    storageImage.image,
			    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			    VK_IMAGE_LAYOUT_GENERAL,
			    subresourceRange);

			//@todo: Default render pass setup will overwrite contents
			//vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
			//drawUI(drawCmdBuffers[i]);
			//vkCmdEndRenderPass(drawCmdBuffers[i]);
			{
				// Barrier to ensure that shader writes are finished before buffer is read back from GPU
				VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
				bufferBarrier.srcAccessMask         = VK_ACCESS_SHADER_WRITE_BIT;
				bufferBarrier.dstAccessMask         = VK_ACCESS_TRANSFER_READ_BIT;
				bufferBarrier.buffer                = deviceBuffer.buffer;
				bufferBarrier.size                  = VK_WHOLE_SIZE;

				vkCmdPipelineBarrier(
				    drawCmdBuffers[i],
				    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
				    VK_PIPELINE_STAGE_TRANSFER_BIT,
				    VK_FLAGS_NONE,
				    0, nullptr,
				    1, &bufferBarrier,
				    0, nullptr);
			}
			// Read back to host visible buffer
			VkBufferCopy copyBufferRegion = {0, 0, deviceBuffer.size};
			vkCmdCopyBuffer(drawCmdBuffers[i], deviceBuffer.buffer, hostBuffer.buffer, 1, &copyBufferRegion);

			// Barrier to ensure that buffer copy is finished before host reading from it
			VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
			bufferBarrier.srcAccessMask         = VK_ACCESS_TRANSFER_WRITE_BIT;
			bufferBarrier.dstAccessMask         = VK_ACCESS_HOST_READ_BIT;
			bufferBarrier.buffer                = hostBuffer.buffer;
			bufferBarrier.size                  = VK_WHOLE_SIZE;

			vkCmdPipelineBarrier(
			    drawCmdBuffers[i],
			    VK_PIPELINE_STAGE_TRANSFER_BIT,
			    VK_PIPELINE_STAGE_HOST_BIT,
			    VK_FLAGS_NONE,
			    0, nullptr,
			    1, &bufferBarrier,
			    0, nullptr);

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	ObjModel createObject(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices)
	{
		ObjModel model;
		model.nbIndices  = static_cast<uint32_t>(indices.size());
		model.nbVertices = static_cast<uint32_t>(vertices.size());
		updateVertices(model, vertices);
		updateIndices(model, indices);
		return model;
	}

	MyObj createMyObj(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices, const glm::mat3x4 &transform = glm::mat3x4{1})
	{
		MyObj obj{};
		obj.model               = createObject(vertices, indices);
		obj.geom                = createVkGeometryNV(obj.model);
		obj.instance.transforms = {transform};
		return obj;
	}

	void updateVertices(ObjModel &obj, const std::vector<Vertex> &vertices)
	{
		vks::Buffer stagingBuffer;

		auto bufferSize = vertices.size() * sizeof(Vertex);
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		    &stagingBuffer,
		    bufferSize,
		    vertices.data()));

		if (!obj.vertexBuffer.buffer)
		{
			VK_CHECK_RESULT(vulkanDevice->createBuffer(
			    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			    &obj.vertexBuffer,
			    bufferSize));
		}
		else
		{
			//update only works if number of vertices did not change
			assert(vertices.size() == obj.nbVertices);
		}

		VkBufferCopy region{0, 0, bufferSize};
		vulkanDevice->copyBuffer(&stagingBuffer, &obj.vertexBuffer, queue, &region);
		stagingBuffer.destroy();
	}

	void updateIndices(ObjModel &obj, const std::vector<uint32_t> &indices)
	{
		vks::Buffer stagingBuffer;

		auto bufferSize = indices.size() * sizeof(uint32_t);
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		    &stagingBuffer,
		    bufferSize,
		    indices.data()));

		if (!obj.indexBuffer.buffer)
		{
			VK_CHECK_RESULT(vulkanDevice->createBuffer(
			    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			    &obj.indexBuffer,
			    bufferSize));
		}
		else
		{
			//update only works if number of indices did not change
			assert(indices.size() == obj.nbIndices);
		}
		VkBufferCopy region{0, 0, bufferSize};
		vulkanDevice->copyBuffer(&stagingBuffer, &obj.indexBuffer, queue, &region);
		stagingBuffer.destroy();
	}

	bool     updates_enabled = false;
	uint32_t frameNumber     = 0;
	int64_t  last_update_sec = -1;
	int64_t  start           = current_time_msec() / 1000;

	static std::vector<Hit> get_valid_hits(Hit *hits, uint32_t raycount)
	{
		std::vector<Hit> valid_hits{};
		for (int i = 0; i < raycount; i++)
		{
			auto hit = &hits[i];
			if (hit->valid)
			{
				valid_hits.push_back(hits[i]);
			}
		}
		return valid_hits;
	}

	Hit *draw()
	{
		int64_t start_frame = current_time_msec();
		prepareFrame();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers    = &drawCmdBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
		submitFrame();
		frameNumber++;
		int64_t sec = current_time_msec() / 1000;
		if (updates_enabled && last_update_sec != sec)
		{
			last_update_sec = sec;
			auto elapsed    = sec - start;
			if (elapsed <= 5)
			{
				//update vertices
				vertices0[0].pos[2] -= 0.1f;
				printf("%d %ld update instance transform\n", frameNumber, elapsed);
				auto &obj0 = objects.at(0);
				updateVertices(obj0.model, vertices0);
				obj0.geom = createVkGeometryNV(obj0.model);
				createOrUpdateBlas(obj0.blas, obj0.geom, true);

				//update instance transform
				auto &obj1 = objects.at(1);
				obj1.instance.transforms[0][0] += 0.01f;
				obj1.instance.transforms[1][1] += 0.01f;
				updateTlas();
			}
			else
			{
				if (objects.find(2) == objects.end())
				{
					//add
					printf("create object\n");
					objects.emplace(std::make_pair(2, createMyObj(vertices2, indices2)));
					auto &obj0 = objects.at(2);
					createOrUpdateBlas(obj0.blas, obj0.geom);
					buildTlas();
					createDescriptorSets();
					destroyCommandBuffers();
					createCommandBuffers();
					buildCommandBuffers();
					vkDeviceWaitIdle(device);
				}
				if (elapsed == 10 && objects.find(1) != objects.end())
				{
					// delete
					printf("delete object\n");
					auto &obj = objects.at(1);
					destroyObjectInternal(obj);
					objects.erase(1);
					buildTlas();
					createDescriptorSets();
					destroyCommandBuffers();
					createCommandBuffers();
					buildCommandBuffers();
					vkDeviceWaitIdle(device);
				}
			}
		}

		// Make device writes visible to the host
		if (hostBuffer.mapped == nullptr)
		{
			hostBuffer.map(hostBuffer.size);
		}
		hostBuffer.invalidate(hostBuffer.size);        //Invalidate a memory range of the buffer to make it visible to the host

		//		std::vector<HitPy> computeOutput(width * height, HitPy{});
		//		memcpy(computeOutput.data(), hostBuffer.mapped, hostBuffer.size);
		// Copy to output
		Hit *computeOutput = static_cast<Hit *>(hostBuffer.mapped);
		return computeOutput;
		//		hostBuffer.flush(hostBuffer.size);        //make writes visible to device
		//		hostBuffer.unmap();
		//		VkBufferCopy copyRegion = {0, 0, hostBuffer.size};
		//		vulkanDevice->copyBuffer(&hostBuffer, &deviceBuffer, queue, &copyRegion);

		//		long time = current_time_msec() - start_frame;
		//		printf("rays %d %ldmsec hit0 valid: %d lidar %d inst %d prim %d point (%f, %f, %f) dist %f norm (%f,%f,%f)\n", cnt, time, hit0.valid, hit0.lidar_id, hit0.instID, hit0.primID, hit0.point.x, hit0.point.y, hit0.point.z, hit0.distance, hit0.normal.x, hit0.normal.y, hit0.normal.z);
	}

	/** @brief Setup the vulkan instance, enable required extensions and connect to the physical device (GPU) */
	bool initVulkan()
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

	xcb_window_t setupWindow()
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

		std::string result;
		result                  = title + " - " + deviceProperties.deviceName;
		std::string windowTitle = result;
		xcb_change_property(connection, XCB_PROP_MODE_REPLACE,
		                    window, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8,
		                    windowTitle.size(), windowTitle.c_str());

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
	void prepare()
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

		// Query the ray tracing properties of the current implementation, we will need them later on
		rayTracingProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES_NV;
		VkPhysicalDeviceProperties2 deviceProps2{};
		deviceProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		deviceProps2.pNext = &rayTracingProperties;
		vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProps2);

		// Get VK_NV_ray_tracing related function pointers
		vkCreateAccelerationStructureNV                = reinterpret_cast<PFN_vkCreateAccelerationStructureNV>(vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureNV"));
		vkDestroyAccelerationStructureNV               = reinterpret_cast<PFN_vkDestroyAccelerationStructureNV>(vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureNV"));
		vkBindAccelerationStructureMemoryNV            = reinterpret_cast<PFN_vkBindAccelerationStructureMemoryNV>(vkGetDeviceProcAddr(device, "vkBindAccelerationStructureMemoryNV"));
		vkGetAccelerationStructureHandleNV             = reinterpret_cast<PFN_vkGetAccelerationStructureHandleNV>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureHandleNV"));
		vkGetAccelerationStructureMemoryRequirementsNV = reinterpret_cast<PFN_vkGetAccelerationStructureMemoryRequirementsNV>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureMemoryRequirementsNV"));
		vkCmdBuildAccelerationStructureNV              = reinterpret_cast<PFN_vkCmdBuildAccelerationStructureNV>(vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructureNV"));
		vkCreateRayTracingPipelinesNV                  = reinterpret_cast<PFN_vkCreateRayTracingPipelinesNV>(vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesNV"));
		vkGetRayTracingShaderGroupHandlesNV            = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesNV>(vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesNV"));
		vkCmdTraceRaysNV                               = reinterpret_cast<PFN_vkCmdTraceRaysNV>(vkGetDeviceProcAddr(device, "vkCmdTraceRaysNV"));

		createStorageImage();
		createUniformBuffer();
		createStorageBuffer();
	}

	void handleEvent(const xcb_generic_event_t *event)
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
			default:
				break;
		}
	}

	void uploadRays(const std::vector<Ray> &rays_)
	{
		const auto         maxRays    = width * height;
		const VkDeviceSize bufferSize = maxRays * sizeof(Ray);
		if (hostBuffer.mapped == nullptr)
		{
			VK_CHECK_RESULT(hostBuffer.map());
		}
		assert(hostBuffer.mapped);

		Ray *computeInput = (Ray *) hostBuffer.mapped;

		for (size_t i = 0; i < maxRays; i++)
		{
			computeInput[i] = rays_[i % rays_.size()];
		}
		hostBuffer.flush();

		VkBufferCopy copyRegion = {0, 0, bufferSize};
		vulkanDevice->copyBuffer(&hostBuffer, &deviceBuffer, queue, &copyRegion);
	}

	void destroyTlas()
	{
		if (topLevelAS.memory != nullptr)
			vkFreeMemory(device, topLevelAS.memory, nullptr);
		if (topLevelAS.accelerationStructure != nullptr)
			vkDestroyAccelerationStructureNV(device, topLevelAS.accelerationStructure, nullptr);

		topLevelAS = {};
	}

	void destroyDescPool()
	{
		if (descriptorPool != VK_NULL_HANDLE)
		{
			vkDestroyDescriptorPool(device, descriptorPool, nullptr);
			descriptorPool = VK_NULL_HANDLE;
		}
	}

  public:
	VulkanExample()
	{
		// Check for a valid asset path
		struct stat info
		{};
		if (stat(getAssetPath().c_str(), &info) != 0)
		{
			std::cerr << "Error: Could not find asset path in " << getAssetPath() << "\n";
			exit(-1);
		}

		initxcbConnection();

		assert(sizeof(Hit) == 64);
		assert(sizeof(Ray) == 64);

		// Enable instance and device extensions required to use VK_NV_ray_tracing
		enabledInstanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
		enabledDeviceExtensions.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
		enabledDeviceExtensions.push_back(VK_NV_RAY_TRACING_EXTENSION_NAME);
	}

	~VulkanExample()
	{
		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		vkDestroyImageView(device, storageImage.view, nullptr);
		vkDestroyImage(device, storageImage.image, nullptr);
		vkFreeMemory(device, storageImage.memory, nullptr);
		destroyTlas();

		for (auto &obj : objects)
		{
			destroyObjectInternal(obj.second);
		}
		shaderBindingTable.destroy();
		ubo.destroy();

		// Clean up Vulkan resources
		swapChain.cleanup();
		destroyDescPool();
		destroyCommandBuffers();
		vkDestroyRenderPass(device, renderPass, nullptr);
		for (auto &frameBuffer : frameBuffers)
		{
			vkDestroyFramebuffer(device, frameBuffer, nullptr);
		}

		for (auto &shaderModule : shaderModules)
		{
			vkDestroyShaderModule(device, shaderModule.second, nullptr);
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

	void update_animated_object(const uint32_t id, const std::vector<Vertex> &vertices, const glm::mat3x4 &transform, const glm::vec3 &isovelocity = glm::vec3{0})
	{
		assert(objects.count(id) == 1);
		auto &obj = objects.at(id);
		assert(vertices.size() == obj.model.nbVertices);        //cant increase or decrease vertex count
		updateVertices(obj.model, vertices);
		obj.geom = createVkGeometryNV(obj.model);
		createOrUpdateBlas(obj.blas, obj.geom, true);
		update_object(id, transform, isovelocity);
	}

	void add_object(const uint32_t id, const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices)
	{
		assert(objects.count(id) == 0);
		objects.emplace(std::make_pair(id, createMyObj(vertices, indices)));
		auto &obj0 = objects.at(id);
		createOrUpdateBlas(obj0.blas, obj0.geom);

		// adding or deleting an object requires full tlas rebuild
		destroyTlas();
		destroyDescPool();
		destroyCommandBuffers();
		vkDeviceWaitIdle(device);
	}

	void delete_object(const uint32_t id)
	{
		assert(objects.count(id) != 0);
		auto &obj = objects.at(id);
		destroyObjectInternal(obj);
		objects.erase(id);
		destroyTlas();
		destroyDescPool();
		destroyCommandBuffers();
		vkDeviceWaitIdle(device);
	}

	void update_object(const uint32_t id, const glm::mat3x4 &transform, const glm::vec3 &isovelocity = glm::vec3{0})
	{
		assert(objects.count(id) == 1);
		auto &obj = objects.at(id);
		obj.instance.transforms.resize(NUM_TIME_STEPS);

		for (unsigned int t = 0; t < NUM_TIME_STEPS; t++)
		{
			glm::mat3x4 copy = transform;

			// Apply linear velocity for now, we don't have angular velocity supported
			copy[0][3] += (isovelocity[0] * 0.1f * t / NUM_TIME_STEPS);
			copy[1][3] += (isovelocity[1] * 0.1f * t / NUM_TIME_STEPS);
			copy[2][3] += (isovelocity[2] * 0.1f * t / NUM_TIME_STEPS);
			obj.instance.transforms[t] = copy;
		}
	}

	std::vector<Hit> trace_rays(const std::vector<Ray> &rays_)
	{
		if (topLevelAS.accelerationStructure == nullptr)
		{
			buildTlas();
		}
		if (pipeline == VK_NULL_HANDLE)
		{
			createRayTracingPipeline();
			createShaderBindingTable();
		}
		if (descriptorPool == VK_NULL_HANDLE)
		{
			createDescriptorSets();
			createCommandBuffers();
			buildCommandBuffers();
			vkDeviceWaitIdle(device);
		}

		uploadRays(rays_);
		std::set<float> sorted_ray_times;
		for (const auto &ray : rays_)
		{
			sorted_ray_times.emplace(ray.time);
		}
		assert(!sorted_ray_times.empty());

		for (const auto ray_time : sorted_ray_times)
		{
			printf("ray time %f\n", ray_time);
			updateTlas(ray_time);
			uniformData.time = ray_time;
			memcpy(ubo.mapped, &uniformData, sizeof(uniformData));

			const auto &valid_hits_ = get_valid_hits(draw(), rays_.size());
			printf("valid hits at time %f\n", ray_time);
			for (const auto &hit : valid_hits_)
			{
				std::cout << to_string(hit) << std::endl;
			}
		}
		return get_valid_hits((Hit *) hostBuffer.mapped, rays_.size());
	}

	void test_pre_motion_blur()
	{
		glm::mat3x4 transforms = {
		    1.0f, 0.0f, 0.0f, 0.855f,
		    0.0f, 1.0f, 0.0f, 0.0f,
		    0.0f, 0.0f, 1.0f, 10.0f};

		add_object(0, vertices0, indices0);
		update_object(0, transforms);

		auto               valid_hits = trace_rays(rays2);
		std::vector<HitPy> expecteds  = {
            {{1.5, 0.5, 10.0}, {0.0, 0.0, 1.0}, 11.0, 0.5, 0.355f, ignore, 0, 1, 0, true},
        };
		assert_near(expecteds, valid_hits);
	}

	void test_motion_blur()
	{
		add_object(0, vertices0, indices0);
		glm::mat3x4 transform = {
		    1.0f, 0.0f, 0.0f, 0.0f,
		    0.0f, 1.0f, 0.0f, 0.0f,
		    0.0f, 0.0f, 1.0f, 10.0f};
		glm::vec3 isovelocity = {10, 0, 0};
		update_object(0, transform, isovelocity);
		std::cout << glm::to_string(objects.at(0).instance.transforms[0]) << std::endl;

		auto               valid_hits       = trace_rays(rays);
		std::vector<HitPy> expecteds_before = {
		    {{0.5, 0.5, 10.0}, {0.0, 0.0, -1.0}, 11.0, 0.5, 0.5, ignore, 0, 0, 0, true},
		    {{0.0, 0.0, 10.0}, {0.0, 0.0, -1.0}, 10.0, 0.0, 0.0, ignore, 0, 0, 0, true},
		};
		assert_near(expecteds_before, valid_hits);

		valid_hits                   = trace_rays(rays2);
		std::vector<HitPy> expecteds = {
		    {{0.5, 0.5, 10.0}, {0.0, 0.0, -1.0}, 11.0, 0.5, 0.5f, ignore, 0, 0, 0, true},
		    {{0.0, 0.0, 10.0}, {0.0, 0.0, -1.0}, 10.0, -0.0, -0.0f, ignore, 0, 0, 0, true},
		    {{1.5, 0.5, 10.0}, {0.0, 0.0, 1.0}, 11.0, 0.5, 0.355f, ignore, 0, 1, 0, true},
		};
		assert_near(expecteds, valid_hits);

		update_object(0, transform, glm::vec3{0});

		valid_hits                    = trace_rays(rays2);
		std::vector<HitPy> expecteds2 = {
		    {{0.5, 0.5, 10.0}, {0.0, 0.0, -1.0}, 11.0, 0.5, 0.5, ignore, 0, 0, 0, true},
		    {{0.0, 0.0, 10.0}, {0.0, 0.0, -1.0}, 10.0, -0.0, -0.0, ignore, 0, 0, 0, true},
		};
		assert_near(expecteds2, valid_hits);
	}

	void test_add_two_objects_and_transform2()
	{
		culling_enabled         = true;
		glm::mat3x4 transforms0 = {
		    1.0f, 0.0f, 0.0f, 0.0f,
		    0.0f, 1.0f, 0.0f, 0.0f,
		    0.0f, 0.0f, 1.0f, -10.0f};
		glm::mat3x4 transforms1 = {
		    0.1f, 0.0f, 0.0f, 0.0f,
		    0.0f, 0.1f, 0.0f, 0.0f,
		    0.0f, 0.0f, 0.1f, 0.0f};
		add_object(0, vertices0, indices0);
		add_object(1, vertices1, indices1);
		update_object(0, transforms0);
		update_object(1, transforms1);

		auto valid_hits = trace_rays(rays);
		// clang-format off
        std::vector<HitPy> expecteds = {
            {{0.000001f, 0.00, 0.05f}, {-0.0, 0.0, 1.0}, 1.95f, 0.49999f, 0.00001f, ignore, 1, 11, 0, true},
            {{0.000000,  0.05f,0.00f}, { 0.0, 1.0,-0.0}, 1.95f, 0.500000, 0.500000, ignore, 1, 6, 0, true},
        };
		// clang-format on
		assert_near(expecteds, valid_hits);

		culling_enabled = false;

		//delete
		delete_object(1);

		auto               valid_hits2 = trace_rays(rays);
		std::vector<HitPy> expecteds2  = {
            {{0.000001f, 0.0, -10.0}, {0.0, 0.0, -1.0}, 12.0, 0.0, 0.000001f, ignore, 0, 0, 0, true}};
		assert_near(expecteds2, valid_hits2);
	}

	void test_animated_mesh()
	{
		glm::mat3x4 transforms = {
		    1.0f, 0.0f, 0.0f, 0.0f,
		    0.0f, 1.0f, 0.0f, 0.0f,
		    0.0f, 0.0f, 1.0f, 10.0f};
		add_object(0, vertices0, indices0);
		update_object(0, transforms);

		auto               valid_hits = trace_rays(rays);
		std::vector<HitPy> expecteds  = {
            {{0.5, 0.5, 10.0}, {0.0, 0.0, -1.0}, 11.0, 0.5, 0.5, ignore, 0, 0, 0, true},
            {{0.0, 0.0, 10.0}, {0.0, 0.0, -1.0}, 10.0, -0.0, -0.0, ignore, 0, 0, 0, true},
        };
		assert_near(expecteds, valid_hits);

		// clang-format off
        std::vector<Vertex> vertices_animated{
            {-1.0f,-1.0f,  0.0f, 0.0f},
            {-1.0f, 1.0f,  0.0f, 0.0f},
            {0.25f, 0.25f, 0.0f, 0.0f},
            {0.25f, 0.25f, 1.0f, 0.0f},
        };
		// clang-format on
		glm::vec3 isovelocity_animated = {0, 0, 0};
		update_animated_object(0, vertices_animated, transforms, isovelocity_animated);

		auto               hits_animated = trace_rays(rays);
		std::vector<HitPy> expecteds2    = {
            {{0.0f, 0.0f, 10.0f}, {0.0f, 0.0f, -2.5f}, 10.0f, -0.0f, 0.8f, ignore, 0, 0, 0, true}};
		assert_near(expecteds2, hits_animated);
	}

	void test_add_two_objects()
	{
		add_object(0, vertices0, indices0);
		add_object(1, vertices1, indices1);

		auto valid_hits = trace_rays(rays);

		// clang-format off
        std::vector<HitPy> expecteds = {
            {{0.000001f,0.0, 0.50}, {-0.0, 0.0, 1.0}, 1.5, 0.5, 0.000001f, ignore, 1, 11, 0, true},
            {{0.000000, 0.5, 0.00}, { 0.0, 1.0,-0.0}, 1.5, 0.5, 0.5, ignore, 1, 6, 0, true},
            {{0.500000, 0.0, 0.00}, { 1.0, 0.0, 0.0}, 0.5, 0.5, 0.5, ignore, 1, 0, 0, true},
            {{0.500000, 0.5,-0.50}, { 0.0, 0.0,-1.0}, 0.5,-0.0, 1.0, ignore, 1, 9, 0, true},
            {{0.500000, 0.0, 0.00}, { 1.0, 0.0, 0.0}, 0.5, 0.5, 0.5, ignore, 1, 0, 0, true},
            {{0.000000, 0.5, 0.00}, { 0.0, 1.0,-0.0}, 0.5, 0.5, 0.5, ignore, 1, 6, 0, true},
            {{0.000000, 0.0, 0.50}, { 0.0, 0.0, 1.0}, 0.5, 0.5, 0.5, ignore, 1, 10, 0, true},
        };
		// clang-format on
		assert_near(expecteds, valid_hits);
	}

	void test_simple_trace()
	{
		add_object(0, vertices0, indices0);
		auto valid_hits = trace_rays(rays);

		// clang-format off
        std::vector<HitPy> expecteds = {
            {{0.0, 0.0, 0.0}, {0.0, 0.0, -1.0}, 2.0, 0.0, 0.0, ignore, 0, 0, 0, true},
            {{0.5, 0.5,-0.0}, {0.0, 0.0, -1.0}, 1.0, 0.5, 0.5, ignore, 0, 0, 0, true}
        };
		// clang-format on
		assert_near(expecteds, valid_hits);
	}

	void test_add_two_objects_and_transform1()
	{
		culling_enabled        = true;
		glm::mat3x4 transform_ = {
		    0.1f, 0.0f, 0.0f, 0.0f,
		    0.0f, 0.1f, 0.0f, 0.0f,
		    0.0f, 0.0f, 0.1f, 0.0f};

		add_object(0, vertices0, indices0);
		add_object(1, vertices1, indices1);
		update_object(1, transform_);
		auto valid_hits = trace_rays(rays);
		// clang-format off
        std::vector<HitPy> expecteds = {
            {{0.000001f, 0.00,  0.05f},{-0.0, 0.0, 1.0}, 1.95f, 0.49999f, 0.00001f, ignore, 1, 11, 0, true},
            {{0.000000,  0.05f, 0.00}, { 0.0, 1.0,-0.0}, 1.95f, 0.5, 0.5, ignore, 1, 6, 0, true},
            {{0.500000,  0.50, -0.00}, { 0.0, 0.0,-1.0}, 1.00f, 0.5, 0.5, ignore, 0, 0, 0, true},
        };
		// clang-format on
		assert_near(expecteds, valid_hits);
	}

	void main()
	{
		initVulkan();
		setupWindow();
		prepare();

		xcb_flush(connection);
		xcb_generic_event_t *event;
		while ((event = xcb_poll_for_event(connection)))
		{
			handleEvent(event);
			free(event);
		}
        test_animated_mesh();
		// Flush device to make sure all resources can be freed
		vkDeviceWaitIdle(device);
	}
};

int main(const int argc, const char *argv[])
{
	VulkanExample vulkanExample{};
	vulkanExample.main();
	return 0;
}

int main1()
{
	glm::mat3x4 transform_3x4 = {
	    1.0f, 0.0f, 0.0f, 0.0f,
	    0.0f, 1.0f, 0.0f, 0.0f,
	    0.0f, 0.0f, 1.0f, 10.0f};
	std::vector<float> isovelocity  = {10.0f, 0, 0};
	std::vector<float> isovelocity2 = {10.0f, 0, 0};
	bool               eq           = isovelocity == isovelocity2;
	assert(eq);
	std::vector<glm::mat3x4> transforms;

	for (uint32_t t = 0; t < NUM_TIME_STEPS; t++)
	{
		// Create copy of transform
		glm::mat3x4 transform_3x4_copy = transform_3x4;

		// Apply linear velocity for now, we don't have angular velocity supported
		transform_3x4_copy[0][3] += (isovelocity[0] * 0.1f * t / NUM_TIME_STEPS);
		transform_3x4_copy[1][3] += (isovelocity[1] * 0.1f * t / NUM_TIME_STEPS);
		transform_3x4_copy[2][3] += (isovelocity[2] * 0.1f * t / NUM_TIME_STEPS);
		std::cout << "step " << t << std::endl;
		std::cout << glm::to_string(transform_3x4_copy) << std::endl;
		transforms.emplace_back(transform_3x4_copy);
	}
	float ray_time = 0.42f;
	float x        = 1.0f / (NUM_TIME_STEPS - 1);

	for (int32_t step = 0; step < NUM_TIME_STEPS; step++)
	{
		int32_t next_step = step + 1;
		float   t         = x * step;
		float   tnext     = x * next_step;
		printf("%d %f\n", step, t);
		if (tnext >= ray_time && ray_time > t)
		{
			float d  = (ray_time - t) / x;
			float x_ = transforms[step][0][3] * (1.0f - d) + transforms[next_step][0][3] * d;
			float y_ = transforms[step][1][3] * (1.0f - d) + transforms[next_step][1][3] * d;
			float z_ = transforms[step][2][3] * (1.0f - d) + transforms[next_step][2][3] * d;
			printf("\t%f (%f, %f, %f)\n", d, x_, y_, z_);
			break;
		}
	}
	auto xform = transform_at_time(transform_3x4, glm::vec3(isovelocity[0], isovelocity[1], isovelocity[3]), 0.95f);
	std::cout << glm::to_string(xform) << std::endl;

	std::set<float> sorted_times{};
	for (const auto &ray : rays2)
	{
		sorted_times.emplace(ray.time);
	}
	for (auto time : sorted_times)
	{
		std::cout << time << std::endl;
	}

	return 0;
}