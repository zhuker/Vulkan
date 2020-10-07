/*
* Vulkan Example - Basic example for ray tracing using VK_NV_ray_tracing
*
* Copyright (C) by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"

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
std::vector<Vertex> vertices1 = {
    {{0.0f, 0.0f, -2.0f, 0.0f}},
    {{0.0f, 1.0f, -2.0f, 0.0f}},
    {{1.0f, 0.0f, -2.0f, 0.0f}},
    {{1.0f, 1.0f, -2.0f, 0.0f}},
};

std::vector<Vertex> vertices2 = {
    {{ 0.5f, -0.5f,  0.5f, 0.0f}},
    {{ 0.5f, -0.5f, -0.5f, 0.0f}},
    {{-0.5f, -0.5f,  0.5f, 0.0f}},
    {{ 0.5f,  0.5f,  0.5f, 0.0f}},
    {{-0.5f,  0.5f,  0.5f, 0.0f}},
    {{-0.5f,  0.5f, -0.5f, 0.0f}},
    {{-0.5f, -0.5f, -0.5f, 0.0f}},
    {{ 0.5f,  0.5f, -0.5f, 0.0f}},
};
std::vector<Vertex> vertices3 = {
    {{2.0f, 0.0f, -5.0f, 0.0f}},
    {{2.0f, 1.0f, -5.0f, 0.0f}},
    {{3.0f, 0.0f, -5.0f, 0.0f}},
    {{3.0f, 1.0f, -5.0f, 0.0f}},
};
// clang-format on

std::vector<uint32_t> indices1 = {0, 1, 2, 2, 3, 0};
std::vector<uint32_t> indices2 = {0, 1, 3, 3, 1, 7, 2, 6, 0, 0, 6, 1, 4, 5, 2, 2, 5, 6,
                                  3, 7, 4, 4, 7, 5, 6, 5, 1, 1, 5, 7, 4, 2, 3, 3, 2, 0};
std::vector<uint32_t> indices3 = {0, 1, 2, 2, 3, 0};

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

// Indices for the different ray tracing shader types used in this example
#define INDEX_RAYGEN 0
#define INDEX_MISS 1
#define INDEX_CLOSEST_HIT 2

#define NUM_SHADER_GROUPS 3

struct ObjModel
{
	uint32_t    nbIndices{0};
	uint32_t    nbVertices{0};
	vks::Buffer vertexBuffer;        // Device buffer of all 'Vertex'
	vks::Buffer indexBuffer;         // Device buffer of the indices forming triangles
};

// Instance of the OBJ
struct ObjInstance
{
	uint32_t    objIndex{0};         // Reference to the `m_objModel`
	glm::mat3x4 transform{1};        // Position of the instance
};

class VulkanExample final : public VulkanExampleBase
{
  public:
	PFN_vkCreateAccelerationStructureNV                vkCreateAccelerationStructureNV;
	PFN_vkDestroyAccelerationStructureNV               vkDestroyAccelerationStructureNV;
	PFN_vkBindAccelerationStructureMemoryNV            vkBindAccelerationStructureMemoryNV;
	PFN_vkGetAccelerationStructureHandleNV             vkGetAccelerationStructureHandleNV;
	PFN_vkCmdBuildAccelerationStructureNV              vkCmdBuildAccelerationStructureNV;
	PFN_vkGetAccelerationStructureMemoryRequirementsNV vkGetAccelerationStructureMemoryRequirementsNV;
	PFN_vkCreateRayTracingPipelinesNV                  vkCreateRayTracingPipelinesNV;
	PFN_vkGetRayTracingShaderGroupHandlesNV            vkGetRayTracingShaderGroupHandlesNV;
	PFN_vkCmdTraceRaysNV                               vkCmdTraceRaysNV;

	VkPhysicalDeviceRayTracingPropertiesNV rayTracingProperties{};
	std::vector<ObjModel>                  objects;
	std::vector<ObjInstance>               instances;

	std::vector<AccelerationStructure> bottomLevelASes;
	AccelerationStructure              topLevelAS{};

	vks::Buffer shaderBindingTable;

	struct StorageImage
	{
		VkDeviceMemory memory;
		VkImage        image;
		VkImageView    view;
		VkFormat       format;
	} storageImage;

	struct UniformData
	{
		glm::mat4 viewInverse;
		glm::mat4 projInverse;
	} uniformData;
	vks::Buffer ubo;

	VkPipeline            pipeline;
	VkPipelineLayout      pipelineLayout;
	VkDescriptorSet       descriptorSet;
	VkDescriptorSetLayout descriptorSetLayout;

	VulkanExample() :
	    VulkanExampleBase()
	{
		title            = "VK_NV_ray_tracing";
		settings.overlay = true;
		camera.type      = Camera::CameraType::lookat;
		camera.setPerspective(60.0f, (float) width / (float) height, 0.1f, 512.0f);
		camera.setRotation(glm::vec3(0.0f, 0.0f, 0.0f));
		camera.setTranslation(glm::vec3(0.0f, 0.0f, -2.5f));
		// Enable instance and device extensions required to use VK_NV_ray_tracing
		enabledInstanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
		enabledDeviceExtensions.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
		enabledDeviceExtensions.push_back(VK_NV_RAY_TRACING_EXTENSION_NAME);
	}

	~VulkanExample() final
	{
		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		vkDestroyImageView(device, storageImage.view, nullptr);
		vkDestroyImage(device, storageImage.image, nullptr);
		vkFreeMemory(device, storageImage.memory, nullptr);
		for (auto &blas : bottomLevelASes)
		{
			vkFreeMemory(device, blas.memory, nullptr);
			vkDestroyAccelerationStructureNV(device, blas.accelerationStructure, nullptr);
		}
		vkFreeMemory(device, topLevelAS.memory, nullptr);
		vkDestroyAccelerationStructureNV(device, topLevelAS.accelerationStructure, nullptr);
		for (auto &obj : objects)
		{
			obj.vertexBuffer.destroy();
			obj.indexBuffer.destroy();
		}
		shaderBindingTable.destroy();
		ubo.destroy();
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
	void createTopLevelAccelerationStructure(uint32_t instanceCount, bool update = false)
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

	static void rwBarrier(VkCommandBuffer cmdBuffer)
	{
		VkMemoryBarrier memoryBarrier = vks::initializers::memoryBarrier();
		memoryBarrier.srcAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
		memoryBarrier.dstAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
		vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
	}
	GeometryInstance createGeometryInstance(const ObjInstance &instance) const
	{
		GeometryInstance geometryInstance{};
		geometryInstance.transform                   = instance.transform;
		geometryInstance.instanceId                  = instance.objIndex;
		geometryInstance.mask                        = 0xff;
		geometryInstance.instanceOffset              = 0;
		geometryInstance.flags                       = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV;
		geometryInstance.accelerationStructureHandle = bottomLevelASes[instance.objIndex].handle;
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

		rwBarrier(cmdBuffer);
		vulkanDevice->flushCommandBuffer(cmdBuffer, queue);
		scratchBuffer.destroy();
	}

	void buildBlas()
	{
		std::vector<VkGeometryNV> geoms;
		std::transform(objects.begin(), objects.end(), std::back_inserter(geoms), [](const ObjModel &obj) {
			return createVkGeometryNV(obj);
		});

		bottomLevelASes.resize(objects.size(), AccelerationStructure{});
		for (size_t i = 0; i < geoms.size(); i++)
		{
			createOrUpdateBlas(bottomLevelASes[i], geoms[i]);
		}
	}

	void buildTlas(bool update = false)
	{
		vks::Buffer instanceBuffer;

		std::vector<GeometryInstance> geometryInstances;
		std::transform(instances.begin(), instances.end(), std::back_inserter(geometryInstances), [this](const ObjInstance &obj) {
			return createGeometryInstance(obj);
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
		rwBarrier(cmdBuffer);

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

	/*
		Create scene geometry and ray tracing acceleration structures
	*/
	void createScene()
	{
		buildBlas();
		buildTlas();
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
		    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
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

		std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
		    accelerationStructureWrite,
		    resultImageWrite,
		    uniformBufferWrite};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, VK_NULL_HANDLE);
	}

	/*
		Create our ray tracing pipeline
	*/
	void createRayTracingPipeline()
	{
		VkDescriptorSetLayoutBinding accelerationStructureLayoutBinding{};
		accelerationStructureLayoutBinding.binding         = 0;
		accelerationStructureLayoutBinding.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;
		accelerationStructureLayoutBinding.descriptorCount = 1;
		accelerationStructureLayoutBinding.stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_NV;

		VkDescriptorSetLayoutBinding resultImageLayoutBinding{};
		resultImageLayoutBinding.binding         = 1;
		resultImageLayoutBinding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		resultImageLayoutBinding.descriptorCount = 1;
		resultImageLayoutBinding.stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_NV;

		VkDescriptorSetLayoutBinding uniformBufferBinding{};
		uniformBufferBinding.binding         = 2;
		uniformBufferBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uniformBufferBinding.descriptorCount = 1;
		uniformBufferBinding.stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_NV;

		std::vector<VkDescriptorSetLayoutBinding> bindings({accelerationStructureLayoutBinding,
		                                                    resultImageLayoutBinding,
		                                                    uniformBufferBinding});

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

		updateUniformBuffers();
	}

	/*
		Command buffer generation
	*/
	void buildCommandBuffers() final
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
			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, pipelineLayout, 0, 1, &descriptorSet, 0, 0);

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

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void updateUniformBuffers()
	{
		uniformData.projInverse = glm::inverse(camera.matrices.perspective);
		uniformData.viewInverse = glm::inverse(camera.matrices.view);
		memcpy(ubo.mapped, &uniformData, sizeof(uniformData));
	}

	ObjModel createObject(std::vector<Vertex> &vertices, std::vector<uint32_t> &indices)
	{
		ObjModel model;
		model.nbIndices  = static_cast<uint32_t>(indices.size());
		model.nbVertices = static_cast<uint32_t>(vertices.size());
		updateVertices(model, vertices);
		updateIndices(model, indices);
		return model;
	}

	static ObjInstance createInstance(uint32_t objIndex, const glm::mat3x4 &transform = glm::mat3x4{1})
	{
		ObjInstance instance;
		instance.objIndex  = objIndex;
		instance.transform = transform;
		return instance;
	}

	void prepare() final
	{
		VulkanExampleBase::prepare();

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

		objects.emplace_back(createObject(vertices1, indices1));
		objects.emplace_back(createObject(vertices2, indices2));
		instances.emplace_back(createInstance(0));
		glm::mat3x4 t2 = {
		    0.1f,
		    0.0f,
		    0.0f,
		    0.0f,
		    0.0f,
		    1.0f,
		    0.0f,
		    0.0f,
		    0.0f,
		    0.0f,
		    1.0f,
		    0.0f,
		};
		instances.emplace_back(createInstance(1, t2));

		createScene();
		createStorageImage();
		createUniformBuffer();
		createRayTracingPipeline();
		createShaderBindingTable();
		createDescriptorSets();
		buildCommandBuffers();
		prepared = true;
	}

	uint32_t frameNumber = 0;

	static int64_t current_time_msec()
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	}

	int64_t last_update_sec = -1;
	int64_t start           = current_time_msec() / 1000;

	void updateVertices(ObjModel &obj, std::vector<Vertex> &vertices)
	{
		obj.vertexBuffer.destroy();
		vks::Buffer stagingBuffer;

		auto bufferSize = vertices.size() * sizeof(Vertex);
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		    &stagingBuffer,
		    bufferSize,
		    vertices.data()));

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    &obj.vertexBuffer,
		    bufferSize));

		VkBufferCopy region{0, 0, bufferSize};
		vulkanDevice->copyBuffer(&stagingBuffer, &obj.vertexBuffer, queue, &region);
		stagingBuffer.destroy();
	}

	void updateIndices(ObjModel &obj, std::vector<uint32_t> &indices)
	{
		obj.indexBuffer.destroy();
		vks::Buffer stagingBuffer;

		auto bufferSize = indices.size() * sizeof(uint32_t);
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		    &stagingBuffer,
		    bufferSize,
		    indices.data()));

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    &obj.indexBuffer,
		    bufferSize));

		VkBufferCopy region{0, 0, bufferSize};
		vulkanDevice->copyBuffer(&stagingBuffer, &obj.indexBuffer, queue, &region);
		stagingBuffer.destroy();
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers    = &drawCmdBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
		VulkanExampleBase::submitFrame();
		frameNumber++;
		int64_t sec = current_time_msec() / 1000;
		if (last_update_sec != sec)
		{
			last_update_sec = sec;
			auto elapsed    = sec - start;
			if (elapsed <= 5)
			{
				//update instance transform
				vertices1[0].pos[0] += 0.1f;
				printf("%f\n", vertices1[0].pos[0]);
				printf("%d %ld update instance transform\n", frameNumber, elapsed);
				updateVertices(objects[0], vertices1);
				auto updated_geom = createVkGeometryNV(objects[0]);
				createOrUpdateBlas(bottomLevelASes[0], updated_geom, true);
				//			instances[1].transform[0][0] += 0.01f;
				//            instances[1].transform[1][1] += 0.01f;
				buildTlas(true);
			}
			else
			{
				if (instances.size() < 3)
				{
					printf("create object\n");
					objects.emplace_back(createObject(vertices3, indices3));
					instances.emplace_back(createInstance(2));
					auto geom = createVkGeometryNV(objects[2]);
					bottomLevelASes.resize(objects.size(), AccelerationStructure{});
					createOrUpdateBlas(bottomLevelASes[2], geom);
					buildTlas();
					createDescriptorSets();
					destroyCommandBuffers();
					createCommandBuffers();
					buildCommandBuffers();
					vkDeviceWaitIdle(device);
				}
			}
		}
	}

	void render() final
	{
		if (!prepared)
			return;
		draw();
		if (camera.updated)
			updateUniformBuffers();
	}
};

VULKAN_EXAMPLE_MAIN()
