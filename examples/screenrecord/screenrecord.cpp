/*
* Vulkan Example - Taking screenshots
*
* This sample shows how to get the conents of the swapchain (render output) and store them to disk (see saveScreenshot)
*
* Copyright (C) 2016-2023 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"
#include <media/NdkMediaCodec.h>
#include "codecutils.h"

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
    vks::Buffer uniformBuffer;

    VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
    VkPipeline pipeline{ VK_NULL_HANDLE };
    VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
    VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };


    VulkanExample() : VulkanExampleBase()
    {
        title = "Record Screen to Video";
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
            uniformBuffer.destroy();
        }
        if (codec) {
            codecStarted = false;
            AM_CHECK_RESULT_ERR("AMediaCodec_stop", AMediaCodec_stop(codec));
            AM_CHECK_RESULT_ERR("AMediaCodec_delete", AMediaCodec_delete(codec));
            codec = nullptr;
        }
        if (h264file) {
            fclose(h264file);
            h264file = nullptr;
        }
    }

    static constexpr int VideoWidth = 384;
    static constexpr int VideoHeight = 512;
    VulkanSwapChain codecSwapChain;
    // Active frame buffer index
    uint32_t codecCurrentBuffer = 0;
    AMediaCodec *codec = nullptr;
    FILE *h264file = nullptr;
    bool codecStarted = false;
    uint32_t framesEncoded = 0;

    std::vector<VkSemaphore> codecAcquireSemaphores;
    VkCommandBuffer copyCmd = VK_NULL_HANDLE;
    VkSemaphore codecBlitSemaphore;
    VkSemaphore renderCompleteDup;

    uint32_t codec_width_ = 0, codec_height_ = 0;
    VkFormat codec_format_ = VK_FORMAT_UNDEFINED;

    void setupCodec() {
        codec = AMediaCodec_createEncoderByType("video/avc");
        uint32_t formatsToTry[] = {COLOR_FormatSurface};

        AMediaFormat *format = AMediaFormat_new();
        AMediaFormat_setString(format, AMEDIAFORMAT_KEY_MIME, "video/avc");
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_WIDTH, VideoWidth);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_HEIGHT, VideoHeight);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_BIT_RATE, 2000000);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_FRAME_RATE, 30);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_I_FRAME_INTERVAL, 1);
        AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_COLOR_FORMAT, COLOR_FormatSurface);

        AM_CHECK_RESULT("AMediaCodec_configure", AMediaCodec_configure(codec, format, nullptr, nullptr, AMEDIACODEC_CONFIGURE_FLAG_ENCODE));

        ANativeWindow *w;

        AM_CHECK_RESULT("AMediaCodec_createInputSurface", AMediaCodec_createInputSurface(codec, &w));
        assert(w);

        codecSwapChain.setContext(instance, physicalDevice, device);
        codecSwapChain.initSurface(w);
        codecSwapChain.create(codec_width_, codec_height_, false, true);

        AM_CHECK_RESULT("AMediaCodec_start", AMediaCodec_start(codec));

        h264file = fopen("/data/data/de.saschawillems.vulkanScreenRecord/video.h264", "w");
        if (!h264file) {
            LOGE("cant open file for writing");
        }

        VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();

        for (int i = 0; i < codecSwapChain.images.size(); ++i) {
            VkSemaphore semaphore;
            VK_CHECK_RESULT(
                    vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphore));
            codecAcquireSemaphores.push_back(semaphore);
        }

        VK_CHECK_RESULT(
                vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &codecBlitSemaphore));
        VK_CHECK_RESULT(
                vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &renderCompleteDup));

        codec_format_ = codecSwapChain.colorFormat;
        codecStarted = true;
    }

    void writeEncoded() {
        AMediaCodecBufferInfo info{};
        ssize_t idx = AMediaCodec_dequeueOutputBuffer(codec, &info, 1);
        if (idx >= 0) {
            size_t outsize = 0;
            uint8_t *encoded = AMediaCodec_getOutputBuffer(codec, idx, &outsize);
            if (h264file) {
                if (1 != fwrite(encoded, info.size, 1, h264file)) {
                    LOGE("write failed");
                } else {
                    framesEncoded++;
                }
            }
            AM_CHECK_RESULT_ERR("AMediaCodec_releaseOutputBuffer",
                                AMediaCodec_releaseOutputBuffer(codec, idx, false));
        } else if (idx != AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
            LOGI("AMediaCodec_dequeueOutputBuffer %zd", idx);
        }

    }

    void loadAssets() {
        model.loadFromFile(getAssetPath() + "models/chinesedragon.gltf", vulkanDevice, queue,
                           vkglTF::FileLoadingFlags::PreTransformVertices |
                           vkglTF::FileLoadingFlags::PreMultiplyVertexColors |
                           vkglTF::FileLoadingFlags::FlipY);
        setupCodec();
    }


    void buildCommandBuffers() override
    {
        VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

        VkClearValue clearValues[2];
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

        for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
        {
            // Set target frame buffer
            renderPassBeginInfo.framebuffer = frameBuffers[i];

            VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

            vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

            VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
            vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

            VkRect2D scissor = vks::initializers::rect2D(width, height,	0, 0);
            vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            model.draw(drawCmdBuffers[i]);

            drawUI(drawCmdBuffers[i]);

            vkCmdEndRenderPass(drawCmdBuffers[i]);

            VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
        }
    }

    void setupDescriptors()
    {

        // Pool
        std::vector<VkDescriptorPoolSize> poolSizes = {
                vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),

        };
        VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

        // Layout
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),		// Binding 0: Vertex shader uniform buffer
        };
        VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

        // Set
        VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
        std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
                vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffer.descriptor),	// Binding 0: Vertex shader uniform buffer
        };
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
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
        // Vertex shader uniform buffer block
        vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &uniformBuffer, sizeof(UniformData));
        VK_CHECK_RESULT(uniformBuffer.map());
    }

    void updateUniformBuffers()
    {
        uniformData.projection = camera.matrices.perspective;
        uniformData.view = camera.matrices.view;
        uniformData.model = glm::mat4(1.0f);
        uniformBuffer.copyTo(&uniformData, sizeof(UniformData));
    }

    void prepare() override
    {
        VulkanExampleBase::prepare();
        loadAssets();
        prepareUniformBuffers();
        setupDescriptors();
        preparePipelines();
        buildCommandBuffers();
        prepared = true;
    }

    // Blit-style rendering that doesn't work on the Amazon tablet.
    void codecBlit() {
        if (!codecStarted) return;

        VkSemaphore acquireSemaphore = codecAcquireSemaphores[0]; // Note: We only have one frame in flight, so no need to care about the other semaphores.
        VK_CHECK_RESULT(
                codecSwapChain.acquireNextImage(acquireSemaphore, codecCurrentBuffer));
        VkImage dstImage = codecSwapChain.images[codecCurrentBuffer];

        assert(copyCmd == VK_NULL_HANDLE);
        copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                                    true);

        vks::tools::insertImageMemoryBarrier(
                copyCmd,
                dstImage,
                0,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

        VkImage srcImage = swapChain.images[currentBuffer];
        vks::tools::insertImageMemoryBarrier(
                copyCmd,
                srcImage,
                VK_ACCESS_MEMORY_READ_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

        VkOffset3D blitSize;
        blitSize.x = width;
        blitSize.y = height;
        blitSize.z = 1;

        VkOffset3D blitDstSize;
        blitDstSize.x = VideoWidth;
        blitDstSize.y = VideoHeight;
        blitDstSize.z = 1;
        VkImageBlit imageBlitRegion{};
        imageBlitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlitRegion.srcSubresource.layerCount = 1;
        imageBlitRegion.srcOffsets[1] = blitSize;
        imageBlitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlitRegion.dstSubresource.layerCount = 1;
        imageBlitRegion.dstOffsets[1] = blitDstSize;

        // Issue the blit command
        vkCmdBlitImage(
                copyCmd,
                srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &imageBlitRegion,
                VK_FILTER_LINEAR);

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
                VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
        // Transition destination image to general layout, which is the required layout for mapping the image memory later on
        vks::tools::insertImageMemoryBarrier(
                copyCmd,
                dstImage,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_MEMORY_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

        VK_CHECK_RESULT(vkEndCommandBuffer(copyCmd));

        // Configure the `copyCmd` buffer submission.
        VkSubmitInfo codecSubmitInfo = submitInfo;
        codecSubmitInfo.pCommandBuffers = &copyCmd;
        codecSubmitInfo.commandBufferCount = 1;

        // Wait for the acquire image and draw command buffers to finish.
        const VkSemaphore waitSemaphores[2] = {acquireSemaphore, renderCompleteDup};
        VkPipelineStageFlags submitPipelineStages[2] = {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT };
        codecSubmitInfo.pWaitDstStageMask = submitPipelineStages;
        codecSubmitInfo.pWaitSemaphores = waitSemaphores;
        codecSubmitInfo.waitSemaphoreCount = 2;
        // And signal our codec blit semaphore.
        codecSubmitInfo.pSignalSemaphores = &codecBlitSemaphore;
        codecSubmitInfo.signalSemaphoreCount = 1;

        // Submit the command buffer!
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &codecSubmitInfo, VK_NULL_HANDLE));

        // Then, present to the codec swapchain as soon as the `copyCmd` buffer is complete.
        VK_CHECK_RESULT(
                codecSwapChain.queuePresent(queue, codecCurrentBuffer, codecBlitSemaphore));

    }

    void draw()
    {
        float deltaY = 1.0f;
        camera.rotate(glm::vec3(0.0f, -deltaY, 0.0f));
        VulkanExampleBase::prepareFrame();

        VkCommandBuffer commandBuffers[2] = {drawCmdBuffers[currentBuffer], copyCmd};
        VkSemaphore waitSemaphores[2] = {semaphores.presentComplete, codecAcquireSemaphores[0]};
        VkPipelineStageFlags waitPipelineStages[2] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = commandBuffers;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitPipelineStages;

        // This vkQueueSubmit will signal the `renderComplete` semaphore when it's complete. We
        // later wait for that semaphore in `codecDraw`.
        VkSemaphore signalingSemas[2] = {semaphores.renderComplete, renderCompleteDup};
        submitInfo.pSignalSemaphores = signalingSemas;
        submitInfo.signalSemaphoreCount = 2;

        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

        // `submitFrame` calls vkQueueWaitIdle, so we do it last.
        VulkanExampleBase::submitFrame();

        // Then, present to the codec swapchain as soon as rendering is complete.
        codecBlit();

        VK_CHECK_RESULT(vkQueueWaitIdle(queue));

        writeEncoded();
        // Now that we're done with `copyCmd`, free it here.
        if (copyCmd != VK_NULL_HANDLE)
        {
            vkFreeCommandBuffers(device, vulkanDevice->commandPool, 1, &copyCmd);
            copyCmd = VK_NULL_HANDLE;
        }

    }

    void render() override
    {
        if (!prepared)
            return;
        updateUniformBuffers();
        draw();
    }

    void OnUpdateUIOverlay(vks::UIOverlay *overlay) override
    {
        if (overlay->header("Statistics")) {
            if (codecStarted) {
                overlay->text("Frames encoded %u", framesEncoded);
            }
        }
    }

};

VULKAN_EXAMPLE_MAIN()