#include "vulkan/vulkan.h"
#include "VkBootstrap.h"
#include <vma/vk_mem_alloc.h>

#include <cstdio>
#include <cassert>
#include <random>

#define UNIQUE_NAME2(name, line) name##line
#define UNIQUE_NAME(name, line) UNIQUE_NAME2(name, line)

struct DeferCreator {
	static DeferCreator creator() {
		return {};
	}

	template<typename T>
	struct DeferCall {
		T call;
		~DeferCall() {
			call();
		}
	};

	template<typename T>
	DeferCall<T> operator +(T call) {
		return DeferCall<T>{call};
	}
};

// cleanup is handled with this which will run the call on scope end
#define DEFER auto UNIQUE_NAME(defer_creator_, __LINE__) = DeferCreator::creator() + [&]()

VkShaderModule LoadShaderModule(VkDevice device, const char* filename) {
	FILE *fp = std::fopen(filename, "rb");
	assert(fp);

	std::fseek(fp, 0L, SEEK_END);
	std::vector<uint8_t> bytes(std::ftell(fp));
	std::fseek(fp, 0L, SEEK_SET);
	std::fread(bytes.data(), bytes.size(), 1, fp);
	std::fclose(fp);

	VkShaderModuleCreateInfo shader_module_create_info {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = bytes.size(),
		.pCode = reinterpret_cast<uint32_t*>(bytes.data())
	};

	VkShaderModule result;
	assert(vkCreateShaderModule(device, &shader_module_create_info, nullptr, &result) == VK_SUCCESS);
	return result;
}

struct Buffer {

	const VmaAllocator vma_allocator;
	VkBuffer buffer;
	VmaAllocation allocation;
	void* mapped_data;

	Buffer(VmaAllocator vma_allocator, size_t size, VkBufferUsageFlags usage)
	: vma_allocator(vma_allocator) {

		const VkBufferCreateInfo buffer_create_info {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = uint32_t(size),
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};

	    constexpr VmaAllocationCreateInfo vma_allocation_create_info {
	        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
	        .usage =  VMA_MEMORY_USAGE_AUTO,
	    };


	    VmaAllocationInfo device_allocation_info {};

	    auto vk_result = vmaCreateBuffer(vma_allocator, &buffer_create_info, &vma_allocation_create_info,  &buffer, &allocation, &device_allocation_info);
	    assert(vk_result == VK_SUCCESS);

	    VkMemoryPropertyFlags memory_property_flags {};
	    vmaGetAllocationMemoryProperties(vma_allocator, allocation, &memory_property_flags);
	    assert(memory_property_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	    mapped_data = device_allocation_info.pMappedData;
	}

	~Buffer() {
		vmaDestroyBuffer(vma_allocator, buffer, allocation);
	}
};

struct ComputeProgram {

	VkDescriptorPool descriptor_pool;

	VkDescriptorSetLayout descriptor_set_layout;
	VkPipeline pipeline;
	VkPipelineLayout pipeline_layout;
};

void DestroyComputeProgram(ComputeProgram& program, VkDevice device) {
	vkDestroyPipeline(device, program.pipeline, nullptr);
	vkDestroyPipelineLayout(device, program.pipeline_layout, nullptr);
	vkDestroyDescriptorSetLayout(device, program.descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, program.descriptor_pool, nullptr);
}

ComputeProgram CreateConvolutionProgram(VkDevice device, const char* filename) {

	ComputeProgram result;

	// this is setting the budget for our descriptor pool, which is fixed ahead of time based on how many descriptor sets
	// we will have in flight at any point in time. You can have one descriptor pool shared across multiple descriptor sets but
	// that would have complicated the example too much

	static constexpr VkDescriptorPoolSize descriptor_pool_sizes[] {
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
	};

    static constexpr VkDescriptorPoolCreateInfo descriptor_pool_create_info {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = 0,
        .maxSets = 1,
        .poolSizeCount = uint32_t(std::size(descriptor_pool_sizes)),
        .pPoolSizes = descriptor_pool_sizes,
    };
	assert(vkCreateDescriptorPool(device, &descriptor_pool_create_info, nullptr, &result.descriptor_pool) == VK_SUCCESS);

	// Load our shader, this was compiled from convolution.comp using glslc in the cmake file
	VkShaderModule convolution_shader_module = LoadShaderModule(device, filename);
	DEFER { vkDestroyShaderModule(device, convolution_shader_module, nullptr); };

	const VkDescriptorSetLayoutBinding descriptor_set_layout_bindings[] = { {
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
		}, {
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT			
		}, {
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
		}
	};

	const VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = uint32_t(std::size(descriptor_set_layout_bindings)),
		.pBindings = descriptor_set_layout_bindings
	};

	assert(vkCreateDescriptorSetLayout(device, &descriptor_set_layout_create_info, nullptr, &result.descriptor_set_layout) == VK_SUCCESS);

	// Create the pipeline layout, which is a collection of set layouts that we use
	const VkPipelineLayoutCreateInfo pipeline_layout_create_info {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &result.descriptor_set_layout
	};

	assert(vkCreatePipelineLayout(device, &pipeline_layout_create_info, nullptr, &result.pipeline_layout) == VK_SUCCESS);

	// Create the compute pipeline

	const VkComputePipelineCreateInfo convolution_create_pipeline_info {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_COMPUTE_BIT,
			.module = convolution_shader_module,
			.pName = "main" // shader entry point
		},
		.layout = result.pipeline_layout
	};

	assert(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &convolution_create_pipeline_info, nullptr, &result.pipeline) == VK_SUCCESS);

	return result;
}

void RunConvolutionProgram(const ComputeProgram& program, VkDevice device, VkCommandBuffer command_buffer, const Buffer &constants_buffer, const Buffer &input_buffer, const Buffer &output_buffer, uint32_t work_size) {

	// setup the inputs for the pipeline to point to the buffers we created above
    const VkDescriptorSetAllocateInfo descriptor_set_allocate_info {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = program.descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &program.descriptor_set_layout
    };

    VkDescriptorSet descriptor_set;
	assert(vkAllocateDescriptorSets(device, &descriptor_set_allocate_info, &descriptor_set) == VK_SUCCESS);

	const VkDescriptorBufferInfo write_constants_info {
		.buffer = constants_buffer.buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE
	};

	const VkDescriptorBufferInfo write_input_info {
		.buffer = input_buffer.buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE
	};

	const VkDescriptorBufferInfo write_output_info {
		.buffer = output_buffer.buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE
	};

	const VkWriteDescriptorSet descriptor_writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = descriptor_set,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &write_constants_info
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = descriptor_set,
			.dstBinding =1,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &write_input_info
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = descriptor_set,
			.dstBinding = 2,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &write_output_info
		}
	};

	vkUpdateDescriptorSets(device, uint32_t(std::size(descriptor_writes)), descriptor_writes, 0, nullptr);		

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, program.pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, program.pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
    vkCmdDispatch(command_buffer, work_size/64, 1, 1);
}

int main(int argc, char* argv[]) {

	// Use vk-bootstrap to setup our device to save some typing. Leaving out error checking for compactness as it asserts internally anyway
	vkb::Instance instance = vkb::InstanceBuilder()
		.set_app_name("Vulkan Compute Example")
		.set_headless() // this means we don't need to have a surface setup
		.build()
		.value();
	DEFER { destroy_instance(instance); };

	vkb::PhysicalDevice physical_device = vkb::PhysicalDeviceSelector(instance)
		.select()
		.value();

	vkb::Device device = vkb::DeviceBuilder(physical_device)
		.build()
		.value();
    DEFER { destroy_device(device); };

	// use VMA to save some work when allocating buffers later on 
	VmaAllocator vma_allocator;
    const VmaAllocatorCreateInfo vma_create_info {
        .flags = 0,
        .physicalDevice = physical_device,
        .device = device,
        .instance = instance,
        .vulkanApiVersion = VK_API_VERSION_1_0,
    };

	assert(vmaCreateAllocator(&vma_create_info, &vma_allocator) == VK_SUCCESS);
	DEFER { vmaDestroyAllocator(vma_allocator); };

	const uint32_t compute_family_index = *device.get_queue_index(vkb::QueueType::compute);

	const VkCommandPoolCreateInfo compute_command_pool_create_info {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.queueFamilyIndex = compute_family_index
	};

	VkCommandPool compute_command_pool;
	assert(vkCreateCommandPool(device, &compute_command_pool_create_info, nullptr, &compute_command_pool) == VK_SUCCESS);
	DEFER { vkDestroyCommandPool(device, compute_command_pool, nullptr); };

	ComputeProgram convolution = CreateConvolutionProgram(device, "convolution.spv");
	DEFER { DestroyComputeProgram(convolution, device); };

	const float kernel[] = {
		0.001f, 0.002f, 0.003f, 0.004f,
		0.005f, 0.006f, 0.007f, 0.008f,
		0.009f, 0.010f, 0.011f, 0.012f,
		0.013f, 0.014f, 0.447f, 0.448f,
	};

	// Create the buffers we use to store everything
	constexpr size_t WORK_SIZE = 1024*1024;
	Buffer constants_buffer(vma_allocator, 16*sizeof(float), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	Buffer input_buffer(vma_allocator, WORK_SIZE*sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
	Buffer output_buffer(vma_allocator, WORK_SIZE*sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

	memcpy(constants_buffer.mapped_data, kernel, 16*sizeof(float));

	// setup some source data
	std::random_device random_device;
	std::mt19937 random_generator(random_device());
	std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);

	for (size_t n = 0; n < WORK_SIZE; ++n) {
		reinterpret_cast<float*>(input_buffer.mapped_data)[n] = distribution(random_generator);
		reinterpret_cast<float*>(output_buffer.mapped_data)[n] = 0.0f;
	}

    const VkCommandBufferAllocateInfo command_buffer_alloc_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = compute_command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    VkCommandBuffer command_buffer;
    assert(vkAllocateCommandBuffers(device, &command_buffer_alloc_info, &command_buffer) == VK_SUCCESS);

    const VkCommandBufferBeginInfo command_buffer_begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0,
    };

    assert(vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info) == VK_SUCCESS);
	RunConvolutionProgram(convolution, device, command_buffer, constants_buffer, input_buffer, output_buffer, WORK_SIZE);
    assert(vkEndCommandBuffer(command_buffer) == VK_SUCCESS);

    const VkSubmitInfo submit_info {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
    };

    std::printf("Begin compute job\n");
    assert(vkQueueSubmit(*device.get_queue(vkb::QueueType::compute), 1, &submit_info, VK_NULL_HANDLE) == VK_SUCCESS);
    std::printf("Finished compute job\n");

    vkQueueWaitIdle(*device.get_queue(vkb::QueueType::compute));

    // print a bit of the middle so we can see if it did anything
    for (size_t n = 1000; n < 1000+32; ++n) {
    	std::printf("[%zu] = %f -> %f\n", n, reinterpret_cast<float*>(input_buffer.mapped_data)[n], reinterpret_cast<float*>(output_buffer.mapped_data)[n]);
    }

    vkDeviceWaitIdle(device);

	return 0;
}
