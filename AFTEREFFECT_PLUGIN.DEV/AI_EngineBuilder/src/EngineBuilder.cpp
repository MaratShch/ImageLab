#include <iostream>
#include <fstream>
#include <memory>
#include <string>

// The TensorRT APIs
#include <NvInfer.h>
#include <NvOnnxParser.h>

// Check CUDA version and architecture
#if defined(_NVIDIA_CUDA_13) && defined(_NVIDIA_CUDA_10)
#error Configuration error in EngineBuilder.cpp: Both _NVIDIA_CUDA_13 and _NVIDIA_CUDA_10 are defined. Only one must be defined.
#endif

#if !defined(_NVIDIA_CUDA_13) && !defined(_NVIDIA_CUDA_10)
#error Configuration error in EngineBuilder.cpp: One CUDA configuration _NVIDIA_CUDA_13 or _NVIDIA_CUDA_10 must be defined.
#endif

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "CRASH: Missing input arguments!" << std::endl;
        std::cerr << "Usage:   " << argv[0] << " <input_onnx_path> <output_engine_path>" << std::endl;
        return -1;
    }

    std::string onnxFilePath = argv[1];
    std::string engineFilePath = argv[2];

    std::cout << "Starting NOISE CLINIC Engine Builder..." << std::endl;

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    
    if (!parser->parseFromFile(onnxFilePath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "Failed to parse ONNX file!" << std::endl;
        return -1;
    }

    // 5. CONFIGURING THE OPTIMIZER & FIXING DYNAMIC SHAPES
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    
    // --- START DYNAMIC PROFILE BLOCK ---
    // This is required because your ONNX has dynamic H/W axes.
    auto profile = builder->createOptimizationProfile();
    const char* inputName = network->getInput(0)->getName(); // Usually "input"

    // Set valid dimensions: Min (32x32), Opt (1080p), Max (4K)
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 32, 32});
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 1080, 1920});
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, 2160, 3840});
    
    config->addOptimizationProfile(profile);
    // --- END DYNAMIC PROFILE BLOCK ---

#ifdef _NVIDIA_CUDA_13    
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1 GB
    std::cout << "Compiling for RTX 2000 Ada with Optimization Profile..." << std::endl;
#endif

#ifdef _NVIDIA_CUDA_10
    // Older TensorRT versions use setMaxWorkspaceSize
    config->setMaxWorkspaceSize(1536ULL << 20); // 1.5 GB
    std::cout << "Compiling for GTX-1060 with Optimization Profile..." << std::endl;
#endif
    
    // 6. COMPILING THE ENGINE
    auto serializedModel = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!serializedModel) {
        std::cerr << "Failed to build the serialized engine." << std::endl;
        return -1;
    }

    // 7. SAVING THE RESULT
    std::ofstream engineFile(engineFilePath, std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    engineFile.close();

    std::cout << "SUCCESS: NOISE CLINIC Engine is ready for production!" << std::endl;
    return 0;
}