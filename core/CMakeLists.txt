add_library(dlssvid_core STATIC
  util/Log.h util/Log.cpp
  util/Error.h util/Error.cpp
  util/Half.h
  util/Sha256.h util/Sha256.cpp
  pipeline/Frame.h pipeline/Frame.cpp
  pipeline/IStage.h
  pipeline/Pipeline.h pipeline/Pipeline.cpp
  gpu/D3D12Device.h gpu/D3D12Device.cpp
  gpu/GpuFrameCache.h gpu/GpuFrameCache.cpp
  io/VideoDecoder.h io/VideoDecoder.cpp
  io/VideoEncoder.h io/VideoEncoder.cpp
  stages/passthrough/PassthroughStage.h stages/passthrough/PassthroughStage.cpp
  passes/PassImage.h passes/PassImage.cpp
  passes/Manifest.h passes/Manifest.cpp
  passes/PassFile.h passes/PassFile.cpp
  passes/PassSequence.h passes/PassSequence.cpp
  passes/formats/ExrIO.h passes/formats/ExrIO.cpp
  passes/formats/PngIO.h passes/formats/PngIO.cpp
  passes/formats/TiffIO.h passes/formats/TiffIO.cpp
  passes/formats/NpzIO.h passes/formats/NpzIO.cpp
  passes/formats/RawIO.h passes/formats/RawIO.cpp
  convert/DepthConvert.h convert/DepthConvert.cpp
  convert/MvConvert.h convert/MvConvert.cpp
  convert/ColorConvert.h convert/ColorConvert.cpp
)

if(DLSSVID_WITH_CUDA)
  target_sources(dlssvid_core PRIVATE gpu/CudaInterop.h gpu/CudaInterop.cpp)
  target_link_libraries(dlssvid_core PUBLIC CUDA::cudart)
endif()

target_include_directories(dlssvid_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dlssvid_core
  PUBLIC
    dlssvid_settings
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    Microsoft::DirectX-Headers
    Microsoft::DirectX-Guids
    ffmpeg::ffmpeg
    OpenEXR::OpenEXR
    PNG::PNG
    TIFF::TIFF
    ZLIB::ZLIB
    d3d12 dxgi dxguid bcrypt
)
