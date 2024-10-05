cmake_minimum_required(VERSION 3.12)
project(vit
        DESCRIPTION "Inference Vision Transformer (ViT) in plain C/C++ with ggml"
        HOMEPAGE_URL "https://github.com/staghado/vit.cpp")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# Set the compiler flags, modify here if needed----------------GBLANCO-------------------

# Definir el ejecutable de forma predeterminada
set(SOURCES main.cpp vit.cpp)  # Define los archivos fuente predeterminados

if(DEFINED ENV{VIT_XRT})
    message(STATUS "Using XRT")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR} common)       #MK_CPPFLAGS  = -I. -Icommon
    include_directories(ggml/src)                                 #Para anadir el directorio de ggml-xrt.h y ggml-xrt.cpp
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c17 -fPIC")          #MK_CFLAGS    = -std=c17   -fPIC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fPIC")    #MK_CXXFLAGS  = -std=c++17 -fPIC
    set(CMAKE_NVCC_FLAGS "${CMAKE_NVCC_FLAGS} -std=c++17")        #MK_NVCCFLAGS = -std=c++17
    add_definitions(-DGGML_USE_XRT)                               #MK_CPPFLAGS += -DGGML_USE_XRT -I${XILINX_XRT}/include -I./ecas-scripts/SW/hls-typ
    include_directories($ENV{XILINX_XRT}/include ./ecas-scripts/SW/hls-types/include) #Comentario anterior
    link_directories($ENV{XILINX_XRT}/lib)                        #MK_LDFLAGS  += -L${XILINX_XRT}/lib -lxrt_core -lxrt_coreutil
    #target_link_libraries(${PROJECT_NAME} PUBLIC xrt_core xrt_coreutil)               #Comentario Anterior
    list(APPEND SOURCES ggml/src/ggml-xrt.cpp)  # Agrega ggml-xrt.cpp a la lista de fuentes
    #add_executable(${PROJECT_NAME} ggml-xrt.cpp) #OBJS        += ggml-xrt.o
#La ultima linea al parecer tambien ejecuta: #ifdef LLAMA_XRT
#ggml-xrt.o: ggml-xrt.cpp ggml-xrt.h ggml.h
#	$(CXX) $(CXXFLAGS) -c $< -o $@
endif()

add_executable(${PROJECT_NAME} ${SOURCES})

if(DEFINED ENV{VIT_XRT_CLOCK})  #ifdef LLAMA_XRT_CLOCK
    add_definitions(-DXRT_CLOCK)  #MK_CPPFLAGS  += -DXRT_CLOCK
endif()                           #endif


# Vincular las bibliotecas XRT si están definidas
if(DEFINED ENV{VIT_XRT})
    target_link_libraries(${PROJECT_NAME} PUBLIC xrt_core xrt_coreutil)
endif()
#----------------------------------------------------------------FIN GBLANCO---------------------------------------------


# Set the compiler flags, modify here if needed!
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")


# Additional flags for C++ (the ffast-math is to be used with caution!)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")

# deps
add_subdirectory(ggml)

# vit executable
target_link_libraries(${PROJECT_NAME} PUBLIC ggml)
target_include_directories(${PROJECT_NAME} PUBLIC .)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

# quantize executable
option(BUILD_QUANTIZE "Build the quantization executable" ON)

if (BUILD_QUANTIZE)
    add_executable(quantize quantize.cpp)
    target_link_libraries(quantize PUBLIC ggml)
    target_include_directories(quantize PUBLIC .)
    target_compile_features(quantize PUBLIC cxx_std_11)
endif()

# benchmark executable (optional)
option(BUILD_BENCHMARK "Build the benchmark executable" OFF)

if(BUILD_BENCHMARK)
    find_package(nlohmann_json 3.2.0 REQUIRED)  # JSON package needed for benchmark

    # JSON package needed for benchmark
    add_executable(benchmark tests/benchmark.cpp)
    target_link_libraries(benchmark PRIVATE nlohmann_json::nlohmann_json)
    target_link_libraries(benchmark PUBLIC ggml)
    target_include_directories(benchmark PUBLIC .)
    target_compile_features(benchmark PUBLIC cxx_std_17)
endif()
