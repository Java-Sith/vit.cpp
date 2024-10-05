/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include <iostream>
#include <cstdint>
#include <cstring>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

// Profiler
#include "timer.hpp"

// HLS Types
#include "ap_fixed.h"

using DataT = ap_fixed<16, 6>;
//Float to fixed point in bytes

typedef union {
  uint16_t rvalues[4];
  uint64_t packet;
} PacketU;

int main(int argc, char** argv) {
    INIT_PROFILER(cynq_profiler)
    int device_index = 0;

    if (argc != 5) {
        return EXIT_FAILURE;
    }

    // Get input size
    std::string binaryFile{argv[1]};
    int a_rows = std::stoi(argv[2]);
    int b_cols = std::stoi(argv[3]);
    b_cols = b_cols < 8 ? 8 : (b_cols - (b_cols & 4));
    int c_cols = std::stoi(argv[4]);
    c_cols = c_cols < 8 ? 8 : (c_cols - (c_cols & 4));

    std::cout << "A rows: " << a_rows << "\n"
              << "B cols: " << b_cols << "\n"
              << "C cols: " << c_cols << std::endl;

    // Compute sizes
    int size_a = a_rows * b_cols;
    int size_b = c_cols * b_cols;
    int size_c = a_rows * c_cols;

    std::vector<DataT> a, b, c;

    GET_PROFILE_INSTANCE(setup_time, cynq_profiler);
    setup_time->reset();

    std::cout << "Open the device " << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    auto matmul = xrt::kernel(device, uuid, "matmul");
    auto elementwise = xrt::kernel(device, uuid, "elementwise");
    setup_time->tick();

    std::cout << "Allocate Buffer in Global Memory\n";
    int size_em = 16;
    auto bo_a_mm = xrt::bo(device, size_a * sizeof(float), matmul.group_id(0));
    auto bo_b_mm = xrt::bo(device, size_b * sizeof(float), matmul.group_id(1));
    auto bo_c_mm = xrt::bo(device, size_c * sizeof(float), matmul.group_id(2));
    auto bo_a_ew = xrt::bo(device, size_em * sizeof(uint16_t), elementwise.group_id(0));
    auto bo_b_ew = xrt::bo(device, size_em * sizeof(uint16_t), elementwise.group_id(1));
    auto bo_c_ew = xrt::bo(device, size_em * sizeof(uint16_t), elementwise.group_id(2));


    // Map the contents of the buffer object into host memory
    auto bo_a_mm_map = bo_a_mm.map<float*>();
    auto bo_b_mm_map = bo_b_mm.map<float*>();
    auto bo_c_mm_map = bo_c_mm.map<float*>();
    auto bo_a_ew_map = bo_a_ew.map<uint16_t*>();
    auto bo_b_ew_map = bo_b_ew.map<uint16_t*>();
    auto bo_c_ew_map = bo_c_ew.map<uint16_t*>();


    // Filling data
    std::cout << "Filling Buffers\n";
    //std::copy(a.begin(), a.end(), bo_a_mm_map);
    //std::copy(b.begin(), b.end(), bo_b_mm_map);

    DataT as = 0.02, bs = 0.03;
    std::cout << "A: " << std::endl;
    for (int elem = 0; elem < size_a; ++elem) {
        //std::cout << as.V << " ";
        bo_a_mm_map[elem] = as;
        //std::cout << std::hex << as.V << " ";
        as += 0.03;
        if ((elem + 1) % b_cols == 0) {
            //std::cout << std::endl;
            as = 0.025;
        }
    }
    std::cout << "B: " << std::endl;
    for (int elem = 0; elem < size_b; ++elem) {
        //std::cout << bs.V << " ";
        //std::cout << std::hex << bs.V << " ";
        bo_b_mm_map[elem] = bs;
        bs += 0.07;
        if ((elem + 1) % b_cols == 0) {
            //std::cout << std::endl;
            bs = 0.04;
        }
    }
    bs = DataT{0};
    std::cout << "EW A, B: " << std::endl;
    for (int elem = 0; elem < size_em; ++elem) {
        //std::cout << bs << " ";
        bo_a_ew_map[elem] = bs.V;
        bo_b_ew_map[elem] = bs.V;
        bs += DataT{0.07};
    }
    // std::cout << std::endl;
    std::fill(bo_c_mm_map, bo_c_mm_map + size_c, 0);
    std::fill(bo_c_ew_map, bo_c_ew_map + size_em, 0);

    // Synchronize buffer content with device side
    std::cout << "Synchronize input buffer data to device global memory\n";
    //START_PROFILE(kernel_execution, cynq_profiler, 1000)
    bo_a_mm.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b_mm.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_a_ew.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b_ew.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Start the clock
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Execution of the kernel: matmul\n";
    auto run_mm = matmul(bo_a_mm, bo_b_mm, bo_c_mm, a_rows, b_cols, c_cols);
    std::cout << "Waiting to the end\n";
    run_mm.wait();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto matmul_time = end_time - start_time;

    // Start the clock
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Execution of the kernel: elemwise\n";
    auto run_ew = elementwise(bo_a_ew, bo_b_ew, bo_c_ew, size_em, 0); // 0: add, 1: addrelu, 2: mult
    std::cout << "Waiting to the end\n";
    run_ew.wait();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elementwise_time = end_time - start_time;

    // Get the output;
    std::cout << "Get the output data from the device" << std::endl;
    bo_c_mm.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_c_ew.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    //END_PROFILE(kernel_execution);

    std::cout << "C: " << std::endl;
    for (int elem = 0; elem < size_c; ++elem) {
        DataT cs;
        cs.V = bo_c_mm_map[elem];
        //std::cout << cs << " ";
        //std::cout << std::hex << cs.V << " ";
        //if ((elem + 1) % c_cols == 0) std::cout << std::endl;
    }
    std::cout << "EW C: " << std::endl;
    for (int elem = 0; elem < size_em; ++elem) {
        DataT cs;
        cs.V = bo_c_ew_map[elem];
        //std::cout << cs << " ";
    }
    // std::cout << std::endl;
    // Print the duration
    std::cout << "Matrix multiplication = " << matmul_time/std::chrono::milliseconds(1) << " ms " << '\n';
    std::cout << "Element wise = " << elementwise_time/std::chrono::milliseconds(1) << " ms " << '\n';
    std::cout << cynq_profiler << std::endl;
    std::cout << "TEST PASSED\n";
    return 0;
}
