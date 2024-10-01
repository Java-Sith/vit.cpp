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

// Function to find the next power of two greater than or equal to n
int next_power_of_two(int n) {
    if (n <= 64) {
    	return 64;
    } else {
    	return pow(2, ceil(log2(n)));
    }
}


int main(int argc, char** argv) {
    INIT_PROFILER(cynq_profiler)
    int device_index = 0;

    if (argc != 4) {
        return EXIT_FAILURE;
    }
    
    // Get input size
    static std::string binaryFile = "../HW/package.hw/kernels.xclbin";
    int a_rows = std::stoi(argv[1]);
    int b_cols = std::stoi(argv[2]);
    int b_rows = std::stoi(argv[3]);
    b_cols = b_cols < 8 ? 8 : (b_cols - (b_cols & 0b111));
    int c_cols = 1;

    // Compute sizes
    //int padded_rows = next_power_of_two(a_rows);
    //int padded_cols = next_power_of_two(b_cols);

    std::cout << "A rows: " << a_rows << "\n"
          << "B cols: " << b_cols << "\n"
          << "C cols: " << c_cols << std::endl;

    int size_a = a_rows * b_cols;
    int size_b = c_cols * b_cols;
    int size_c = a_rows * c_cols;

    GET_PROFILE_INSTANCE(setup_time, cynq_profiler);
    setup_time->reset();

    std::cout << "Open the device " << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);;

    auto matvecmul = xrt::kernel(device, uuid, "matvecmul");
    setup_time->tick();

    std::cout << "Allocate Buffer in Global Memory\n";
    auto bo_a = xrt::bo(device, size_a * sizeof(float), matvecmul.group_id(0));
    auto bo_b = xrt::bo(device, size_b * sizeof(float), matvecmul.group_id(1));
    auto bo_c = xrt::bo(device, size_c * sizeof(float), matvecmul.group_id(2));

    // Map the contents of the buffer object into host memory
    auto bo_a_map = bo_a.map<float*>();
    auto bo_b_map = bo_b.map<float*>();
    auto bo_c_map = bo_c.map<float*>();

    std::fill(bo_a_map, bo_a_map + size_a, 0.0f);
    std::fill(bo_b_map, bo_b_map + size_b, 0.0f);
    std::fill(bo_c_map, bo_c_map + size_c, 0.0f);
    
    // Filling data
    std::cout << "Filling Buffers\n";
    // Fill A
    float as = 0.3f;
    std::cout << "A: " << std::endl;
    for (int row = 0; row < a_rows; ++row) {
        for (int col = 0; col < b_cols; ++col) {
          bo_a_map[col + row * b_cols] = as;
          if (a_rows < 16 && b_cols < 16) {
            std::cout << as << ", ";
          }
          as += 0.01;
        }
        if (a_rows < 16 && b_cols < 16) {
          std::cout << std::endl;
        }
    }

    // Fill B
    float bs = 1.0f;
    std::cout << "B: " << std::endl;
    for (int col = 0; col < b_cols; ++col) {
        bo_b_map[col] = bs;
        if (a_rows < 16 && b_cols < 16) {
          std::cout << bs << ", ";
        }
        bs += 0.05;
    }
    if (a_rows < 16 && b_cols < 16) {
      std::cout << std::endl;
    }

    for (int row = 0; row < b_rows; ++row)
    {
      // Synchronize buffer content with device side
      std::cout << "Synchronize input buffer data to device global memory\n";
      START_PROFILE(kernel_execution, cynq_profiler, 10)
      bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      //std::cout << "Execution of the kernel\n";
      auto run = matvecmul(bo_a, bo_b, bo_c, a_rows, b_cols, c_cols);
      //std::cout << "Waiting to the end\n";
      run.wait();

      // Get the output;
      //std::cout << "Get the output data from the device" << std::endl;
      bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      END_PROFILE(kernel_execution);

      // Multiply by software
      float c_sw[size_c];
      std::fill(c_sw, c_sw + size_c, 0.0f);
      for (int row = 0; row < a_rows; ++row) {
          c_sw[row] = 0.f;
          for (int k = 0; k < b_cols; ++k) {
            c_sw[row] += bo_a_map[row * b_cols + k] * bo_b_map[k];
          }
      }

      // Compare results
      std::cout << "C_SW - C_HW: " << std::endl;
      for (int row = 0; row < a_rows; ++row) {
        if (a_rows < 16 && b_cols < 16) {
          std::cout << c_sw[row] << " - " << bo_c_map[row] << std::endl;
        }
        float err = fabs(c_sw[row] - bo_c_map[row]) / c_sw[row];
        // Check for 0.1%
        if (err > 0.001) {
          std::cerr << "Error in index: " << row
                    << " Val: " << err
                    << " C_HW: " << bo_c_map[row] << " C_SW: " << c_sw[row] << std::endl;
          return -1;
        }
      }
    }
    
    std::cout << cynq_profiler << std::endl;
    std::cout << "TEST PASSED\n";
    return 0;
}
