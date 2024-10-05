//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <float.h>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <mutex>

#include "ggml-xrt.h"
#include "ggml-backend-impl.h"
#include "ggml-quants.h"
#include "ap_fixed.h"

// XRT includes
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "ecas-scripts/SW/timer.hpp"

#define MATRIX_ROW_PADDING 512
#define UNUSED GGML_UNUSED

static int g_device_count = -1;
static int g_main_device = 0;
static const size_t CACHE_LINE_SIZE_F32 = 64/sizeof(float);

static xrt::device myDevice;
static std::string binaryFile = "./ecas-scripts/HW/package.hw/kernels.xclbin";
//static xrt::kernel matvecmul;
static xrt::kernel elementwise;
//static xrt::kernel softmax;
//static xrt::kernel rmsnorm;
//static xrt::kernel unary;

static bool g_xrt_loaded = false;

bool ggml_xrt_loaded(void) {
    return g_xrt_loaded;
}

inline static void ggml_vec_cpy_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i]  = x[i]; }
inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) { for (int i = 0; i < n; ++i) y[i] *= v; }
inline static void ggml_vec_acc_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] += x[i]; }

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

GGML_CALL static void ggml_xrt_set_device(const int main_device) {
    if (main_device >= g_device_count) {
        fprintf(stderr, "warning: cannot set main_device=%d because there are only %d devices. Using device %d instead.\n",
                main_device, g_device_count, g_main_device);
        return;
    }

    if (g_main_device != main_device && g_device_count > 1) {
        g_main_device = main_device;
        //cudaDeviceProp prop;
        //CUDA_CHECK(cudaGetDeviceProperties(&prop, g_main_device));
        //fprintf(stderr, "%s: using device %d (%s) as main device\n", __func__, g_main_device, prop.name);
    }
    std::cout << "Open the device: " << g_main_device << std::endl;
    myDevice = xrt::device(g_main_device);
    std::cout << "Load the xclbin: " << binaryFile << std::endl;
    auto uuid = myDevice.load_xclbin(binaryFile);
    //matvecmul = xrt::kernel(myDevice, uuid, "matvecmul");
    elementwise = xrt::kernel(myDevice, uuid, "elementwise");
    //softmax = xrt::kernel(myDevice, uuid, "softmax");
    //rmsnorm = xrt::kernel(myDevice, uuid, "rmsnorm");
    //unary = xrt::kernel(myDevice, uuid, "unary");
    fprintf(stderr, "Using device %d as main device\n", g_main_device);
}

// void ggml_xrt_dup(
//         const struct ggml_compute_params * params,
//         struct ggml_tensor * dst) {

//     ggml_compute_forward_dup(params, dst);
// }

// ggml_compute_forward_add

extern "C" void ggml_xrt_add_f32(const struct ggml_compute_params * params,
              struct ggml_tensor * dst);

void ggml_xrt_add_f32(const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    // Lock the mutex at the start of the function
    //std::lock_guard<std::mutex> lock(kernel_mutex);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    // Determine next power of two sizes
    // int padded_ne00 = next_power_of_two(ne00);
    // int padded_ne01 = next_power_of_two(ne01);
    // int padded_ne10 = next_power_of_two(ne10);
    // int padded_ne11 = ne11;

    int64_t src0_size = ne00 * ne01;
    int64_t src1_size = ne10 * ne11;
    int64_t dst_size = ne0 * ne1;

    // int padded_size0 = padded_ne00 * padded_ne01;
    // int padded_size1 = padded_ne10 * padded_ne11;
    // int padded_dst_size = padded_ne00 * padded_ne01;

    // Allocate XRT buffers
    auto bo_a = xrt::bo(myDevice, src0_size * sizeof(float), elementwise.group_id(0));
    auto bo_b = xrt::bo(myDevice, src1_size * sizeof(float), elementwise.group_id(1));
    auto bo_c = xrt::bo(myDevice, dst_size * sizeof(float), elementwise.group_id(2));

    // Map buffers to host memory
    auto bo_a_map = bo_a.map<float*>();
    auto bo_b_map = bo_b.map<float*>();
    auto bo_c_map = bo_c.map<float*>();

    // Fill the buffers with zeroes
    std::fill(bo_a_map, bo_a_map + src0_size, 0.0f);
    std::fill(bo_b_map, bo_b_map + src1_size, 0.0f);
    std::fill(bo_c_map, bo_c_map + dst_size, 0.0f);

    for (int64_t i03 = 0; i03 < ne03; i03++)
    {
        for (int64_t i02 = 0; i02 < ne02; i02++)
        {
            // Copy tensor data to buffers with broadcasting

            float *x = (float *)src0->data + i02*nb2 + i03*nb3;
            float *y = (float *)src1->data + i02*nb2 + i03*nb3;
            float *d  = (float *)dst->data + i02*nb2 + i03*nb3;

            ggml_vec_cpy_f32(src0_size, bo_a_map, x);
            ggml_vec_cpy_f32(src1_size, bo_b_map, y);

#ifndef NDEBUG
            std::cout << "Execution of the kernel Elementwise Add\n";
#endif

            // Synchronize buffers with device
            bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Execute the elementwise kernel
            auto run = elementwise(bo_a, bo_b, bo_c, dst_size, 0);
            run.wait();

            // Synchronize results back to host
            bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

#ifndef NDEBUG
            std::cout << "Get the output data from the device" << std::endl;
#endif

            ggml_vec_cpy_f32(dst_size, d, bo_c_map);
        }
    }
}

/*static void ggml_xrt_add(
    const struct ggml_compute_params *params,
    struct ggml_tensor *dst)
{

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_xrt_add_f32(params, dst);
            } break;
        default:
            {
                ggml_compute_forward_add(params, dst);
            } break;
    }
}*/

// ggml_compute_forward_mul

extern "C" void ggml_xrt_mul_f32(const struct ggml_compute_params * params,
              struct ggml_tensor * dst);

void ggml_xrt_mul_f32(const struct ggml_compute_params * params,
                      struct ggml_tensor * dst) {

    // Lock the mutex at the start of the function
    //std::lock_guard<std::mutex> lock(kernel_mutex);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    // Determine next power of two sizes
    // int padded_ne00 = next_power_of_two(ne00);
    // int padded_ne01 = next_power_of_two(ne01);
    // int padded_ne10 = next_power_of_two(ne10);
    // int padded_ne11 = ne11;

    int64_t src0_size = ne00 * ne01;
    int64_t src1_size = ne10 * ne11;
    int64_t dst_size = ne0 * ne1;

    // int padded_size0 = padded_ne00 * padded_ne01;
    // int padded_size1 = padded_ne10 * padded_ne11;
    // int padded_dst_size = padded_ne00 * padded_ne01;

    // Allocate XRT buffers
    auto bo_a = xrt::bo(myDevice, src0_size * sizeof(float), elementwise.group_id(0));
    auto bo_b = xrt::bo(myDevice, src1_size * sizeof(float), elementwise.group_id(1));
    auto bo_c = xrt::bo(myDevice, dst_size * sizeof(float), elementwise.group_id(2));

    // Map buffers to host memory
    auto bo_a_map = bo_a.map<float*>();
    auto bo_b_map = bo_b.map<float*>();
    auto bo_c_map = bo_c.map<float*>();

    // Fill the buffers with zeroes
    std::fill(bo_a_map, bo_a_map + src0_size, 0.0f);
    std::fill(bo_b_map, bo_b_map + src1_size, 0.0f);
    std::fill(bo_c_map, bo_c_map + dst_size, 0.0f);

    for (int64_t i03 = 0; i03 < ne03; i03++)
    {
        for (int64_t i02 = 0; i02 < ne02; i02++)
        {

            float *x = (float *)src0->data + i02*nb2 + i03*nb3;
            float *y = (float *)src1->data + i02*nb2 + i03*nb3;
            float *d  = (float *)dst->data + i02*nb2 + i03*nb3;

            ggml_vec_cpy_f32(src0_size, bo_a_map, x);
            ggml_vec_cpy_f32(src1_size, bo_b_map, y);

#ifndef NDEBUG
            std::cout << "Execution of the kernel Elementwise Mul\n";
#endif

            // Synchronize buffers with device
            bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Execute the elementwise kernel
            auto run = elementwise(bo_a, bo_b, bo_c, dst_size, 1);
            run.wait();

            // Synchronize results back to host
            bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

#ifndef NDEBUG
            std::cout << "Get the output data from the device" << std::endl;
#endif

            // Copy results to dst
            ggml_vec_cpy_f32(dst_size, d, bo_c_map);
        }
    }
}

/*static void ggml_xrt_mul(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_xrt_mul_f32(params, dst);
            } break;
        default:
            {
                ggml_compute_forward_mul(params, dst);
            } break;
    }
}*/

// ggml_compute_forward_transpose

static void ggml_xrt_nop(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * dst) {
    // NOP
    UNUSED(params);
    UNUSED(dst);
}

// ggml_compute_forward_get_rows

// static void ggml_xrt_get_rows(
//         const struct ggml_compute_params * params,
//         struct ggml_tensor * dst) {

//     ggml_compute_forward_get_rows(params, dst);

//     //static bool first = true;
//     //printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
//     //if (first) {
//     //    first = false;
//     //} else {
//     //    for (int k = 0; k < dst->ne[1]; ++k) {
//     //        for (int j = 0; j < dst->ne[0]/16; ++j) {
//     //            for (int i = 0; i < 16; ++i) {
//     //                printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
//     //            }
//     //            printf("\n");
//     //        }
//     //        printf("\n");
//     //    }
//     //    printf("\n");
//     //    exit(0);
//     //}
// }

// extern "C" void ggml_xrt_rms_norm_f32(const struct ggml_compute_params * params,
//               struct ggml_tensor * dst);

// void ggml_xrt_rms_norm_f32(const struct ggml_compute_params * params,
//               struct ggml_tensor * dst) {

//     // Lock the mutex at the start of the function
//     //std::lock_guard<std::mutex> lock(kernel_mutex);

//     const struct ggml_tensor * src0 = dst->src[0];

//     if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
//         return;
//     }

//     GGML_ASSERT(src0->nb[0] == sizeof(float));

//     GGML_TENSOR_UNARY_OP_LOCALS

//     // Determine next power of two sizes
//     // int padded_ne00 = next_power_of_two(ne00);
//     // int padded_ne01 = next_power_of_two(ne01);

//     // Compute the total size of the tensor
//     int64_t size = ne00 * ne01;

//     // Compute the padded size
//     // int64_t padded_size = padded_ne00 * padded_ne01;
//     float eps;
//     memcpy(&eps, dst->op_params, sizeof(float));

//     GGML_ASSERT(eps > 0.0f);

//     // Declare Buffers
//     auto bo_a = xrt::bo(myDevice, size * sizeof(float), rmsnorm.group_id(0));
//     auto bo_c = xrt::bo(myDevice, size * sizeof(float), rmsnorm.group_id(1));

//     auto bo_a_map = bo_a.map<float*>();
//     auto bo_c_map = bo_c.map<float*>();

//     // Fill the buffers with zeroes
//     std::fill(bo_a_map, bo_a_map + size, 0.0f);
//     std::fill(bo_c_map, bo_c_map + size, 0.0f);

//     for (int64_t i03 = 0; i03 < ne03; i03++)
//     {
//         for (int64_t i02 = 0; i02 < ne02; i02++)
//         {
//             const float * x = (float *) src0->data + i02*nb2 + i03*nb3;
//             ggml_vec_cpy_f32(size, bo_a_map, x);

// #ifndef NDEBUG
//             std::cout << "Execution of the kernel RMS Norm\n";
// #endif
//             // Synchronize input buffer with device
//             bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);

//             // Execute the RMSNorm kernel
//             auto run = rmsnorm(bo_a, bo_c, size, eps);
//             run.wait();

//             // Synchronize output buffer with host
//             bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

// #ifndef NDEBUG
//             std::cout << "Get the output data from the device" << std::endl;
// #endif
//             // Copy Data from Buffers to Tensors
//             float * y = (float *) dst->data + i02*nb2 + i03*nb3;
//             ggml_vec_cpy_f32(size, y, bo_c_map);
//         }  
//     }
// }

// static void ggml_xrt_rms_norm(
//         const struct ggml_compute_params * params,
//         struct ggml_tensor * dst) {

//     const struct ggml_tensor * src0 = dst->src[0];

//     GGML_ASSERT(src0->type == GGML_TYPE_F32);

//     switch (src0->type) {
//         case GGML_TYPE_F32:
//             {
//                 ggml_xrt_rms_norm_f32(params, dst);
//             } break;
//         default:
//             {
//                 ggml_compute_forward_rms_norm(params, dst);
//             } break;
//     }
// }

// static void ggml_xrt_rope(
//         const struct ggml_compute_params * params,
//         struct ggml_tensor * dst) {

//     ggml_compute_forward_rope(params, dst);
// }

// extern "C" void ggml_xrt_soft_max_f32(const struct ggml_compute_params * params,
//               struct ggml_tensor * dst);

// void ggml_xrt_soft_max_f32(const struct ggml_compute_params * params,
//         struct ggml_tensor * dst) {

//     // Lock the mutex at the start of the function
//     //std::lock_guard<std::mutex> lock(kernel_mutex);

//     const struct ggml_tensor * src0 = dst->src[0];
//     const struct ggml_tensor * src1 = dst->src[1];

//     assert(ggml_is_contiguous(dst));
//     assert(ggml_are_same_shape(src0, dst));

//     if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
//         return;
//     }

//     float scale    = 1.0f;
//     float max_bias = 0.0f;

//     memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
//     memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

//     const int ith = params->ith;
//     const int nth = params->nth;

//     GGML_TENSOR_UNARY_OP_LOCALS

//     const uint32_t n_head      = ne02;
//     const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

//     const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
//     const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

//     const int nc = src0->ne[0];
//     const int nr = ggml_nrows(src0);

//     // rows per thread
//     const int dr = (nr + nth - 1)/nth;

//     // row range for this thread
//     const int ir0 = dr*ith;
//     const int ir1 = MIN(ir0 + dr, nr);

//     float * wp = (float *) params->wdata + (nc + CACHE_LINE_SIZE_F32) * ith;

//     // int padded_nc = next_power_of_two(nc);

//     // Initialize XRT device and buffers
//     auto bo_a = xrt::bo(myDevice, nc * sizeof(float), softmax.group_id(0));
//     auto bo_c = xrt::bo(myDevice, nc * sizeof(float), softmax.group_id(1));

//     for (int i1 = ir0; i1 < ir1; i1++) {
//         float * sp = (float *)((char *) src0->data + i1 * src0->nb[1]);
//         float * dp = (float *)((char *)  dst->data +  i1 * dst->nb[1]);

//         float * mp = src1 ? (float *)((char *) src1->data) + (i1 % ne01) * ne00 : NULL;

//         const uint32_t h = (i1/ne01)%ne02; // head
//         const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

//         auto bo_a_map = bo_a.map<float*>();
//         auto bo_c_map = bo_c.map<float*>();

//         ggml_vec_cpy_f32  (nc, wp, sp);
//         ggml_vec_scale_f32(nc, wp, scale);
//         if (mp) {
//             for (int i = 0; i < nc; ++i) {
//                 wp[i] += slope * mp[i];
//             }
//         }

// #ifndef NDEBUG
//         for (int i = 0; i < nc; ++i) {
//             //printf("p[%d] = %f\n", i, p[i]);
//             assert(!isnan(wp[i]));
//         }
// #endif

//         // Fill the buffers with zeroes
//         std::fill(bo_a_map, bo_a_map + nc, 0.0f);
//         std::fill(bo_c_map, bo_c_map + nc, 0.0f);

//         // Copy input data to device buffer
//         ggml_vec_cpy_f32(nc, bo_a_map, wp);

// #ifndef NDEBUG
//         std::cout << "Execution of the kernel Softmax\n";
// #endif

//         // Synchronize buffers to device
//         bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);

//         // Launch the softmax kernel
//         auto run = softmax(bo_a, bo_c, nc);
//         run.wait();

//         // Synchronize buffers from device
//         bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

// #ifndef NDEBUG
//         std::cout << "Get the output data from the device" << std::endl;
// #endif

//         // Copy output data to destination tensor
//         ggml_vec_cpy_f32(nc, dp, bo_c_map);

// #ifndef NDEBUG
//         for (int i = 0; i < nc; ++i) {
//             assert(!isnan(dp[i]));
//             assert(!isinf(dp[i]));
//         }
// #endif
//     }
// }

// static void ggml_xrt_soft_max(
//         const struct ggml_compute_params * params,
//         struct ggml_tensor * dst) {

//     const struct ggml_tensor * src0 = dst->src[0];
//     const struct ggml_tensor * src1 = dst->src[1];

//     GGML_ASSERT(src1->type == GGML_TYPE_F32);

//     switch (src0->type) {
//         case GGML_TYPE_F32:
//             {
//                 ggml_xrt_soft_max_f32(params, dst);
//             } break;
//         default:
//             {
//                 ggml_compute_forward_soft_max(params, dst);
//             } break;
//     }
// }

// extern "C" void ggml_xrt_mul_mat(const struct ggml_compute_params * params,
//               struct ggml_tensor * dst);

// void ggml_xrt_mul_mat(const struct ggml_compute_params * params,
//               struct ggml_tensor * dst) {

//     // Lock the mutex at the start of the function
//     //std::lock_guard<std::mutex> lock(kernel_mutex);
//     const struct ggml_tensor * src0 = dst->src[0]; // Matrix
//     const struct ggml_tensor * src1 = dst->src[1]; // Vector

//     GGML_TENSOR_BINARY_OP_LOCALS

//     const enum ggml_type type = src0->type;
//     const int ith = params->ith;
//     const int nth = params->nth;

//     GGML_ASSERT(ne0 == ne01);
//     GGML_ASSERT(ne1 == ne11);
//     GGML_ASSERT(ne2 == ne12);
//     GGML_ASSERT(ne3 == ne13);

//     // we don't support permuted src0 or src1
//     GGML_ASSERT(nb00 == ggml_type_size(src0->type));
//     GGML_ASSERT(nb10 == ggml_type_size(src1->type));

//     // dst cannot be transposed or permuted
//     GGML_ASSERT(nb0 == sizeof(float));
//     GGML_ASSERT(nb0 <= nb1);
//     GGML_ASSERT(nb1 <= nb2);
//     GGML_ASSERT(nb2 <= nb3);

//     // Determine next power of two sizes
//     // int padded_ne00 = next_power_of_two(ne00); 
//     // int padded_ne01 = next_power_of_two(ne01); 
//     // int padded_ne10 = next_power_of_two(ne10); 
//     // int padded_ne11 = ne11;                    

//     int src0_size = ne00 * ne01;
//     int src1_size = ne10;
//     int dst_size = ne01;
//     const size_t  desired_wsize = ne13 * ne12 * src0_size * sizeof(float);
//     ggml_to_float_t to_float = ggml_internal_get_type_traits(type).to_float;

//     // int padded_size0 = padded_ne00 * padded_ne01;
//     // int padded_size1 = padded_ne10;  
//     // int padded_dst_size = padded_ne00 * padded_ne11;

//     // Allocate XRT buffers
//     auto bo_a = xrt::bo(myDevice, src0_size * sizeof(float), matvecmul.group_id(0));
//     auto bo_b = xrt::bo(myDevice, src1_size * sizeof(float), matvecmul.group_id(1)); // Single row vector size
//     auto bo_c = xrt::bo(myDevice, dst_size * sizeof(float), matvecmul.group_id(2));

//     // Map buffers to host memory
//     auto bo_a_map = bo_a.map<float*>();
//     auto bo_b_map = bo_b.map<float*>();
//     auto bo_c_map = bo_c.map<float*>();

//     // Fill the buffers with zeroes
//     std::fill(bo_a_map, bo_a_map + src0_size, 0.0f);
//     std::fill(bo_b_map, bo_b_map + src1_size, 0.0f);
//     std::fill(bo_c_map, bo_c_map + dst_size, 0.0f);

//     // broadcast factors
//     const int64_t r2 = ne12 / ne02;
//     const int64_t r3 = ne13 / ne03;

//     if (params->type == GGML_TASK_INIT) {
//         if (type != GGML_TYPE_F32)
//         {
//             float * wdata = (float *)params->wdata;
//             const size_t row_size = ggml_row_size(GGML_TYPE_F32, ne00);  // Dequantized row size in float

//             //assert(params->wsize >= ne01*ne02*ne03*row_size);

//             for (int64_t i03 = 0; i03 < ne03; ++i03) {
//                 for (int64_t i02 = 0; i02 < ne02; ++i02) {
//                     for (int64_t i01 = 0; i01 < ne01; ++i01) {
//                         // Dequantize data from src0 (quantized) into wdata (float32)
//                         to_float((char *)src0->data + i03*nb03 + i02*nb02 + i01*nb01, wdata, ne00);
//                         wdata += ne00;  // Move the wdata pointer to the next row
//                     }
//                 }
//             }
//         }
//         return;
//     }

//     if (params->type == GGML_TASK_FINALIZE) {
//         return;
//     }

//     for (int64_t i13 = 0; i13 < ne13; i13++)
//     {
//         for (int64_t i12 = 0; i12 < ne12; i12++)
//         {
//             const int64_t i03 = i13/r3;
//             const int64_t i02 = i12/r2;

//             const void * x = (char *) src0->data + i02*nb02 + i03*nb03;
//             if (type != GGML_TYPE_F32) {
//                 x = (float *) params->wdata + i03 * ne12 * ne01 * ne00 + i02 * ne01 * ne00;
//             }
//             ggml_vec_cpy_f32(src0_size, bo_a_map, (float *)x);

//             for (int64_t row = 0; row < ne11; ++row)
//             {
//                 const float * y = (float *) ((char *) src1->data + i12*nb12 + i13*nb13 + row * nb11);
//                 float * d = (float *) ((char *)  dst->data + i12*nb2  + i13*nb3 + row * nb11);
//                 ggml_vec_cpy_f32(src1_size, bo_b_map, y);

// #ifndef NDEBUG
//                 std::cout << "Execution of the kernel Matmul\n";
// #endif

//                 // Synchronize buffers with device
//                 bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
//                 bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

//                 // Execute the GEMV kernel
//                 auto run = matvecmul(bo_a, bo_b, bo_c, ne01, ne10, 1);
//                 run.wait();

//                 // Synchronize results back to host
//                 bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

// #ifndef NDEBUG
//                 std::cout << "Get the output data from the device" << std::endl;
// #endif

//                 ggml_vec_cpy_f32(dst_size, d, bo_c_map);
//             }          
//         }
//     }
// }

// extern "C" void ggml_xrt_unary_f32(const struct ggml_compute_params * params,
//               struct ggml_tensor * dst);

// void ggml_xrt_unary_f32(const struct ggml_compute_params * params,
//               struct ggml_tensor * dst) {

//     // Lock the mutex at the start of the function
//     //std::lock_guard<std::mutex> lock(kernel_mutex);
        
//     const struct ggml_tensor * src0 = dst->src[0];

//     GGML_ASSERT(src0->nb[0] == sizeof(float));
//     assert(ggml_are_same_shape(src0, dst));

//     if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
//         return;
//     }

//     GGML_TENSOR_UNARY_OP_LOCALS

//     // Determine next power of two sizes
//     // int padded_ne00 = next_power_of_two(ne00);
//     // int padded_ne01 = next_power_of_two(ne01);

//     const enum ggml_unary_op operation = ggml_get_unary_op(dst);
//     int op = 0;
//     if (operation == GGML_UNARY_OP_SILU)
//     {
//         op = 2;
//     }
//     else if (operation == GGML_UNARY_OP_RELU) {
//         op = 1;
//     } else {
//         op = 0;
//     }

//     // Compute the padded size
//     // int64_t padded_size = padded_ne00 * padded_ne01;

//     // Compute the total size of the tensor
//     int64_t size = ne00 * ne01;

//     // Declare Buffers
//     auto bo_a = xrt::bo(myDevice, size * sizeof(float), unary.group_id(0));
//     auto bo_c = xrt::bo(myDevice, size * sizeof(float), unary.group_id(1));

//     auto bo_a_map = bo_a.map<float*>();
//     auto bo_c_map = bo_c.map<float*>();

//     // Fill the buffers with zeroes
//     std::fill(bo_a_map, bo_a_map + size, 0.0f);
//     std::fill(bo_c_map, bo_c_map + size, 0.0f);

//     for (int64_t i03 = 0; i03 < ne03; i03++)
//     {
//         for (int64_t i02 = 0; i02 < ne02; i02++)
//         {
//             // Copy Data from Tensors to Buffers
//             /*for (int64_t i = 0; i < size; ++i) {
//                 bo_a_map[i] = ((float*)src0->data)[i];
//             }*/
//             const float * x = (float *) src0->data + i02*nb2 + i03*nb3;
//             ggml_vec_cpy_f32(size, bo_a_map, x);
// #ifndef NDEBUG
//             std::cout << "Execution of the kernel Unary\n";
// #endif
//             // Synchronize input buffer with device
//             bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);

//             // Execute the RMSNorm kernel
//             auto run = unary(bo_a, bo_c, size, op);
//             run.wait();

//             // Synchronize output buffer with host
//             bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

// #ifndef NDEBUG
//             std::cout << "Get the output data from the device" << std::endl;
// #endif

//             // Copy Data from Buffers to Tensors
//             /*for (int64_t i = 0; i < size; ++i) {
//                 ((float*)dst->data)[i] = bo_c_map[i];
//             }*/
//             float * y = (float *) dst->data + i02*nb2 + i03*nb3;
//             ggml_vec_cpy_f32(size, y, bo_c_map);
//         }  
//     }
// }

// static void ggml_xrt_unary(
//         const struct ggml_compute_params * params,
//         struct ggml_tensor * dst) {

//     const struct ggml_tensor * src0 = dst->src[0];
//     const enum ggml_unary_op op = ggml_get_unary_op(dst);

//     GGML_ASSERT(src0->type == GGML_TYPE_F32);

//     if (src0->type == GGML_TYPE_F32 && (op == GGML_UNARY_OP_SILU || op == GGML_UNARY_OP_RELU)) {
//         ggml_xrt_unary_f32(params, dst);
//     } else {
//         ggml_compute_forward_unary(params, dst);
//     }
// }

// #ifdef XRT_CLOCK
// int operationCounters[GGML_OP_COUNT] = {0};
// struct timespec start_times, end_times;
// #endif

// bool ggml_xrt_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
//     if (tensor->op == GGML_OP_MUL_MAT) {
//        if (tensor->src[0]->ne[3] != tensor->src[1]->ne[3]) {
// #ifndef NDEBUG
//            fprintf(stderr, "%s: cannot compute %s: src0->ne[3] = %" PRId64 ", src1->ne[3] = %" PRId64 " - fallback to CPU\n", __func__, tensor->name, tensor->src[0]->ne[3], tensor->src[1]->ne[3]);
// #endif
//            return false;
//        }
//    }
// #ifdef XRT_CLOCK
//     // Get the starting time
//     int64_t elapsed_ns;
//     clock_gettime(CLOCK_MONOTONIC, &start_times);
// #endif
//    switch (tensor->op) {
//         case GGML_OP_GET_ROWS:
//             ggml_xrt_get_rows(params, tensor);
//             break;
//         case GGML_OP_ADD:
//             ggml_xrt_add(params, tensor);
//             break;
//         case GGML_OP_MUL:
//             ggml_xrt_mul(params, tensor);
//             break;
//         case GGML_OP_UNARY:
//             ggml_xrt_unary(params, tensor);
//             break;
//         case GGML_OP_SOFT_MAX:
//             ggml_xrt_soft_max(params, tensor);
//             break;
//         case GGML_OP_ROPE:
//             ggml_xrt_rope(params, tensor);
//             break;
//         case GGML_OP_RMS_NORM:
//             ggml_xrt_rms_norm(params, tensor);
//             break;
//         case GGML_OP_MUL_MAT:
//             ggml_xrt_mul_mat(params, tensor);
//             break;
//         case GGML_OP_CPY:
//             ggml_xrt_dup(params, tensor);
//             break;
//         case GGML_OP_CONT:
//             ggml_xrt_dup(params, tensor);
//             break;
//         case GGML_OP_NONE:
//         case GGML_OP_RESHAPE:
//         case GGML_OP_VIEW:
//         case GGML_OP_PERMUTE:
//         case GGML_OP_TRANSPOSE:
//             ggml_xrt_nop(params, tensor);
//             break;
//         case GGML_OP_DUP:
//             //func = ggml_xrt_dup;
//             //break;
//         case GGML_OP_ACC:
//             //func = ggml_xrt_acc;
//             //break;
//         case GGML_OP_DIV:
//             //func = ggml_xrt_div;
//             //break;
//         case GGML_OP_REPEAT:
//             //func = ggml_xrt_repeat;
//             //break;
//         case GGML_OP_NORM:
//             //func = ggml_xrt_norm;
//             //break;
//         case GGML_OP_GROUP_NORM:
//             //func = ggml_xrt_group_norm;
//             //break;
//         case GGML_OP_CONCAT:
//             //func = ggml_xrt_concat;
//             //break;
//         case GGML_OP_UPSCALE:
//             //func = ggml_xrt_upscale;
//             //break;
//         case GGML_OP_PAD:
//             //func = ggml_xrt_pad;
//             //break;
//         case GGML_OP_LEAKY_RELU:
//             //func = ggml_xrt_leaky_relu;
//             //break;
//         case GGML_OP_DIAG_MASK_INF:
//             //func = ggml_xrt_diag_mask_inf;
//             //break;
//         case GGML_OP_ALIBI:
//             //func = ggml_xrt_alibi;
//             //break;
//         case GGML_OP_IM2COL:
//             //func = ggml_xrt_im2col;
//             //break;
//         case GGML_OP_MUL_MAT_ID:
//             //ggml_xrt_nop(params, tensor);
//             //break;
//         case GGML_OP_SUM_ROWS:
//             //func = ggml_xrt_sum_rows;
//             //break;
//         case GGML_OP_ARGSORT:
//             //func = ggml_xrt_argsort;
//             //break;
//         case GGML_OP_SCALE:
//             //func = ggml_xrt_scale;
//             //break;
//         case GGML_OP_SQR:
//             //func = ggml_xrt_sqr;
//             //break;
//         case GGML_OP_CLAMP:
//             //func = ggml_xrt_clamp;
//             //break;
//         default:
//             return false;
//     }
//     #ifdef XRT_CLOCK
//     // Get the ending time
//     clock_gettime(CLOCK_MONOTONIC, &end_times);
//     // Calculate the elapsed time in nanoseconds
//     elapsed_ns = (end_times.tv_sec - start_times.tv_sec) * BILLION + (end_times.tv_nsec - start_times.tv_nsec);
//     operationCounters[tensor->op]++;
//     printf("Operation %d executed in %llu nanoseconds. Count: %d\n", tensor->op, elapsed_ns, operationCounters[tensor->op]);
//     #endif
//     return true;
// }

int ggml_xrt_get_device_count() {
    int device_count = 1; //Here comes the device
    return device_count;
}

GGML_CALL void ggml_init_xrt() {
    static bool initialized = false;

    if (!initialized) {

        int user_device_id = g_main_device;

        //hardcode, force set to 1 device
        g_device_count = 1;
        ggml_xrt_set_device(user_device_id);
        fprintf(stderr, "Using Device %d\n", user_device_id);

        // for (int id = 0; id < g_all_sycl_device_count; ++id) {
        //     GGML_SYCL_DEBUG("id=%d  g_device_caps[%d].device_id=%d g_sycl_device_id2index[%d].index=%d ", id, id,
        //     g_device_caps[id].device_id, id, g_sycl_device_id2index[id].index);
        // }

        initialized = true;
        g_xrt_loaded = true;
    }
}

GGML_CALL void ggml_end_xrt() {
    g_xrt_loaded = false;
}

////////////////////////////////////////////////////////////////////////////////

// backend interface goes here