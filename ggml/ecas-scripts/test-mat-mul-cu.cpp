#include <iostream>
#include <string>
#include "mat-mul.h"

#if defined(GGML_USE_HIPBLAS)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#ifdef __HIP_PLATFORM_AMD__
// for rocblas_initialize()
#include "rocblas/rocblas.h"
#endif // __HIP_PLATFORM_AMD__
#endif

const int M = 1280;
const int N = 1536;
const int K = 1280;

#if defined(GGML_USE_HIPBLAS)
#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPBLAS_ERROR
#define CHECK_HIPBLAS_ERROR(error)                              \
    if(error != HIPBLAS_STATUS_SUCCESS)                         \
    {                                                           \
        fprintf(stderr, "hipBLAS error: ");                     \
        if(error == HIPBLAS_STATUS_NOT_INITIALIZED)             \
            fprintf(stderr, "HIPBLAS_STATUS_NOT_INITIALIZED");  \
        if(error == HIPBLAS_STATUS_ALLOC_FAILED)                \
            fprintf(stderr, "HIPBLAS_STATUS_ALLOC_FAILED");     \
        if(error == HIPBLAS_STATUS_INVALID_VALUE)               \
            fprintf(stderr, "HIPBLAS_STATUS_INVALID_VALUE");    \
        if(error == HIPBLAS_STATUS_MAPPING_ERROR)               \
            fprintf(stderr, "HIPBLAS_STATUS_MAPPING_ERROR");    \
        if(error == HIPBLAS_STATUS_EXECUTION_FAILED)            \
            fprintf(stderr, "HIPBLAS_STATUS_EXECUTION_FAILED"); \
        if(error == HIPBLAS_STATUS_INTERNAL_ERROR)              \
            fprintf(stderr, "HIPBLAS_STATUS_INTERNAL_ERROR");   \
        if(error == HIPBLAS_STATUS_NOT_SUPPORTED)               \
            fprintf(stderr, "HIPBLAS_STATUS_NOT_SUPPORTED");    \
        if(error == HIPBLAS_STATUS_INVALID_ENUM)                \
            fprintf(stderr, "HIPBLAS_STATUS_INVALID_ENUM");     \
        if(error == HIPBLAS_STATUS_UNKNOWN)                     \
            fprintf(stderr, "HIPBLAS_STATUS_UNKNOWN");          \
        fprintf(stderr, "\n");                                  \
        exit(EXIT_FAILURE);                                     \
    }
#endif
#endif

int main(int argc, char* argv[]) {
    assert(sizeof(gq_quant_t)*8 == gq_t_bits);
    int m = M, n = N, k = K;

    ggml_time_init();

    float * src0  = loadMatrixFromFile(m, k, "ecas-scripts/tensor1.txt");
    float * src1  = loadMatrixFromFile(k, n, "ecas-scripts/tensor2.txt");
    float* dst = loadMatrix(m, n);

    double iM = 1.0/m;
    double sum = 0.0f;

    #if defined(GGML_USE_HIPBLAS)
        // allocate memory on device
        float *dsrc0, *dsrc1, *ddst;
        CHECK_HIP_ERROR(hipMalloc(&dsrc0, M * K * sizeof(float)));
        CHECK_HIP_ERROR(hipMalloc(&dsrc1, N * K * sizeof(float)));
        CHECK_HIP_ERROR(hipMalloc(&ddst, M * N * sizeof(float)));

        // copy matrices from host to device
        CHECK_HIP_ERROR(hipMemcpy(dsrc0, src0, sizeof(float) * M * K, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dsrc1, src1, sizeof(float) * N * K, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(ddst, dst, sizeof(float) * M * N, hipMemcpyHostToDevice));

        hipblasHandle_t handle;
        CHECK_HIPBLAS_ERROR(hipblasCreate(&handle));
    #endif

    int method = 0;
    if (argc > 1) {
        method = std::stoi(argv[1]);
    }

    const int64_t start = ggml_cycles();
    const int64_t start_us = ggml_time_us();

    float alpha = 1.0f;
    float beta = 0.0f;

    if (method == 0) {
        #if defined(GGML_USE_HIPBLAS)
            CHECK_HIPBLAS_ERROR(
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, dsrc0, m, dsrc1, k, &beta, ddst, m));
            CHECK_HIP_ERROR(hipMemcpy(dst, ddst, sizeof(float) * m * n, hipMemcpyDeviceToHost));
            saveMatrixToFile((gq_scale_t *) dst, m, n, "ecas-scripts/result.txt");
        #else
            mul_mat(src0, src1, dst, m, n, k);
            saveMatrixToFile((gq_scale_t *) dst, m, n, "ecas-scripts/result.txt");
        #endif
    }

    if (method == 1) {
        #if defined(GGML_USE_HIPBLAS)
            CHECK_HIPBLAS_ERROR(
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, dsrc0, m, dsrc1, k, &beta, ddst, m));
            CHECK_HIP_ERROR(hipMemcpy(dst, ddst, sizeof(float) * m * n, hipMemcpyDeviceToHost));
            saveMatrixToFile((gq_scale_t *) dst, m, n, "ecas-scripts/result.txt");
        #else
            mul_mat_gq_4(src0, src1, dst, m, n, k);
            saveMatrixToFile((gq_scale_t *) dst, m, n, "ecas-scripts/result.txt");
        #endif
    }
    for (int i = 0; i < n; i++) {
        sum += dst[i]*iM;
    }

    const int64_t end = ggml_cycles();
    const int64_t end_us = ggml_time_us();

    for (int i = 0; i < 16; ++i) {
        printf("%f\n", dst[i]);
    }

    printf("%s: elapsed ticks: %" PRIu64 "\n",  __func__, end - start);
    printf("%s: elapsed us:    %d / %f ms\n",  __func__, (int)(end_us - start_us), (end_us - start_us) / 1000.0);
    printf("%f\n", sum);

    #if defined(GGML_USE_HIPBLAS)
        CHECK_HIP_ERROR(hipFree(dsrc0));
        CHECK_HIP_ERROR(hipFree(dsrc1));
        CHECK_HIP_ERROR(hipFree(ddst));
        CHECK_HIPBLAS_ERROR(hipblasDestroy(handle));
    #endif

    free(src0);
    free(src1);
    free(dst);

    return 0;
}
