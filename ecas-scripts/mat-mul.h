#ifndef MAT_MUL_CU_H
#define MAT_MUL_CU_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ggml.h"
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(GGML_GQ_USE_FP16_SCALE)
#define gq_scale_t ggml_fp16_t
#define GGML_FP32_TO_GQ(x) ggml_fp32_to_fp16(x)
#define GGML_GQ_TO_FP32(x) ggml_fp16_to_fp32(x)
#else
#define gq_scale_t float
#define GGML_FP32_TO_GQ(x) (x)
#define GGML_GQ_TO_FP32(x) (x)
#endif

#define QK 64
#define QB 4
#define gq_t_bits 64
#define gq_quant_t uint64_t

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void mul_mat(float*  src0, float*  src1, float *dst, int m, int n, int k);
void vec_dot_gq_4 (const int n, float *  s, const float *  x, const float *  y);
void mul_mat_gq_4(const float * src0, const float * src1, float * dst, int m, int n, int k);
void printMatrix(float *matrix, int M, int N);
void saveMatrixToFile(float *matrix, int M, int N, const char *filename);
float* loadMatrixFromFile(int M, int N, const char *filename);
float* loadMatrix(int M, int N);

#ifdef __cplusplus
}
#endif

#endif // MY_C_FUNCTIONS_H
