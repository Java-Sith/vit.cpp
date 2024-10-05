#include <stdio.h>
#include "mat-mul.h"

//Naive implementation of Mul Mat
void mul_mat(float*  src0, float*  src1, float *dst, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int l = 0; l < k; l++) {
                sum += src0[i*k + l] * src1[j*k + l];
            }
            dst[i*n + j] = sum;
        }
    }
}

static inline int quantize_4_blocks_per_row(int k) {
    return k/QK;
}

static inline int quantize_4_row_size(int k) {
    const int nb = quantize_4_blocks_per_row(k);

    return nb*(2*sizeof(gq_scale_t) + QK/2);
}

void vec_dot_gq_4 (const int n, float *  s, const float *  x, const float *  y) {
    const int nb = quantize_4_blocks_per_row(n);

    const gq_scale_t *  pm0 = (const gq_scale_t *) x;
    const gq_scale_t *  pm1 = (const gq_scale_t *) y;

    const gq_scale_t *  pd0 = pm0 + nb;
    const gq_scale_t *  pd1 = pm1 + nb;

    const uint8_t *  pb0 = (const uint8_t *) (pd0 + nb);
    const uint8_t *  pb1 = (const uint8_t *) (pd1 + nb);

    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        const float m0 = GGML_GQ_TO_FP32(pm0[i]);
        const float d0 = GGML_GQ_TO_FP32(pd0[i]);

        const float m1 = GGML_GQ_TO_FP32(pm1[i]);
        const float d1 = GGML_GQ_TO_FP32(pd1[i]);

        const uint8_t *  p0 = pb0 + i*QK/2;
        const uint8_t *  p1 = pb1 + i*QK/2;

        for (int j = 0; j < QK/2; j++) {
            const uint8_t v0 = p0[j];
            const uint8_t v1 = p1[j];

            const float f0 = d0*(v0 & 0xf) + m0;
            const float f1 = d0*(v0 >> 4)  + m0;

            const float f2 = d1*(v1 & 0xf) + m1;
            const float f3 = d1*(v1 >> 4)  + m1;

            sumf += f0*f2 + f1*f3;
        }
    }
    *s = sumf;
}

void mul_mat_gq_4(const float * src0, const float * src1, float * dst, int m, int n, int k) {
    assert(k % QK == 0);
    for (int ir0 = 0; ir0 < m; ir0++) {
        for (int ir1 = 0; ir1 < n; ir1++) {
            vec_dot_gq_4(k, dst + ir1, src0, src1);
            src1 = (const char *) src1 + quantize_4_row_size(k);
        }
        src0 = (const char *) src0 +   quantize_4_row_size(k);
        src1 = (const char *) src1 - n*quantize_4_row_size(k);
        dst = (float *) dst + n;
    }
}

void printMatrix(float *matrix, int M, int N) {
    // Print matrix contents
    printf("Matrix Contents:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f\t", matrix[i * N + j]);
        }
        printf("\n");
    }
}

void saveMatrixToFile(float *matrix, int M, int N, const char *filename) {
    // Save matrix contents to file
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(file, "%.2f\t", matrix[i * N + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

float* loadMatrixFromFile(int M, int N, const char *filename) {
    // Open file for reading
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file!\n");
        return NULL;
    }

    // Allocate memory for the matrix
    float *matrix = (float *)malloc(M * N * sizeof(float));
    if (matrix == NULL) {
        printf("Memory allocation failed!\n");
        fclose(file);
        return NULL;
    }

    // Read matrix values from file
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fscanf(file, "%f", &matrix[i * N + j]) != 1) {
                printf("Error reading file!\n");
                fclose(file);
                free(matrix);
                return NULL;
            }
        }
    }

    // Close file
    fclose(file);

    return matrix;
}

float* loadMatrix(int M, int N) {
    // Allocate memory for the matrix
    float *matrix = (float *)malloc(M * N * sizeof(float));
    if (matrix == NULL) {
        printf("Memory allocation failed!\n");
        return NULL;
    }
}