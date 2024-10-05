#include <iostream>
#include <string>
#include "mat-mul.h"

#if defined(GGML_USE_OPENBLAS)
#include <cblas.h>
#endif

#ifdef GGML_USE_OPENBLAS
void mul_mat_blas(const float * src0, const float * src1, float * dst, int m, int n, int k) {
    assert(k % QK == 0);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, src0, k, src1, n, 0.0f, dst, n);
}
#endif

// Function to generate a matrix with given rows and columns
void generateMatrix(float* matrix, int rows, int cols) {
    srand(static_cast<unsigned>(time(0))); // Seed for random number generation

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand() % 1000) / 100.0f; // Generate random float values between 0.0 and 9.99
    }
}

int main(int argc, char* argv[]) {
    assert(sizeof(gq_quant_t)*8 == gq_t_bits);

    ggml_time_init();

    // Check for the correct number of arguments
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K>" << std::endl;
        return 1;
    }

    // Parse command line arguments
    int M = atoi(argv[1]); // Number of rows for matrixA and expected_result
    int N = atoi(argv[2]); // Number of columns for matrixB and expected_result
    int K = atoi(argv[3]); // Number of columns for matrixA and rows for matrixB

    // Allocate memory for the matrices
    float* src0 = new float[M * K];
    float* src1 = new float[K * N];
    float* dst = new float[M * N];

    generateMatrix(src0, M, K);
    generateMatrix(src1, N, K);

    int m = M, n = N, k = K;

    double iM = 1.0/m;
    double sum = 0.0f;

    int method = 0;
    const int64_t start = ggml_cycles();
    const int64_t start_ms = ggml_time_ms();

    if (method == 0) {
        #ifdef GGML_USE_OPENBLAS
            mul_mat_blas(src0, src1, dst, m, n, k);
            //saveMatrixToFile((gq_scale_t *) dst, m, n, "ecas-scripts/result.txt");
        #else
            mul_mat(src0, src1, dst, m, n, k);
            //saveMatrixToFile((gq_scale_t *) dst, m, n, "ecas-scripts/result.txt");
        #endif
    }

    if (method == 1) {
        #ifdef GGML_USE_OPENBLAS
            mul_mat_blas(src0, src1, dst, m, n, k);
            //saveMatrixToFile((gq_scale_t *) dst, m, n, "ecas-scripts/result.txt");
        #else
            mul_mat_gq_4(src0, src1, dst, m, n, k);
            //saveMatrixToFile((gq_scale_t *) dst, m, n, "ecas-scripts/result.txt");
        #endif
    }

    const int64_t end_ms = ggml_time_ms();

    for (int i = 0; i < n; i++) {
        sum += dst[i]*iM;
    }

    const int64_t end = ggml_cycles();

    printf("%s: elapsed ticks: %" PRIu64 "\n",  __func__, end - start);
    printf("%s: elapsed ms:    %d\n",  __func__, (int)(end_ms - start_ms));
    printf("%f\n", sum);

    delete[] src0;
    delete[] src1;
    delete[] dst;

    return 0;
}
