#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

//#define GGML_USE_CUBLAS // uncomment this to use cuda backend, make sure build ggml lib with GGML_CUBLAS=ON

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>

#if false
struct test_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_model(test_model & model, float* a, float* b, int M, int N, int K, bool use_gpu = false) {
    size_t buffer_size = 0;
    {
        buffer_size += (M * N) * ggml_type_size(GGML_TYPE_F32); // tensor a
        buffer_size += (N * K) * ggml_type_size(GGML_TYPE_F32); // tensor b
        buffer_size += 1024; // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %d bytes\n", __func__, (int) buffer_size);

    int num_tensors = 2;
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    // initialize the backend
#ifdef GGML_USE_CUBLAS
    if (use_gpu) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init(0);
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (use_gpu) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    if(!model.backend) {
        // fallback to CPU backend
        model.backend = ggml_backend_cpu_init();
    }

    model.buffer = ggml_backend_alloc_buffer(model.backend, buffer_size);

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, K, M);
    printf("Matrix A: [%i, %i]\n", K, M);
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, K, N);
    printf("Matrix B: [%i, %i]\n", K, N);

    // create a allocator
    ggml_tallocr_t alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(alloc, model.a);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.a->data, a, ggml_nbytes(model.a));
    } else {
        ggml_backend_tensor_set(model.a, a, 0, ggml_nbytes(model.a)); // cuda requires copy the data directly to device
    }

    // alloc memory
    ggml_tallocr_alloc(alloc, model.b);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.b->data, b, ggml_nbytes(model.b));
    } else {
        ggml_backend_tensor_set(model.b, b, 0, ggml_nbytes(model.b));  // cuda requires copy the data directly to device
    }

    ggml_tallocr_free(alloc);
}

struct ggml_cgraph * build_graph(const test_model& model) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // zT = x @ yT
    struct ggml_tensor * result = ggml_mul_mat(ctx0, model.a, ggml_cont(ctx0, model.b));

    // z = (zT)T
    ggml_build_forward_expand(gf, ggml_cont(ctx0, ggml_transpose(ctx0, result)));

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

struct ggml_tensor* compute(const test_model & model, ggml_gallocr_t allocr) {
    struct ggml_cgraph * gf = build_graph(model);

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);
    int n_threads = 1;

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif

    ggml_backend_graph_compute(model.backend, gf);

    //ggml_graph_print(gf);

    // in this case, the output tensor is the last one in the graph
    return gf->nodes[gf->n_nodes - 1];
}
#endif

// Function to generate a matrix with given rows and columns
void generateMatrix(float* matrix, int rows, int cols) {
    srand(static_cast<unsigned>(time(0))); // Seed for random number generation

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand() % 1000) / 100.0f; // Generate random float values between 0.0 and 9.99
    }
}


static void ggml_vec_dot(const int n, float * s, float * x, float * y) {
    float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += x[i] * y[i];
    }
    *s = sumf;
}

static void gemm(int m, int n, int k,
                             float * A,
                             float * B,
                             float * C,
                             const int ith, const int nth) {
    // does not seem to make a difference
    int m0, m1, n0, n1;
    // patches per thread
    if (m > n) {
        n0 = 0;
        n1 = n;

        // total patches in dst
        const int np = m;

        // patches per thread
        const int dp = (np + nth - 1)/nth;

        // patch range for this thread
        m0 = dp*ith;
        m1 = std::min(m0 + dp, np);
    } else {
        m0 = 0;
        m1 = m;

        // total patches in dst
        const int np = n;

        // patches per thread
        const int dp = (np + nth - 1)/nth;

        // patch range for this thread
        n0 = dp*ith;
        n1 = std::min(n0 + dp, np);
    }

    for (int i = m0; i < m1; i++) {
        for (int j = n0; j < n1; j++) {
            ggml_vec_dot(k, C + i * n + j, A + i * k, B + j * k);
        }
    }
}


void perform_gemm_test(float* a, float* b, int M, int N, int K) {
    printf("\nPerforming gemm_f16_out_f32 test:\n");

    auto start = std::chrono::high_resolution_clock::now();

    float* gemm_out = new float[M * N];
    gemm(M, N, K, a, b, gemm_out, 0, 1);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed_seconds = end - start;
    auto elapsed_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Time in seconds: " << elapsed_seconds.count() << " s\n";
    std::cout << "Time in milliseconds: " << elapsed_milliseconds.count() << " ms\n";
}

int main(int argc, char* argv[])
{
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
    float* matrixA = new float[M * K];
    float* matrixB = new float[K * N];
    float* expected_result = new float[M * N];

    // Generate random matrices
    generateMatrix(matrixA, M, K);
    generateMatrix(matrixB, K, N);

    perform_gemm_test(matrixA, matrixB, M, N, K);

#if false
    test_model model;
    load_model(model, matrixA, matrixB, M, N, K, true);

    ggml_gallocr_t allocr = NULL;

    {
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        //create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph(model);

        // compute the required memory
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);
        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0f/1024.0f);
    }

    struct ggml_tensor * result = compute(model, allocr);

    float* out_data = new float[ggml_nelements(result)];

    ggml_backend_tensor_get(result, out_data, 0, ggml_nbytes(result));

    printf("\nPerforming ggml_mul_mat test:\n");

    passed = true;
    for(int i = 0; i < M * N; i++) {
        if(out_data[i] != expected_result[i]) {
            passed = false;
            break;
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.1f ", out_data[i * N + j]);
        }
        printf("\n");
    }

    printf("ggml_mul_mat (%d): %s\n", (int) ggml_nelements(result), passed && (ggml_nelements(result) == M * N) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");

   // free memory
    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(allocr);
#endif

    delete[] matrixA;
    delete[] matrixB;
    delete[] expected_result;
    return 0;
}
