#pragma once

#include "ggml.h"
#include "ggml-impl.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_XRT_MAX_DEVICES 1
#define GGML_XRT_NAME "XRT"
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

void ggml_xrt_add_f32(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_xrt_mul_f32(const struct ggml_compute_params * params, struct ggml_tensor * dst);
GGML_API void ggml_init_xrt(void);
GGML_API void ggml_end_xrt(void);
GGML_API bool   ggml_xrt_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);

#ifdef  __cplusplus
}
#endif
