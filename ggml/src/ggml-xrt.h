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

GGML_API void ggml_init_xrt(void);
GGML_API void ggml_end_xrt(void);
GGML_API bool   ggml_xrt_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);
GGML_API ggml_backend_t ggml_backend_xrt_init(int device);
GGML_API ggml_backend_buffer_type_t ggml_backend_xrt_buffer_type(int device);
GGML_API ggml_backend_buffer_type_t ggml_backend_xrt_host_buffer_type(void);
GGML_API void   ggml_backend_xrt_print_xrt_devices(void);
GGML_API GGML_CALL void   ggml_xrt_get_device_description(int device, char *description, size_t description_size);
#ifdef  __cplusplus
}
#endif
