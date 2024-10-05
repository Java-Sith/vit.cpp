# Setting relevant variables
ROOT_DIR=$(realpath $(dir $(lastword $(MAKEFILE_LIST))))
TARGET := hw
PLATFORM ?= $(ROOT_DIR)/../kv260_custom.xpfm

# Setting input file
HLS_FILES := elementwise.cpp unary.cpp softmax.cpp rmsnorm.cpp matvecmul.cpp
HLS_FILES_NAMES := elementwise unary softmax rmsnorm matvecmul

# Setting directories
TEMP_DIR := ./tmp.$(TARGET)
BUILD_DIR := ./build.$(TARGET)
PACKAGE_OUT = ./package.$(TARGET)
LINK_OUTPUT := $(BUILD_DIR)/kernels.link.xclbin
EMCONFIG_DIR = $(TEMP_DIR)
HLS_BUILD = $(TEMP_DIR)/%.xo

# Packaging
XCL_BIN := $(PACKAGE_OUT)/kernels.xclbin

# Output files
HLS_KERNEL_FILES := $(addprefix $(TEMP_DIR)/,$(HLS_FILES:.cpp=.xo))

# Setting V++ Flags
VPP_PFLAGS :=
VPP_LDFLAGS :=
VPP_FLAGS += --save-temps --jobs 6

# Set the desired clock frequency (e.g., 200 MHz)
KERNEL_FREQ := 250

ifdef USE_FP32
	VPP_FLAGS += -DUSE_FLOAT32
endif

ifdef USE_FP16
	VPP_FLAGS += -DUSE_FLOAT16
endif

ifdef USE_FP8
	VPP_FLAGS += -DUSE_FLOAT8
endif

ifdef USE_FP4
	VPP_FLAGS += -DUSE_FLOAT4
endif

ifdef USE_FX16
	VPP_FLAGS += -DUSE_FIXED16
endif

ifdef USE_LUT
	VPP_FLAGS += -DUSE_LUT
endif

RMDIR = rm -rf

.PHONY: all clean cleanall hls-build

all: build

.PHONY: build
build: check_platform emconfig $(LINK_OUTPUT) $(XCL_BIN)

# Rules for creating the HW
check_platform:
ifndef PLATFORM
	$(error PLATFORM not set. Please set the PLATFORM properly and rerun. Run "make help" for more details.)
endif

$(TEMP_DIR)/%.xo: %.cpp
	mkdir -p $(TEMP_DIR)
	v++ -c $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) -k $(<:.cpp=) --temp_dir $(TEMP_DIR) --kernel_frequency $(KERNEL_FREQ) -I'$(<D)' -o '$@' '$<'

$(LINK_OUTPUT): $(HLS_KERNEL_FILES)
	mkdir -p $(BUILD_DIR)
	v++ -l $(VPP_FLAGS) $(VPP_LDFLAGS) -t $(TARGET) --platform $(PLATFORM) --temp_dir $(TEMP_DIR) --kernel_frequency $(KERNEL_FREQ) -o'$(LINK_OUTPUT)' $^

hls-build: check_platform emconfig $(HLS_KERNEL_FILES)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(PLATFORM) --od $(EMCONFIG_DIR)

$(XCL_BIN): $(LINK_OUTPUT)
	mkdir -p $(PACKAGE_OUT)
	v++ -p $(LINK_OUTPUT) $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) --package.out_dir $(PACKAGE_OUT) --kernel_frequency $(KERNEL_FREQ) -o $(XCL_BIN)

# Cleaning rules
clean:
	$(RMDIR) $(EXECUTABLE) *.xclbin/{*sw_emu*,*hw_emu*}
	$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv
	$(RMDIR) src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	$(RMDIR) build_dir* sd_card*
	$(RMDIR) package.* tmp.*
	$(RMDIR) _x* *xclbin.run_summary qemu-memory-_* emulation _vimage pl* start_simulation.sh *.xclbin
