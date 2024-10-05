# List of source files and corresponding executables
SOURCES := elementwise.cpp rmsnorm.cpp unary.cpp softmax.cpp matvecmul.cpp
EXECUTABLES := elementwise rmsnorm unary softmax matvecmul

CFLAGS := -I/opt/xilinx/xrt/include -Wall -std=c++17 -O0 -g -I. -I./hls-types/include
LDFLAGS := -L/opt/xilinx/xrt/lib -pthread -lxrt_core -lxrt_coreutil

.PHONY: all clean

# Default target to build all executables
all: $(EXECUTABLES)

# Rule to build each executable from its corresponding source file
$(EXECUTABLES): %: %.cpp
	$(CXX) -o $@ $< $(CFLAGS) $(LDFLAGS)

# Clean target to remove all executables
clean:
	$(RM) $(EXECUTABLES)

