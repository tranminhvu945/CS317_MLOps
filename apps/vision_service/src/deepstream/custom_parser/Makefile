# Makefile: libnvdsinfer_custom_impl_Yolo.so
# Builds Yolo.cpp → libnvdsinfer_custom_impl_Yolo.so
#
# Usage:
#   make              # incremental build (fast, only rebuilds changed files)
#   make -j$(nproc)  # parallel build (fastest)
#   make clean        # remove artifacts
#
# For Dockerfile, use:
#   make -j$(nproc)
#
# ─────────────────────────────────────────────────────────────────────────────
# Paths (absolute so Makefile works from any working directory)
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
WORKSPACE    ?= /workspace
DS_LIB       := $(WORKSPACE)/apps/vision_service/libs/deepstream/lib

# CUDA in DeepStream 6.4 container: headers at targets/x86_64-linux/include
# Not at /usr/local/cuda/include (which does not exist)
CUDA_HOME    ?= /usr/local/cuda-12.1
CUDA_INCLUDE := $(CUDA_HOME)/targets/x86_64-linux/include
CUDA_LIB     := $(CUDA_HOME)/targets/x86_64-linux/lib

# ─────────────────────────────────────────────────────────────────────────────
# Compiler
# ─────────────────────────────────────────────────────────────────────────────
CXX          := g++
CXXFLAGS     := -std=c++17 -O2 -fPIC -fvisibility=hidden \
                -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable
INCLUDES    := -I/opt/nvidia/deepstream/deepstream-6.4/sources/includes \
                -I$(CUDA_INCLUDE)

# Auto-generate dependencies: header changes trigger rebuild too
DEPFLAGS     := -MMD -MP
LDFLAGS      := -shared

# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_LIB   := $(DS_LIB)/libnvdsinfer_custom_impl_Yolo.so

# ─────────────────────────────────────────────────────────────────────────────
# Sources & objects
# ─────────────────────────────────────────────────────────────────────────────
SOURCES      := $(wildcard $(SCRIPT_DIR)*.cpp)
OBJECTS      := $(SOURCES:$(SCRIPT_DIR)%.cpp=$(SCRIPT_DIR)%.o)
DEPS         := $(OBJECTS:%.o=%.d)

# ─────────────────────────────────────────────────────────────────────────────
# Linker libs
# ─────────────────────────────────────────────────────────────────────────────
NVINFER_LIBS := -L/opt/nvidia/deepstream/deepstream-6.4/lib \
                -L$(CUDA_LIB) \
                -lnvds_infer \
                -lnvinfer \
                -lnvinfer_plugin \
                -lnvonnxparser \
                -lnvparsers
LIBMISC      := -lstdc++ -lm

# ─────────────────────────────────────────────────────────────────────────────
# Targets
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: all clean info

all: $(OUTPUT_LIB)

info:
	@echo "SOURCES  = $(SOURCES)"
	@echo "OBJECTS  = $(OBJECTS)"
	@echo "OUTPUT   = $(OUTPUT_LIB)"
	@echo "CXXFLAGS = $(CXXFLAGS) $(INCLUDES)"

# ─────────────────────────────────────────────────────────────────────────────
# Build rules
# ─────────────────────────────────────────────────────────────────────────────

$(OUTPUT_LIB): $(OBJECTS) | $(DS_LIB)
	@echo "Linking $@ ← $(words $(OBJECTS)) object file(s)"
	$(CXX) $(LDFLAGS) -o $@ $(OBJECTS) $(NVINFER_LIBS) $(LIBMISC)

$(SCRIPT_DIR)%.o: $(SCRIPT_DIR)%.cpp
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

# Ensure output directory exists (for `make` called from Dockerfile)
$(DS_LIB):
	install -d $(DS_LIB)

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
clean:
	@echo "Cleaning Yolo objects and .so"
	rm -f $(SCRIPT_DIR)*.o $(SCRIPT_DIR)*.d
	rm -f $(OUTPUT_LIB)
	@echo "Done."

# ─────────────────────────────────────────────────────────────────────────────
# Safety nets
# ─────────────────────────────────────────────────────────────────────────────
# Abort if any command fails (prevents half-built artifacts)
.DELETE_ON_ERROR:
# .o files are intermediate — delete them when done linking
.INTERMEDIATE: $(OBJECTS)
# Include auto-generated dependencies (silently ignore if missing)
-include $(DEPS)
