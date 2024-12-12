# RCKangaroo Makefile

# Derleyici ayarları
CXX := g++
NVCC := nvcc

# CUDA ayarları
CUDA_PATH ?= /usr/local/cuda

# Derleme bayrakları
CXXFLAGS := -O3 -std=c++11 -I. -I$(CUDA_PATH)/include
NVCCFLAGS := -O3 -std=c++11 -I. -Xcompiler -fPIC -gencode arch=compute_86,code=sm_86

# Bağlama bayrakları
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart

# Kaynak dosyalar
CPP_SOURCES := \
    Ec.cpp \
    GpuKang.cpp \
    RCKangaroo.cpp \
    utils.cpp

CU_SOURCES := \
    RCGpuCore.cu

# Nesne dosyaları
CPP_OBJECTS := $(CPP_SOURCES:.cpp=.o)
CU_OBJECTS := $(CU_SOURCES:.cu=.o)

# Hedef dosya
TARGET := RCKangaroo

# Kurallar
all: $(TARGET)

$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(CPP_OBJECTS) $(CU_OBJECTS) $(TARGET) 