TARGET    := lambda

NVCC_BIN  := $(shell which nvcc 2>/dev/null)
HIPCC_BIN := $(shell which hipcc 2>/dev/null)
ifeq ($(NVCC_BIN),)
  NVCC_BIN := $(shell ls /opt/conda/bin/nvcc 2>/dev/null || ls /usr/local/cuda/bin/nvcc 2>/dev/null)
endif
ifeq ($(HIPCC_BIN),)
  HIPCC_BIN := $(shell ls /opt/rocm*/bin/hipcc 2>/dev/null | head -n 1)
endif

SRC_CPP   := modinv.cpp lambda.cpp secp256k1.cpp lambda_kernel.cpp
LDLIBS    := -lpthread -ldl -lrt -lcrypto

# Auto-detect architecture compiler
ifneq ($(NVCC_BIN),)
  CXX       := $(NVCC_BIN)
  CXXFLAGS  := -O3 -std=c++14 -I. -MD -Xcompiler -O3,-ffast-math -DUSE_NVCC -rdc=true
  
  # Only compile files that contain __device__ code as CUDA
  secp256k1.o: CXXFLAGS += -x cu --expt-relaxed-constexpr
  lambda_kernel.o: CXXFLAGS += -x cu --expt-relaxed-constexpr
else ifneq ($(HIPCC_BIN),)
  CXX       := $(HIPCC_BIN)
  CXXFLAGS  := -O3 -ffast-math -std=c++14 -I. -MD
else
  CXX       := g++
  CXXFLAGS  := -O3 -ffast-math -std=c++14 -I. -MD -D__device__= -D__host__= -D__global__= -I./hip
  SRC_CPP   += hip_mock.cpp
endif

ifneq (,$(findstring mock,$(MAKECMDGOALS)))
  CXX       := g++
  CXXFLAGS  := -O3 -ffast-math -std=c++14 -I. -MD -D__device__= -D__host__= -D__global__= -I./hip
  ifeq (,$(findstring hip_mock.cpp,$(SRC_CPP)))
    SRC_CPP += hip_mock.cpp
  endif
endif

OBJ_CPP   := $(SRC_CPP:.cpp=.o)
OBJ       := $(OBJ_CPP)

.PHONY: all clean mock

all: $(TARGET)

mock: $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@ $(LDLIBS)

clean:
	@echo "Cleaning..."
	rm -f $(TARGET)
	find . -type f \( -name "*.o" -o -name "*.d" \) -delete
