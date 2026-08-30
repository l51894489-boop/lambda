TARGET    := lambda
# Auto-detect architecture compiler
ifneq (,$(shell which nvcc 2>/dev/null))
  CXX       := nvcc
  CXXFLAGS  := -O3 -std=c++14 -I. -MD -x cu -Xcompiler -O3,-ffast-math
else ifneq (,$(shell which hipcc 2>/dev/null))
  CXX       := hipcc
  CXXFLAGS  := -O3 -ffast-math -std=c++14 -I. -MD
else
  CXX       := g++
  CXXFLAGS  := -O3 -ffast-math -std=c++14 -I. -MD
endif

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
LDLIBS    := -lpthread -ldl -lrt -lcrypto

SRC_CPP   := modinv.cpp lambda.cpp secp256k1.cpp lambda_kernel.cpp
ifneq (,$(findstring mock,$(MAKECMDGOALS)))
SRC_CPP += hip_mock.cpp
endif
OBJ_CPP   := $(SRC_CPP:.cpp=.o)
OBJ       := $(OBJ_CPP)

.PHONY: all clean mock

all: $(TARGET)

mock: CXX = g++
mock: CXXFLAGS = -O3 -ffast-math -std=c++14 -I. -MD -D__device__= -D__host__= -D__global__= -I./hip
mock: $(TARGET)
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@ $(LDLIBS)

clean:
	@echo "Cleaning..."
	rm -f $(TARGET)
	find . -type f \( -name "*.o" -o -name "*.d" \) -delete
