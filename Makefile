TARGET    := lambda
CXX       := hipcc

LDLIBS    := -lpthread -ldl -lrt -lcrypto

# Rename cpp files in the Makefile if we want, or keep them as .cpp but compiled with hipcc.
SRC_CPP   := modinv.cpp lambda.cpp secp256k1.cpp lambda_kernel.cpp
ifneq (,$(findstring mock,$(MAKECMDGOALS)))
SRC_CPP += hip_mock.cpp
endif
OBJ_CPP   := $(SRC_CPP:.cpp=.o)
OBJ       := $(OBJ_CPP)

.PHONY: all clean mock

all: $(TARGET)

mock: CXX = g++
mock: CXXFLAGS += -D__device__= -D__host__= -D__global__= -I./hip
mock: $(TARGET)

# HIPCC will automatically detect the local architecture when no --offload-arch is specified.
# However, this assumes you're compiling on the machine you want to run.
# For optimal performance, we use -O3, -ffast-math, and native offload arch.
CXXFLAGS  := -O3 -ffast-math -std=c++14 -I. -MD

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@ $(LDLIBS)

clean:
	@echo "Cleaning..."
	rm -f $(TARGET)
	find . -type f \( -name "*.o" -o -name "*.d" \) -delete
