TARGET    := lambda
CXX       := g++

LDLIBS = -lpthread -ldl -lrt -lcrypto -lOpenCL

SRC_CPP   := src/modinv.cpp src/lambda.cpp src/secp256k1.cpp
OBJ_CPP   := $(SRC_CPP:.cpp=.o)
OBJ       := $(OBJ_CPP)

.PHONY: all clean

all: $(TARGET)

CXXFLAGS  := -g -O3 -std=c++14 -pthread -I. -MD

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@ $(LDLIBS)

test_cl: test_cl.o secp256k1.o modinv.o
	$(CXX) $(CXXFLAGS) test_cl.o secp256k1.o modinv.o -o $@ $(LDLIBS)

clean:
	@echo "Cleaning..."
	rm -f $(TARGET)
	find . -type f \( -name "*.o" -o -name "*.d" \) -delete