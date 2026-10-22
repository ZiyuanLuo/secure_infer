CC ?= g++
CXX ?= g++

# Set ASCEND_ROOT or CANN_BASE in the environment.  No installation path is
# embedded in the repository.
CANN_BASE ?= $(ASCEND_ROOT)
ifeq ($(strip $(CANN_BASE)),)
$(error Set ASCEND_ROOT or CANN_BASE before building)
endif

ACL_INC ?= $(CANN_BASE)/aarch64-linux/include
ACL_LIB_PATH ?= $(CANN_BASE)/aarch64-linux/lib64
CANN_LIB_PATH ?= $(CANN_BASE)/lib64

OPENSSL_INC := $(shell pkg-config --cflags openssl 2>/dev/null)
OPENSSL_LIB := $(shell pkg-config --libs openssl 2>/dev/null)
ifeq ($(strip $(OPENSSL_INC)),)
OPENSSL_INC :=
endif
ifeq ($(strip $(OPENSSL_LIB)),)
OPENSSL_LIB := -lssl -lcrypto
endif

CFLAGS := -std=c++17 -fPIC -O2 -Wall -MMD -MP
LDFLAGS := -shared -fPIC -lpthread -ldl
NATIVE_DIR := secure_infer/native
LIB_SOURCES := $(NATIVE_DIR)/secure_infer.cpp $(NATIVE_DIR)/worker.cpp $(NATIVE_DIR)/openssl.cpp
LIB_OBJECTS := $(LIB_SOURCES:.cpp=.o)
DEPFILES := $(LIB_OBJECTS:.o=.d)
OUTPUT_LIB ?= libsecure_infer.so

.PHONY: all lib ghash-aicpu clean run

all: lib

lib: $(OUTPUT_LIB)

$(OUTPUT_LIB): $(LIB_OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(OPENSSL_LIB) -L$(ACL_LIB_PATH) -L$(CANN_LIB_PATH) -lascendcl -lacl_rt

%.o: %.cpp
	$(CXX) $(CFLAGS) -I$(ACL_INC) $(OPENSSL_INC) -c $< -o $@

ghash-aicpu:
	cmake -S $(NATIVE_DIR)/kernels/aicpu -B $(NATIVE_DIR)/kernels/aicpu/build \
		-DASCEND_ROOT=$(CANN_BASE) -DAICPU_ROOT=$(CANN_BASE)
	cmake --build $(NATIVE_DIR)/kernels/aicpu/build --target runtime_aicpu_ghash -j

run:
	@echo "Run: ./run.sh"

clean:
	rm -f $(LIB_OBJECTS) $(DEPFILES) $(OUTPUT_LIB)

-include $(DEPFILES)
