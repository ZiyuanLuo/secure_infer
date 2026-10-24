#pragma once
#include <stdint.h>
struct EntropyArgs {
 uint64_t input, output, scratch, status;
 uint32_t tokens, channels, compressed_bytes, reserved;
};
