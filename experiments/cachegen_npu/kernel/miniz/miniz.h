// Standalone inflate-only adapter for miniz 3.0.2, no allocator or archive APIs.
#pragma once
#define MINIZ_NO_MALLOC
#define MINIZ_NO_STDIO
#define MINIZ_NO_TIME
#define MINIZ_USE_UNALIGNED_LOADS_AND_STORES 0
#define MINIZ_LITTLE_ENDIAN 1
#include "miniz_tinfl.h"
