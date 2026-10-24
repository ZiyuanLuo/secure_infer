#!/usr/bin/env bash
set -euo pipefail
CODEC_TASK_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CODEC_PROJECT_DIR=$(cd -- "$CODEC_TASK_DIR/../.." && pwd)
CODEC_CANN_ROOT="${ASCEND_HOME_PATH:-${CANN_ROOT:-}}"
: "${CODEC_CANN_ROOT:?Load the CANN environment first}"
cd "$CODEC_TASK_DIR"
cmake -G Ninja -S kernel -B codec_build \
  -DASC_DIR="$CODEC_CANN_ROOT/lib64/cmake" -DAICPU_DIR="$CODEC_CANN_ROOT/lib64/cmake"
cmake --build codec_build -j4
cmake -G Ninja -S "$CODEC_PROJECT_DIR/secure_infer/native/kernels/aicpu" -B crypto_build \
  -DASC_DIR="$CODEC_CANN_ROOT/lib64/cmake" -DAICPU_DIR="$CODEC_CANN_ROOT/lib64/cmake"
cmake --build crypto_build --target runtime_aicpu_ghash -j4
g++ -O3 -std=c++17 -march=armv8-a+simd -fopenmp -shared -fPIC \
  encoder.cpp -l:liblz4.so.1 -o libhost_encoder.so
g++ -std=c++17 -shared -fPIC -O2 driver.cpp   -I"$CODEC_CANN_ROOT/include" -L"$CODEC_CANN_ROOT/lib64"   -Lcrypto_build -Lcodec_build -Wl,-rpath,'$ORIGIN'   -Wl,-rpath,'$ORIGIN/../crypto_build' -lopapi -lnnopbase -lascendcl   -lruntime_aicpu_ghash -lcachegen_entropy_restore -o codec_build/libentropy_driver.so
python3 test_encoder.py
python3 test_encoder_precision.py
exec python3 model_bench.py "$@"
