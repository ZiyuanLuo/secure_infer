ARG BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.14.0rc1
FROM ${BASE_IMAGE}
SHELL ["bash", "-c"]
RUN if ! command -v g++ >/dev/null || ! command -v make >/dev/null || ! pkg-config --exists openssl; then \
      apt-get update && apt-get install -y --no-install-recommends g++ make pkg-config libssl-dev cmake && \
      rm -rf /var/lib/apt/lists/*; fi
