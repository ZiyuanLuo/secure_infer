"""Load host KV into paged NPU memory, preserving per-layer dependencies."""
import torch


class LayerwiseGenerator:
    def __init__(self, ops):
        self.ops = ops

    def batched_to_gpu(self, starts, ends, **kwargs):
        """Yield once for each layer payload, then finish the load stream.

        The runtime transfers directly from host KV blocks to their allocated
        device slots; no intermediate GPU allocator or format conversion is used.
        """
        slot_mapping = kwargs['slot_mapping']
        sync = kwargs['sync']
        current_stream = torch.npu.current_stream()
        for layer in range(self.num_layers):
            memory_objects = yield
            if sync:
                current_stream.wait_stream(self.load_stream)
            with torch.npu.stream(self.load_stream):
                for start, end, memory in zip(starts, ends, memory_objects):
                    self.ops.single_layer_kv_transfer(
                        memory.tensor, self.kvcaches[layer],
                        slot_mapping[start:end], False, self.kv_format.value,
                        True, self.vllm_two_major)
        yield
        if sync:
            current_stream.wait_stream(self.load_stream)
        yield
