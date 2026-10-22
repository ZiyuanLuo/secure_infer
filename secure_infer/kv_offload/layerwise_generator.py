"""Layerwise generator used by the measured vLLM-Ascend/LMCache path.

Methods extracted without semantic changes from the recorded
VLLMPagedMemLayerwiseNPUConnector in original_npu_connectors.py.
Torch CUDA stream calls are mapped to the Ascend NPU stream API.
"""
import logging
from typing import List, Optional
from types import SimpleNamespace
import torch as _torch

torch = SimpleNamespace(cuda=_torch.npu, cat=_torch.cat, Tensor=_torch.Tensor)
MemoryObj = object
MemoryFormat = SimpleNamespace(KV_T2D=0)
logger = logging.getLogger(__name__)

class LayerwiseGenerator:
    def __init__(self, ops):
        self.ops = ops

    def _lazy_initialize_buffer(self, kv_caches):
        """
            Lazily initialize the GPU buffer allocator if it is not initialized yet.
            Currently, we use the `kv_caches` (kv cache pointer) to determine
            the gpu buffer size in gpu connector.
            Also, the first request might be a bit slower due to buffer creation.

            Supports both legacy formats and new SEPARATE_KV format:
            - Legacy MERGED_KV: [2, num_blocks, block_size, num_heads, head_size]
            - New SEPARATE_KV: tuple(key_tensor, value_tensor) where each is
              [num_blocks, block_size, num_heads, head_size]
            """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info('Lazily initializing GPU buffer.')
            self.kv_format = KVCacheFormat.detect(kv_caches, use_mla=self.use_mla)
            if self.kv_format == KVCacheFormat.UNDEFINED:
                raise ValueError('Undefined KV cache format detected. Unable to determine the format of input kv_caches.')
            logger.info(f'Detected KV cache format: {self.kv_format.name}')
            first_layer_cache = kv_caches[0]
            if self.kv_format == KVCacheFormat.SEPARATE_KV:
                key_tensor = first_layer_cache[0]
                value_tensor = first_layer_cache[1]
                assert key_tensor.shape == value_tensor.shape, f'Key and Value tensors must have identical shapes, got key={key_tensor.shape}, value={value_tensor.shape}'
                k_cache_shape_per_layer = key_tensor.shape
                self.vllm_two_major = False
            elif self.kv_format == KVCacheFormat.MERGED_KV:
                assert first_layer_cache.shape[0] == 2 or first_layer_cache.shape[1] == 2, f'MERGED_KV format should have shape [num_layers, 2, num_blocks, block_size, num_heads, head_size] or [num_layers, num_blocks, 2, block_size, num_heads, head_size]Got shape: {first_layer_cache.shape}'
                self.vllm_two_major = first_layer_cache.shape[0] == 2
                if self.vllm_two_major:
                    k_cache_shape_per_layer = first_layer_cache[0].shape
                else:
                    k_cache_shape_per_layer = first_layer_cache[:, 0].shape
            else:
                raise ValueError(f'Unsupported KV cache format: {self.kv_format}')
            max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]
            logger.info(f'Lazily initializing GPU buffer:\n  - Format: {self.kv_format.name}\n  - Key cache shape per layer: {k_cache_shape_per_layer}\n  - Max tokens: {max_tokens}')
            num_elements_key = k_cache_shape_per_layer.numel()
            num_elements = num_elements_key * 2
            gpu_buffer_size = num_elements * self.element_size
            self.gpu_buffer_allocator = GPUMemoryAllocator(gpu_buffer_size, device=self.device)

    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
            This function is a generator that moves the KV cache from the memory
            objects to paged GPU memory. The first iteration will prepare some
            related metadata. In each of the following iterations, it will first
            wait until the loading of the previous layer finish, and then load
            one layer of KV cache from the memory objects -> GPU buffer ->
            paged GPU memory. The last iteration simply waits for the last layer
            to finish.
            In total, this the generator will yield num_layers + 2 times.

            :param starts: The starting indices of the KV cache in the corresponding
                token sequence.

            :param ends: The ending indices of the KV cache in the corresponding
                token sequence.

            :raises ValueError: If 'slot_mapping' is not provided in kwargs.
            """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, 'kvcaches should be provided in kwargs or initialized beforehand.'
        if 'slot_mapping' not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if 'sync' not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")
        slot_mapping: torch.Tensor = kwargs['slot_mapping']
        sync: bool = kwargs['sync']
        self._lazy_initialize_buffer(self.kvcaches)
        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)
        num_tokens = len(slot_mapping_full)
        chunk_offsets = []
        chunk_sizes = []
        current_offset = 0
        for start, end in zip(starts, ends, strict=False):
            chunk_size = end - start
            chunk_sizes.append(chunk_size)
            chunk_offsets.append(current_offset)
            current_offset += chunk_size
        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(buffer_shape, self.dtype, MemoryFormat.KV_T2D)
            assert tmp_gpu_buffer_obj is not None, 'Failed to allocate NPU buffer in NPUConnector'
            assert tmp_gpu_buffer_obj.tensor is not None
        current_stream = torch.cuda.current_stream()
        for layer_id in range(self.num_layers):
            memory_objs_layer = (yield)
            if sync:
                current_stream.wait_stream(self.load_stream)
            if layer_id > 0:
                logger.debug(f'Finished loading layer {layer_id - 1}')
            with torch.cuda.stream(self.load_stream):
                if self.use_gpu:
                    cpu_tensors = []
                    for memory_obj in memory_objs_layer:
                        assert memory_obj.tensor is not None
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                        cpu_tensors.append(memory_obj.tensor)
                    self.ops.batched_fused_single_layer_kv_transfer(cpu_tensors, tmp_gpu_buffer_obj.tensor, self.kvcaches[layer_id], slot_mapping_full, chunk_offsets, chunk_sizes, False, self.kv_format.value, True, self.vllm_two_major)
                else:
                    for start, end, memory_obj in zip(starts, ends, memory_objs_layer, strict=False):
                        assert memory_obj.tensor is not None
                        self.ops.single_layer_kv_transfer(memory_obj.tensor, self.kvcaches[layer_id], slot_mapping[start:end], False, self.kv_format.value, True, self.vllm_two_major)
                logger.debug(f'Finished loading layer {layer_id}')
        yield
        if sync:
            current_stream.wait_stream(self.load_stream)
        if self.use_gpu and tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()
        yield
