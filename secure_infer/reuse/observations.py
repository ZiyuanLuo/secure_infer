"""Completed load observations; no offline profile or request-ID control."""
from secure_infer.kv_offload.native_layerwise import log

def add_load(sample):
    log('completed_loads.jsonl', sample)
