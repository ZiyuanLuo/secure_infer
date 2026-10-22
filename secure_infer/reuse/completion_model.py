"""Layerwise loading/compute completion estimate shared by both decisions."""
import numpy as np

def pipeline_finish(suffix_ms, loading_ms, layers):
    return np.maximum(loading_ms + suffix_ms/layers, suffix_ms + loading_ms/layers)
