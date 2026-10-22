"""Carry connector completion metadata through vLLM 0.14 telemetry RPCs."""
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats


class RuntimeControlStats(KVConnectorStats):
    def aggregate(self, other):
        left = self.data.get('metadata')
        right = other.data.get('metadata')
        if left is None:
            self.data['metadata'] = right
        elif right is not None:
            left.aggregate(right)
        return self

    def reset(self):
        self.data.clear()

    def reduce(self):
        return {}

    def is_empty(self):
        return not self.data.get('metadata')
