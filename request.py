"""构造共享前缀和不同后缀，并测量一次主机 KV 重载。"""
import argparse
import json
import math
import os
import time
import urllib.request
import urllib.parse
import uuid
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=os.getenv("SERVER_URL", "http://127.0.0.1:8080"))
    p.add_argument("--model", default=os.getenv("MODEL_NAME", "secure-infer"))
    p.add_argument("--prefix-tokens", type=int, default=int(os.getenv("PREFIX_TOKENS", "512")))
    p.add_argument("--suffix-tokens", type=int, default=int(os.getenv("SUFFIX_TOKENS", "512")))
    p.add_argument("--output-tokens", type=int, default=16)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument(
        "--store-wait", type=float,
        default=float(os.getenv("STORE_WAIT_SECONDS", "8")),
        help="等待异步 D2H 写回和主机缓存发布的时间（秒）",
    )
    p.add_argument(
        "--no-device-reset", action="store_true",
        help="不清空设备侧 prefix cache；仅用于设备缓存命中测试",
    )
    p.add_argument("--output")
    a = p.parse_args()
    if min(a.prefix_tokens, a.suffix_tokens) < 0 or a.prefix_tokens + a.suffix_tokens == 0:
        p.error("前后缀长度必须非负，总长度必须大于零")
    if min(a.output_tokens, a.repeat) < 1:
        p.error("输出长度和重复次数必须大于零")

    if not math.isfinite(a.store_wait) or a.store_wait < 0:
        p.error("写回等待时间必须是非负有限数")

    # 本机服务不经过宿主机或容器继承的 HTTP 代理。
    local = urllib.parse.urlsplit(a.url).hostname in {"localhost", "127.0.0.1", "::1"}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if local else urllib.request.build_opener()

    def post(path, payload):
        req = urllib.request.Request(a.url.rstrip("/") + path,
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        return opener.open(req, timeout=300)

    def tokens(text):
        with post("/tokenize", {"model": a.model, "prompt": text, "add_special_tokens": False}) as r:
            obj = json.load(r)
        ids = obj.get("tokens", obj.get("token_ids"))
        if not ids:
            raise RuntimeError("tokenizer 未返回 token IDs")
        return ids

    def fit(pattern, length):
        return (pattern * ((length + len(pattern) - 1) // len(pattern)))[:length]

    prefix = fit(tokens(" Shared reference context."), a.prefix_tokens)
    # The warmup request creates the host-resident prefix entry.  The suffix
    # is intentionally absent here, matching the reload experiment: the
    # target request supplies a new uncached suffix after the device cache is
    # cleared.
    payload = {"model": a.model, "prompt": prefix,
               "max_tokens": 1, "temperature": 0, "ignore_eos": True}
    if prefix:
        with post("/v1/completions", payload) as r:
            warmed = json.load(r)
        if "error" in warmed:
            raise RuntimeError(warmed["error"])
        if warmed.get("usage", {}).get("prompt_tokens") != a.prefix_tokens:
            raise RuntimeError("预热请求的实际前缀长度不匹配")
    if prefix and not a.no_device_reset:
        # Keep the external/host cache and clear only the device-side prefix
        # cache.  This is the operation that makes the following request use
        # the secure H2D reload path instead of a GPU prefix hit.
        time.sleep(a.store_wait)

    def reset_device_prefix_cache():
        # Advance scheduler steps so the previous request releases its blocks.
        for _ in range(2):
            probe = fit(tokens(uuid.uuid4().hex + " Cache release."), 1)
            with post("/v1/completions", {"model": a.model, "prompt": probe,
                      "max_tokens": 1, "temperature": 0, "ignore_eos": True}) as r:
                result = json.load(r)
            if "error" in result:
                raise RuntimeError(result["error"])
            time.sleep(1)
        with post("/reset_prefix_cache?reset_external=false", {}) as response:
            body = response.read().decode("utf-8", errors="replace")
        print(json.dumps({"device_prefix_cache_reset": body}, ensure_ascii=False), flush=True)

    rows = []
    for index in range(a.repeat):
        # Give each target a different suffix before starting the timer.
        target_pattern = tokens(uuid.uuid4().hex + " New question.")
        if not a.no_device_reset:
            # Reset before every target so repeated measurements do not reuse
            # the target's device KV blocks from the previous iteration.
            if index:
                time.sleep(a.store_wait)
            reset_device_prefix_cache()
        payload.update(prompt=prefix + fit(target_pattern, a.suffix_tokens),
                       max_tokens=a.output_tokens, stream=True,
                       stream_options={"include_usage": True})
        began = time.perf_counter()
        first = last = usage = None
        pieces = []
        done = False
        with post("/v1/completions", payload) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                obj = json.loads(data)
                if "error" in obj:
                    raise RuntimeError(obj["error"])
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices", []):
                    piece = choice.get("text", "")
                    if piece:
                        now = time.perf_counter()
                        first = now if first is None else first
                        last = now
                        pieces.append(piece)
        elapsed = time.perf_counter() - began
        if not done or first is None or not usage:
            raise RuntimeError("流式响应不完整，不计入测量")
        if usage["prompt_tokens"] != a.prefix_tokens + a.suffix_tokens:
            raise RuntimeError("实际输入长度与指定长度不一致")
        n = usage["completion_tokens"]
        row = dict(iteration=index + 1, prefix_tokens=a.prefix_tokens,
                   suffix_tokens=a.suffix_tokens, prompt_tokens=usage["prompt_tokens"],
                   output_tokens=n, ttft_ms=(first - began) * 1000,
                   tpot_ms=(last - first) * 1000 / (n - 1) if n > 1 else None,
                   e2e_ms=elapsed * 1000, stream_chunks=len(pieces), text="".join(pieces))
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if a.output:
        path = Path(a.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
