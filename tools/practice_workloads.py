"""Reproducible extensions to the sets of published CUDA workloads."""
from dataclasses import dataclass
from functools import partial

import torch


BASE_SEED = 2026
GUARD_BYTES = 256
GUARD_VALUE = 0xA5


@dataclass(frozen=True)
class Case:
    number: int
    label: str
    factory: object
    guarded: bool = False


def published_replays(original_count, load, first_number):
    positions = (0, original_count // 2, original_count - 1)
    descriptions = ("首个用例复验", "中间用例复验", "末尾用例复验")
    for offset, (position, description) in enumerate(zip(positions, descriptions)):
        number = first_number + offset
        yield Case(number, f"{description} · 原用例 #{position + 1} · 种子 {BASE_SEED + number} · 缓冲区守卫",
                   partial(load, position + 1, BASE_SEED + number), guarded=True)


def guarded_copy(value, guard_bytes=GUARD_BYTES):
    """Keep CUDA pointer alignment and place byte canaries outside dense tensors."""
    if not (isinstance(value, torch.Tensor) and value.is_cuda and value.numel()
            and value.is_contiguous()):
        return None
    size = value.numel() * value.element_size()
    storage = torch.full((guard_bytes * 2 + size,), GUARD_VALUE, device=value.device,
                         dtype=torch.uint8)
    start, stop = guard_bytes, guard_bytes + size
    result = storage[start:stop].view(value.dtype).reshape(value.shape)
    result.copy_(value)
    return result, storage[:start], storage[stop:]


def verify_guards(judge, number, guards):
    for label, (before, after) in guards.items():
        if not ((before == GUARD_VALUE).all().item() and (after == GUARD_VALUE).all().item()):
            judge.wrong(number, f"{label} 的 CUDA 缓冲区边界被写入")
