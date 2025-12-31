# tu paper goc

import numpy as np
def _next_power_of_two(x: int) -> int:
    x = int(x)
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()

class _SegmentTree:
    def __init__(self, capacity: int, operation, neutral_element: float):
        self._capacity = _next_power_of_two(capacity)
        self._value = np.full(2 * self._capacity, neutral_element, dtype=np.float32)
        self._op = operation
        self.neutral = np.float32(neutral_element)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __setitem__(self, idx: int, val: float) -> None:
        i = int(idx) + self._capacity
        self._value[i] = np.float32(val)
        i //= 2
        while i >= 1:
            self._value[i] = self._op(self._value[2 * i], self._value[2 * i + 1])
            i //= 2

    def __getitem__(self, idx: int) -> float:
        return float(self._value[int(idx) + self._capacity])

    def reduce(self, start: int = 0, end: int | None = None) -> float:
        if end is None:
            end = self._capacity
        start = int(start)
        end = int(end)
        if start >= end:
            return float(self.neutral)

        res_left = self.neutral
        res_right = self.neutral
        start += self._capacity
        end += self._capacity

        while start < end:
            if start & 1:
                res_left = self._op(res_left, self._value[start])
                start += 1
            if end & 1:
                end -= 1
                res_right = self._op(self._value[end], res_right)
            start //= 2
            end //= 2

        return float(self._op(res_left, res_right))

class _SumSegmentTree(_SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity, operation=lambda a, b: a + b, neutral_element=0.0)

    def sum(self, start: int = 0, end: int | None = None) -> float:
        return self.reduce(start, end)

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        ps = float(prefixsum)
        if ps < 0:
            ps = 0.0
        idx = 1
        while idx < self._capacity:
            left = 2 * idx
            if self._value[left] >= ps:
                idx = left
            else:
                ps -= float(self._value[left])
                idx = left + 1
        return idx - self._capacity

class _MinSegmentTree(_SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity, operation=lambda a, b: a if a < b else b, neutral_element=np.inf)

    def min(self, start: int = 0, end: int | None = None) -> float:
        return self.reduce(start, end)