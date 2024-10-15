from typing import List


# TotalMeter 类是用于跟踪数值的总和和数量的基本工具，它提供了两种更新方式：
# 一种是简单的单个值更新，另一种是带有权重的更新，即可以同时更新多个相同值。
class TotalMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    # update 方法用于增加一个单一值。
    def update(self, val: float):

        # 每次update，累加value值，并且统计累计的次数
        self.sum += val
        self.count += 1

    # 带有权重的update
    def update_with_weight(self, val: float, count: int):
        self.sum += val*count
        self.count += count

    # reset 方法重置所有统计数据。
    def reset(self):
        self.sum = 0
        self.count = 0

    # avg 属性返回总的平均值，如果没有任何数据，则返回-1作为无效值标识。
    @property
    def avg(self):
        if self.count == 0:
            return -1
        return self.sum / self.count


# AverageMeter 类是一个固定长度的滑动窗口平均值计算器。
# 它会维护一个固定大小的历史记录列表，
# 每当有新的值加入时，它会根据队列的先进先出（FIFO）原则移除最旧的数据点，并添加新的数据点。
# 这样可以得到一个最近一段时间内的平均值。
class AverageMeter:
    def __init__(self, length: int, name: str = None):
        assert length > 0
        self.name = name
        self.count = 0
        self.sum = 0.0
        self.current: int = -1
        self.history: List[float] = [None] * length

    # val 属性返回最新一次更新的值。
    @property
    def val(self) -> float:
        return self.history[self.current]

    # avg 属性返回所有历史记录的平均值。
    @property
    def avg(self) -> float:
        return self.sum / self.count

    # update 方法更新当前值，并自动维护历史记录。
    def update(self, val: float):
        self.current = (self.current + 1) % len(self.history)
        self.sum += val

        old = self.history[self.current]
        if old is None:
            self.count += 1
        else:
            self.sum -= old
        self.history[self.current] = val


# WeightedMeter 类主要用于追踪一个值的总和、平均值以及最新的值（val），并且可以处理带权重的更新。
# 这意味着每次更新时，你可以指定一个数值及其出现的次数（权重）。这使得它非常适合用于需要按权重来计算平均值的情况。
class WeightedMeter:
    def __init__(self, name: str = None):
        self.name = name
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0
        self.val = 0.0

    # update 方法允许你添加一个新值，并指定该值的权重，默认权重为1。
    def update(self, val: float, num: int = 1):
        self.count += num
        self.sum += val * num
        self.avg = self.sum / self.count
        self.val = val

    # reset 方法允许你重置计数器和总和，也可以传入特定的总和和计数以进行重置
    def reset(self, total: float = 0, count: int = 0):
        self.count = count
        self.sum = total
        self.avg = total / max(count, 1)
        self.val = total / max(count, 1)
