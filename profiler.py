import types
import functools
import importlib
import datetime
from typing import List
from dataclasses import dataclass, field
import operator
from params import SchemeParams

import perf_counter

DECORATION_LIST = [
    "bootstrap",
    "eval_sine",
    "poly_eval",
    "fft",
    "evaluator",
    "logistic_regression",
    "micro_benchmarks",
    "poly",
]


@dataclass
class Frame:
    name: str
    stats: perf_counter.PerfCounter = None
    children: List = field(default_factory=list)

    def prune(self, attr=""):
        out_dict = {
            "name": self.name,
            "children": [child.prune(attr) for child in self.children],
        }
        if attr:
            out_dict["value"] = operator.attrgetter(attr)(self.stats)
        return out_dict


class Profiler:
    def __init__(self, name, archive=False):
        if archive:
            date_str = datetime.datetime.now().strftime("%Y%M%d_%H%m")
            self.name = ("" if name else f"{name}_") + date_str
        else:
            self.name = name
        self.stack = []
        self.data = None

        self.max_cache_size = 0
        self.current_cache_size = 0

        for module_name in DECORATION_LIST:
            module = importlib.import_module(module_name)
            self.decorate(module)

    def profile(self, target, *args, **kwargs):
        module_name, function_name = target.split(".", 1)

        module = importlib.import_module(module_name)
        target_function = getattr(module, function_name)
        target_function(*args, **kwargs)

    def decorate(self, module):
        def profiling_decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                next_frame = Frame(name=f.__name__)
                if self.stack:
                    self.stack[-1].children.append(next_frame)
                self.stack.append(next_frame)

                # for arg in args:
                #     if isinstance(arg, SchemeParams):
                #         cand_size = arg.get_max_cache_size()
                #         if self.max_cache_size != 0:
                #             print(cand_size, self.max_cache_size)
                #             assert(cand_size == self.max_cache_size)
                #         else:
                #             self.max_cache_size = cand_size

                # print("running frame", next_frame.name)
                # print("depth = ", len(self.stack))
                # print("max cache size", self.max_cache_size)
                stats = f(*args, **kwargs)
                # print("max cache size", self.max_cache_size)
                # print("depth = ", len(self.stack))
                # print("finished frame", next_frame.name)

                # if stats is not None:
                #     print("operation list", stats.arch.op_list)
                #     for op in stats.arch.op_list:
                #         if op[0] == perf_counter.OpType.READ:
                #             self.current_cache_size += op[1]
                #             assert(self.current_cache_size < self.max_cache_size)
                #         elif op[0] == perf_counter.OpType.WRITE:
                #             self.current_cache_size -= op[1]
                #         elif op[0] == perf_counter.OpType.AUTO_READ:
                #             pass
                #             # self.current_cache_size += op[1]
                #         else:
                #             raise ValueError("unknown operation type")
                #     stats.arch.op_list = []
                
                # print("current cache size", self.current_cache_size)

                if self.stack:
                    self.stack[-1].stats = stats
                    self.data = self.stack.pop()
                    assert(self.data is not None)
                else:
                    raise ValueError("need stack by now")

                return stats

            return wrapper

        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, types.FunctionType):
                # print("adding profiling to function", obj)
                setattr(module, name, profiling_decorator(obj))
