from os import stat
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, log, log2, sqrt
from dataclasses import dataclass, field


@dataclass
class Stats:
    logN: int = 17
    logq: int = 50
    add: int = 0
    mult: int = 0
    ntt: int = 0
    limb_rd: int = 0
    limb_wr: int = 0
    auto_rd: int = 0
    plain_rd: int = 0
    cache_size: int = 0

    def __add__(self, other):
        assert self.logN == other.logN
        assert self.logq == other.logq
        sum_val = Stats(
            logN=self.logN,
            add=self.add + other.add,
            mult=self.mult + other.mult,
            ntt=self.ntt + other.ntt,
            limb_rd=self.limb_rd + other.limb_rd,
            limb_wr=self.limb_wr + other.limb_wr,
            auto_rd=self.auto_rd + other.auto_rd,
            plain_rd=self.plain_rd + other.plain_rd,
            cache_size=max(self.cache_size, other.cache_size),
        )
        return sum_val

    @property
    def ntt_mult(self):
        ntts = self.ntt
        ntts *= 1 << self.logN
        return (ntts // 2) * self.logN + (ntts // 2) + ntts

    @property
    def ntt_add(self):
        ntts = self.ntt
        ntts *= 1 << self.logN
        return ntts * self.logN

    @property
    def total_gops(self):
        return (self.add + self.mult + self.ntt_add + self.ntt_mult) / 1e9

    def __str__(self):
        out_str = ""
        out_str += "adds: %s\n" % self.add
        out_str += "mults: %s\n" % self.mult
        out_str += "ntt_adds: %s\n" % self.ntt_add
        out_str += "ntt_mults: %s\n" % self.ntt_mult
        out_str += "total ops: %s\n" % self.total_gops
        out_str += "limb rd: %s\n" % ((self.limb_rd * self.logq) / (8 * 1e9))
        out_str += "limb wr: %s\n" % ((self.limb_wr * self.logq) / (8 * 1e9))
        out_str += "auto rd: %s\n" % ((self.auto_rd * self.logq) / (8 * 1e9))
        out_str += "plain rd: %s\n" % ((self.plain_rd * self.logq) / (8 * 1e9))
        out_str += "cache size: %s\n" % ((self.cache_size * self.logq) / (8 * 1e6))
        return out_str


def rotate(logN: int, L: int, dnum: int, chip_mem_size: str, k: int):
    alpha = int(ceil(L / dnum))
    N = 1 << logN

    stats = Stats(logN=logN)

    if chip_mem_size == "small":
        stats.cache_size = N
    elif chip_mem_size == "medium":
        stats.cache_size = N * alpha
    elif chip_mem_size == "large":
        stats.cache_size = N * k

    return stats


def mod_raise(logN: int, L: int, dnum: int, chip_mem_size: str):
    alpha = int(ceil(L / dnum))
    beta = dnum
    N = 1 << logN

    stats = Stats(logN=logN)

    stats.add += N * beta * alpha * (L + 1)
    stats.mult += (2 * N * beta * alpha) + (N * beta * alpha * (L + 2))
    stats.ntt += L + (beta * L)

    if chip_mem_size == "medium":
        stats.limb_rd += beta * N * (L / alpha)
        stats.limb_wr += beta * N * (L / alpha)
        stats.cache_size = 2 * N * alpha
    else:
        stats.limb_rd += (N * L) + (N * L) + (beta * N * L)
        stats.limb_wr += (N * L) + (beta * N * L) + (beta * N * L)
        stats.cache_size = 2 * N

    return stats


def inner_product(logN: int, L: int, dnum: int, chip_mem_size: str, r_idx: int, k: int):
    alpha = int(ceil(L / dnum))
    beta = dnum
    N = 1 << logN

    stats = Stats(logN=logN)

    stats.add += 2 * beta * (L + alpha) * N
    stats.mult += 2 * beta * (L + alpha) * N

    stats.auto_rd += beta * N * (L + alpha)

    if chip_mem_size == "small" or (chip_mem_size == "large" and r_idx == 0):
        stats.limb_rd += beta * N * (L + alpha)
    elif chip_mem_size == "medium":
        stats.limb_rd += (N * ((L + alpha) / alpha)) + (beta * N * L)
        stats.cache_size = 4 * N * alpha

    if chip_mem_size == "small":
        stats.cache_size = 4 * N
    elif chip_mem_size == "large":
        stats.cache_size = 4 * N * k

    return stats


def mod_reduce(logN: int, L: int, dnum: int, chip_mem_size: str):
    alpha = int(ceil(L / dnum))
    N = 1 << logN

    stats = Stats(logN=logN)

    stats.add += N * alpha * L
    stats.mult += N * alpha * (L + 1)
    stats.ntt += L + alpha

    if chip_mem_size == "small" or "large":
        stats.limb_rd += (alpha * N) + (N * L)
        stats.limb_wr += (alpha * N) + (N * L) + (N * L)
        stats.cache_size = 2 * N
    elif chip_mem_size == "medium":
        stats.limb_rd += N * L
        stats.limb_wr += (N * ((L + alpha) / alpha)) + (N * L)
        stats.cache_size = 2 * N * alpha

    return stats


def poly_add(logN: int, L: int, chip_mem_size: str, r_idx: int, k: int):
    N = 1 << logN

    stats = Stats(logN=logN)

    stats.add += N * L

    if (chip_mem_size == "small" and r_idx != -2) or (
        chip_mem_size == "large" and r_idx == 0 and r_idx != -2
    ):
        stats.limb_rd = N * L

    if chip_mem_size == "small":
        stats.cache_size = 2 * N
    elif chip_mem_size == "medium":
        stats.cache_size = 2 * N  # Fix me
    elif chip_mem_size == "large":
        stats.cache_size = 2 * N * k

    return stats


def multiply_plain(logN: int, L: int, chip_mem_size: str, k: int):
    N = 1 << logN

    stats = Stats(logN=logN)

    stats.mult += 2 * N * L

    if chip_mem_size == "small":
        stats.cache_size = 2 * N
    elif chip_mem_size == "medium":
        stats.cache_size = 2 * N  # fix me
    elif chip_mem_size == "large":
        stats.cache_size = 2 * N * k

    return stats


def fft_unrolled(logN: int, L: int, dnum: int, u: int, chip_mem_size: str):
    k = 2 * (pow(2, u) - 1)
    # print("K: %s" % k)

    stats = Stats(logN=logN)

    alpha = int(ceil(L / dnum))

    stats += mod_raise(logN, L, dnum, chip_mem_size)

    for r_idx in range(k):
        stats += rotate(logN, L, dnum, chip_mem_size, k)
        stats += inner_product(logN, L, dnum, chip_mem_size, r_idx, k)
        stats += poly_add(logN, L, chip_mem_size, r_idx, k)

    for _ in range(k + 1):
        stats += multiply_plain(logN, L + alpha, chip_mem_size, k)

    for _ in range(k):
        # do not perform a limb read with r_idx = -2
        stats += poly_add(logN, L + alpha, chip_mem_size, r_idx=-2, k=k)
        stats += poly_add(logN, L + alpha, chip_mem_size, r_idx=-2, k=k)

    for _ in range(2):
        stats += mod_reduce(logN, L, dnum, chip_mem_size)

    return stats


def fft(logN: int, dnum: int, u: int, chip_mem_size: str):
    stats = Stats()

    niter = (logN - 1) // u
    u_fixed = [u] * niter
    correction = logN - 1 - u * niter
    for idx in range(correction):
        u_fixed[idx] += 1

    # print("u_fixed: %s" % u_fixed)

    fresh_limbs = 19
    L = (2 * niter) + 11 + fresh_limbs
    # print("L: %s" % L)

    for uu in u_fixed:
        stats += fft_unrolled(logN, L, dnum, uu, chip_mem_size)
        # drop a limb
        L -= 1

    return stats


def stacked_plot_compute(stats):
    x = ["1", "2", "3", "4", "5"]
    y1 = np.array([v.add for v in stats])
    y2 = np.array([v.mult for v in stats])
    y3 = np.array([v.ntt_add for v in stats])
    y4 = np.array([v.ntt_mult for v in stats])

    plt.bar(x, y1, color="r")
    plt.bar(x, y2, bottom=y1, color="b")
    plt.bar(x, y3, bottom=y1 + y2, color="c")
    plt.bar(x, y4, bottom=y1 + y2 + y3, color="y")
    # plt.bar(x, y5, bottom=y1+y2+y3+y4, color='m')

    plt.xlabel("K")
    plt.ylabel("Operations")
    plt.legend(["Adds", "Mults", "NTT Adds", "NTT Mults"])
    plt.title("Total operations in FFT with varying unrolling")
    plt.show()


def stacked_plot_dram_transfers(stats):
    x = ["1", "2", "3", "4", "5"]
    y1 = np.array([v.limb_rd for v in stats])
    y2 = np.array([v.limb_wr for v in stats])
    y3 = np.array([v.auto_rd for v in stats])

    plt.bar(x, y1, color="r")
    plt.bar(x, y2, bottom=y1, color="b")
    plt.bar(x, y3, bottom=y1 + y2, color="c")

    plt.xlabel("K")
    plt.ylabel("DRAM transfers")
    plt.legend(["Limb rds", "Limb wrs", "Auto rds"])
    plt.title("Dram transfers in FFT with varying unrolling")
    plt.show()


def main():
    stats = []

    logN = 17
    dnum = 2

    # mem size small holds 2-3 limbs
    # mem size medium holds alpha limbs
    # mem size large holds k rotation limbs
    mem_size_options = ["small", "medium", "large"]
    chip_mem_size = mem_size_options[2]

    for u in range(1, 6):
        stats.append(fft(logN, dnum, u, chip_mem_size))

    for val in stats:
        print(val)

    # stacked_plot_compute(stats)
    # stacked_plot_dram_transfers(stats)

    return 0


if __name__ == "__main__":
    main()
