from os import stat
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, log, log2, sqrt, pi
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
    relin_rd: int = 0
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
            relin_rd=self.relin_rd + other.relin_rd,
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
        out_str += "relin rd: %s\n" % ((self.relin_rd * self.logq) / (8 * 1e9))
        out_str += "plain rd: %s\n" % ((self.plain_rd * self.logq) / (8 * 1e9))
        out_str += "cache size: %s\n" % ((self.cache_size * self.logq) / (8 * 1e6))
        return out_str


def mod_raise(logN: int, L: int, dnum: int):
    alpha = int(ceil(L / dnum))
    beta = dnum
    N = 1 << logN
    stats = Stats(logN=logN)

    stats.add += N * beta * alpha * (L + 1)
    stats.mult += (2 * N * beta * alpha) + (N * beta * alpha * (L + 2))
    stats.ntt += beta * alpha + beta * (L + 1)

    stats.limb_rd += (N * L) + (N * L) + (beta * N * L)
    stats.limb_wr += (N * L) + (beta * N * L) + (beta * N * L)

    return stats


def inner_product(logN: int, L: int, dnum: int):
    alpha = int(ceil(L / dnum))
    beta = dnum
    N = 1 << logN
    stats = Stats(logN=logN)

    stats.add += 2 * (beta - 1) * (L + alpha) * N
    stats.mult += 2 * beta * (L + alpha) * N

    stats.relin_rd += beta * N * (L + alpha)
    stats.limb_rd += beta * N * (L + alpha)

    return stats


def mod_reduce(logN: int, starting_limbs: int, ending_limbs: int):
    N = 1 << logN
    stats = Stats(logN=logN)

    extra_limbs = starting_limbs - ending_limbs

    ## basis conversion
    stats.ntt += extra_limbs
    stats.mult += N * extra_limbs * (ending_limbs + 1)
    stats.add += N * extra_limbs * ending_limbs
    stats.ntt += ending_limbs

    ## subtraction

    # stats.mult += N * ending_limbs
    # stats.add += N * ending_limbs

    stats.limb_rd += (starting_limbs * N) + (N * ending_limbs)
    stats.limb_wr += (starting_limbs * N) + (N * ending_limbs)

    return stats


def relinearize(logN: int, L: int, dnum: int):
    alpha = int(ceil(L / dnum))
    beta = dnum
    raised_limbs = L + 1 + alpha
    N = 1 << logN
    stats = Stats(logN=logN)

    stats += mod_raise(logN, L, dnum)
    stats += inner_product(logN, L, dnum)
    stats += mod_reduce(logN=logN, starting_limbs=raised_limbs, ending_limbs=L)
    stats += mod_reduce(logN=logN, starting_limbs=raised_limbs, ending_limbs=L)

    return stats


def ct_ct_mult(logN: int, L: int, dnum: int):
    N = 1 << logN
    stats = Stats(logN=logN)

    # operations as defined in multiply inner in evaluator
    # replaces two mults and one add with one mult and four adds
    stats.mult += 3 * N * L
    stats.add += 4 * N * L

    # relinearize
    stats += relinearize(logN, L, dnum)

    # add
    stats.add += 2 * N * L

    stats.limb_wr += 2 * N * L

    return stats


def ct_ct_add(logN: int, L: int):
    N = 1 << logN
    stats = Stats(logN=logN)

    stats.add += 2 * N * L

    stats.limb_rd += 2 * N * L
    stats.limb_wr += 2 * N * L

    return stats


def ct_pt_add(logN: int, L: int):
    N = 1 << logN
    stats = Stats(logN=logN)

    stats.add += N * L

    return stats


def ct_pt_mult(logN: int, L: int):
    N = 1 << logN
    stats = Stats(logN=logN)

    stats.mult += 2 * N * L

    return stats


def rescale(logN: int, L: int):
    N = 1 << logN
    stats = Stats(logN=logN)

    # inverse ntt on last limb
    stats.ntt += 2

    # ntt wrt to each modulus
    stats.ntt += 2 * (L - 1)

    # subtract last limb from each limb
    stats.add += 2 * N * (L - 1)

    # scalar mults with q^-1
    stats.mult += 2 * N * (L - 1)

    return stats


def eval_sine(logN: int, dnum: int, r: int, d: int, L: int):
    stats = Stats()

    print("eval sine start limbs", L)

    m = int(ceil(log2(d + 1)))
    ell = m // 2

    print("d, m, ell", d, m, ell)

    # baby steps
    max_baby_step_index = pow(2, ell)
    index_reached = 1
    current_level = 1
    while index_reached < max_baby_step_index:
        reachable_index = min(pow(2, current_level), max_baby_step_index)
        for i in range(index_reached, reachable_index):
            current_index = i + 1
            stats += ct_ct_mult(logN, L, dnum)
            stats += rescale(logN, L)
            L_new = L - 1
            stats += ct_ct_add(logN, L_new)
            if current_index % 2 == 0:  # for even indices subtract constant one
                stats += ct_pt_add(logN, L_new)
            else:
                stats += ct_ct_add(logN, L_new)
        index_reached = reachable_index
        current_level += 1
        L -= 1

    print("limbs after baby steps", L)

    # gaint steps
    index = pow(2, ell)
    gs_L = L
    while index < pow(2, m - 1):
        stats += ct_ct_mult(logN, L, dnum)
        stats += rescale(logN, L)
        # drop a limb
        gs_L = L - 1
        stats += ct_ct_add(logN, gs_L)
        stats += ct_ct_add(logN, gs_L)
        index *= 2

    print("limbs before eval recurse", L)

    # start eval_recurse
    ## depth of the tree is the number of giant steps
    num_giant_steps = m - ell  ## [ell, m-1] inclusive
    num_baby_steps = pow(2, ell)
    num_leaves = pow(2, num_giant_steps)
    ## start at the leaves
    print("num leaves, num baby steps", num_leaves, num_baby_steps)
    for _ in range(num_leaves):
        for i in range(num_baby_steps):
            # pass
            stats += ct_pt_mult(logN, L)
            if i == 0:
                stats += ct_pt_add(logN, L)
            else:
                stats += ct_ct_add(logN, L)
        stats += rescale(logN, L)
    L -= 1

    ## now walk up the tree
    ## number of mults + adds is the number of nodes at the next level
    for i in range(num_giant_steps - 1, -1, -1):
        num_nodes = pow(2, i)
        for _ in range(num_nodes):
            stats += ct_ct_mult(logN, L, dnum)
            L_new = L - 1
            stats += ct_ct_add(logN, L_new)
        L -= 1

    print("limbs after eval recurse", L)

    # scaling by r
    for i in range(r):
        stats += ct_ct_mult(logN, L, dnum)
        stats += rescale(logN, L)
        # drop a limb
        L -= 1
        stats += ct_ct_add(logN, L)
        stats += ct_pt_add(logN, L)

    print("limbs after r scaling", L)

    return stats


def main():
    logN = 17
    r = 2
    d = 63
    dnum = 2
    start_limbs = 34

    # mem size small holds 2-3 limbs
    # mem size medium holds alpha limbs
    # mem size large holds k rotation limbs
    mem_size_options = ["small", "medium", "large"]
    chip_mem_size = mem_size_options[2]

    stats = eval_sine(logN, dnum, r, d, start_limbs)
    print(stats)

    return 0


if __name__ == "__main__":
    main()
