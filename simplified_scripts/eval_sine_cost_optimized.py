from os import stat
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, log, log2, sqrt, pi
from dataclasses import dataclass, field

from evaluator import mod_reduce_rescale, multiply_plain


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


def mod_raise(logN: int, starting_limbs: int, new_limbs: int, dnum: int):
    beta = dnum
    N = 1 << logN
    stats = Stats(logN=logN)

    stats.add += N * beta * starting_limbs * new_limbs
    stats.mult += (2 * N * beta * starting_limbs) + (
        N * beta * starting_limbs * (new_limbs + 1)
    )
    ## inverse NTT for the starting limbs
    ## forwad NTT for the new limbs
    stats.ntt += beta * starting_limbs + beta * new_limbs

    return stats


def inner_product(logN: int, L: int, dnum: int):
    alpha = int(ceil(L / dnum))
    beta = dnum
    N = 1 << logN
    stats = Stats(logN=logN)

    stats.add += 2 * (beta - 1) * (L + alpha) * N
    stats.mult += 2 * beta * (L + alpha) * N

    stats.relin_rd += beta * (L + alpha) * N

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

    stats.mult += N * ending_limbs
    stats.add += N * ending_limbs

    return stats


def ct_ct_mult(logN: int, L: int, dnum: int):
    alpha = int(ceil(L / dnum))
    beta = dnum
    N = 1 << logN
    stats = Stats(logN=logN)

    raised_limbs = L + 1 + alpha

    ## begin by reading in a1, a2, b1, b2
    ## one limb at a time. 4 limbs of space being used
    stats.limb_rd += 4 * L * N

    ## compute the products a1*a2, b1*b1, and (a1+a2)*(b1+b2)- a1*a2 - b1*b2
    stats.mult += 3 * N * L
    stats.add += 4 * N * L

    ## as we compute each limb of b1*b2 and a1b2 + a2b1,
    ## multiply by P mod Q to "mod raise"
    stats.mult += 2 * N * L

    ## after these limbs have been "mod raised", write them out
    stats.limb_wr += 2 * N * L

    ## at the same time, as we complete the usage of alpha limbs
    ## of a1*a2, write out all resulting limbs expect for these limbs.
    ## now there are only alpha limbs in memory. perform full modulus raising
    stats += mod_raise(
        logN=logN, starting_limbs=alpha, new_limbs=raised_limbs - alpha, dnum=dnum
    )  ## accounts for all mod raise

    # print("remove this")
    # return stats

    ## as we compute each fresh limb, read in the relin key limb to multiply
    stats.relin_rd += beta * raised_limbs * N
    stats.mult += 2 * beta * raised_limbs * N  ## relin key has two terms
    ## reading in accumulator to add result
    stats.limb_rd += 2 * beta * raised_limbs * N
    stats.add += 2 * (beta - 1) * raised_limbs * N
    ## write out accumulator
    ## only beta-1 writes because on the last block of alpha limbs
    ## the resulting accumulator limbs will be complete, so we can compute
    ## further on them.
    stats.limb_wr += 2 * (beta - 1) * raised_limbs * N

    # print("remove this")
    # return stats

    """
    in the case where we read in the accumulator, add and write it back out, we do
    three reads per limb for the accumulator + rlk, two writes per limb for the accumulator

    in the case where we write out all fresh limbs then read everything back in to accumulate
    two writes per limb, then two reads per limb, two writes per result limb

    in the case where we wait to read in the relin key to accumulate, we have 
    one write per limb, two read per limb (rlk + data), two writes per result limb
    """

    ## one pair of finished accumulator limbs is in memory (plus alpha limbs for mod raising)
    ## read in one limb of a1*b2+b1*a2 and b1*b2 to add
    stats.limb_rd += 2 * N * L  ## only L nontrivial limbs
    stats.add += 2 * N * L
    ## write out result. need to also write out the alpha limbs from mod raising
    ## since these limbs are freshly computed (a1*a2)
    stats.limb_wr += raised_limbs * N

    # print("remove this")
    # return stats

    ## memory is empty

    # start mod reduce
    stats.limb_rd += N * raised_limbs
    mod_red_stats = mod_reduce(
        logN=logN, starting_limbs=raised_limbs, ending_limbs=L - 1
    )
    ## two polynomials to reduce
    stats += mod_red_stats
    stats += mod_red_stats
    stats.limb_wr += 2 * (L - 1) * N

    # print("remove this")
    # return stats

    ## seems like we already did this....
    # # correction
    # stats.limb_rd += 2 * (L - 1) * N
    # stats.add += (L - 1) * N
    # stats.limb_wr += 2 * (L - 1) * N

    return stats


def ct_ct_add(logN: int, L: int):
    N = 1 << logN
    stats = Stats(logN=logN)

    stats.add += 2 * N * L

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
    """
    ciphertext rescale
    """
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
    N = 1 << logN

    print("d, m, ell", d, m, ell)

    # baby steps
    max_baby_step_index = pow(2, ell)  ## [0, 2^ell] inclusive
    ## 2^ell is not actually a baby-step index, but we compute it in the same level
    index_reached = 1
    current_level = 1

    ## read in t = T_1
    # stats.limb_rd += 2 * N * L

    while index_reached < max_baby_step_index:
        reachable_index = min(pow(2, current_level), max_baby_step_index)

        for _ in range(index_reached, reachable_index):
            # print("computing ind", current_index)
            stats += ct_ct_mult(logN, L, dnum)
            L_new = L - 1
            stats += ct_ct_add(logN, L_new)

        for _ in range(index_reached):
            stats += ct_pt_mult(logN, L)
            stats += rescale(logN, L)

        for i in range(index_reached, reachable_index):
            current_index = i + 1  ## index we're currently computing
            if current_index % 2 == 0:  ## even indices just subtract one
                stats += ct_pt_add(logN, L_new)
                # print("sub 1")
            else:
                stats += ct_ct_add(logN, L_new)
                # print("sub ct")
        index_reached = reachable_index
        current_level += 1
        L -= 1

    print("limbs after baby steps", L)

    # giant steps
    index = pow(2, ell)
    gs_L = L - 1  ## drop limb to account for constant mult
    while index < pow(2, m - 1):
        stats += ct_ct_mult(logN, gs_L, dnum)
        # drop a limb
        gs_L -= 1
        # this add replaces multiplication by 2
        stats += ct_ct_add(logN, gs_L)
        stats += ct_ct_add(logN, gs_L)
        index *= 2

    print("limbs before eval recurse", L)

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
            stats += ct_pt_mult(logN, L)
            stats += rescale(logN, L)
            L_new = L - 1
            stats += ct_ct_add(logN, L_new)
        L -= 1

    print("limbs after eval recurse", L)

    for i in range(r):
        stats += ct_ct_mult(logN, L, dnum)
        # drop a limb
        L -= 1
        # this add replces multiplication by 2
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
