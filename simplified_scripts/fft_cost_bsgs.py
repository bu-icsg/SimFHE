import matplotlib.pyplot as plt
import numpy as np
from math import ceil, log, log2, sqrt
from dataclasses import dataclass


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
        return out_str


def rotate(N: int, L: int):
    pass


def mod_raise(logN: int, L: int, dnum: int):
    """
    Raises modulus from beta sets of alpha digits
    to beta sets of (L+alpha) digits.

    This function reads in all inputs and writes out all outputs.
    Algorithm:
    1) Read in input
    2) Decompose (2 multiplication)
    3) Inverse NTT
    4) Write out data
    5) Read in data slot-wise (all limbs of a single element)
    6) Basis Conversion (adds and mults)
    7) Write out new limbs
    8) Read in new limbs limb-wise (all slots of a single limb)
    9) Forward NTT of the new limbs
    10) Write out the new limbs
    """

    alpha = int(ceil(L / dnum))
    beta = dnum
    N = 1 << logN

    print("L", L)
    # print("alpha", alpha)
    # print("beta", beta)

    stats = Stats(logN=logN)

    # adds = 0
    # mults = 0
    # ntts = 0
    new_limbs = L + 1
    # limb_rds = 0
    # limb_wrs = 0
    # print("mod raise new limbs", new_limbs)

    ## decompose.
    stats.mult += 2 * N * beta * alpha
    # stats.limb_rd += N * L  ## reading in input

    ## Basis conversion

    ## inverse NTTs of input
    ## not as accurate as L....
    stats.ntt += alpha * beta
    stats.limb_wr += N * alpha * beta  ## write the result of the iNTT

    # print("put me back")
    ## adds from basis conversion

    ## reading in iNTT output in slot-wise (all limbs of a single element)
    stats.limb_rd += N * alpha * beta

    stats.add += N * beta * alpha * new_limbs
    ##  multiplications from basis conversion
    stats.mult += N * beta * alpha * (new_limbs + 1)

    ## write out the output of basis conversion
    stats.limb_wr += N * new_limbs * beta

    ## forward NTTs of output
    ## reading in input in limb-wise (all slots of a single limb)
    stats.limb_rd += N * beta * new_limbs
    stats.ntt += beta * new_limbs
    ## writing NTT result
    stats.limb_wr += N * beta * new_limbs

    return stats


def inner_product(logN: int, L: int, dnum: int):
    """
    Performs a length-dnum inner product with a decomposed and raised polynomial
    with an automorphism key.
    """
    alpha = int(ceil(L / dnum))
    beta = dnum
    N = 1 << logN

    stats = Stats(logN=logN)

    limbs = L + 1 + alpha

    ## length beta inner product.
    ## times 2 due to ct mult
    stats.mult = 2 * beta * limbs * N
    stats.add = 2 * (beta - 1) * limbs * N

    ## not times 2 because a term is pseudorandom
    stats.auto_rd = beta * N * limbs
    stats.limb_rd = beta * N * limbs

    return stats


def mod_reduce(logN: int, L: int, dnum: int):
    """
    Takes in the original number of limbs (output limbs) as well as dnum.
    Computes the correct starting number of limbs and then counts
    the operations in the mod reduction function from L+alpha limbs
    down to L limbs.
    """
    alpha = int(ceil(L / dnum))
    N = 1 << logN

    stats = Stats(logN=logN)

    input_limbs = L + alpha + 1
    output_limbs = L
    delta_limbs = input_limbs - output_limbs

    print("delta limbs", delta_limbs)

    """
    Algorithm:
    1) Read in the last alpha limbs
    2) Inverse NTT on these limbs
    3) Write out these limbs
    4) Read in these limbs slot-wise (all limbs of a single element)
    5) Convert these limbs into the basis of the L output limbs
    6) Write out these L limbs
    7) Read in these L limbs limb-wise (all slots of a single limb)
    8) Forward NTT
    9) Read in original limbs and subtract away. NOTE: could optimise away this read.
    10) Write out result limbs.
    """

    stats.ntt += delta_limbs
    stats.limb_wr += N * delta_limbs  ## write ntt output

    # print("remove this")
    # return stats

    ## read in basis conversion input
    stats.limb_rd += N * delta_limbs

    ## basis conversion
    stats.add = N * delta_limbs * output_limbs
    stats.mult = N * delta_limbs * (output_limbs + 1)

    stats.limb_wr += N * output_limbs

    # print("remove this")
    # return stats

    ## limb-wise read for the ntt
    stats.limb_rd += N * output_limbs

    stats.ntt += output_limbs

    stats.add += N * output_limbs
    stats.mult += N * output_limbs

    return stats


def poly_add(logN: int, L: int):
    """
    Polynomial add.
    Assumes that both of the operands are already in memory.
    Does not write out result.
    """
    N = 1 << logN

    stats = Stats(logN=logN)

    stats.add = L * N

    return stats


def multiply_plain(logN: int, L: int):
    """
    Plaintext-ciphertext multiplication.
    Assumes that both of the operands are already in memory.
    Does not write out result.
    """
    N = 1 << logN

    stats = Stats(logN=logN)

    stats.mult = 2 * N * L

    return stats


def poly_mult(logN: int, L: int):
    """
    Polynomial multiplication.
    Assumes that both of the operands are already in memory.
    Does not write out result.
    """
    N = 1 << logN

    stats = Stats(logN=logN)

    stats.mult = N * L

    return stats


def mod_reduce_rescale(logN: int, L: int):
    """
    Performs the rescaling operation on one ciphertext (two polynomials).
    Assumes that both of the operands are already in memory.
    Does not write out result.
    Assumes that all necessary NTTs have already been performed.
    """
    N = 1 << logN

    stats = Stats(logN=logN)

    ## 1 inv ntt and L-1 output ntts
    stats.ntt += L

    stats.add = 2 * N * (L - 1)
    stats.mult = stats.add

    return stats


def rotation(logN: int, L: int, dnum: int, chip_mem_size: str):
    """
    Full rotation over a ciphertext
    """

    stats = Stats(logN=logN)

    rotate(logN, L)

    stats += mod_raise(logN, L, dnum)

    # print("remove this")
    # return stats

    stats += inner_product(logN, L, dnum)

    # print("remove this")
    # return stats

    stats += mod_reduce(logN, L, dnum)
    stats += mod_reduce(logN, L, dnum)

    # print("remove this")
    # return stats

    ## Adds message to key switch ciphertext. Just a single polynomial add
    ## Need to read in message from dram
    stats += poly_add(logN, L)
    # stats.limb_rd += (1 << logN) * L

    return stats


def rotate_digits(logN: int, L: int, dnum: int):
    """
    Rotates a ciphertext after hoisting of the modulus raising.
    Assumes that nothing is in memory.
    L is the ORIGINAL number of limbs, NOT the number of limbs in the raised modulus.
    """
    beta = dnum
    alpha = int(ceil(L / dnum))

    raised_limbs = L + 1 + alpha
    N = 1 << logN

    stats = Stats(logN=logN)

    for _ in range(beta):
        stats.limb_rd += N * raised_limbs
        # rotate(logN, raised_limbs)  ## rotate the digits

    # print("raised_limbs", raised_limbs)

    ## length beta inner product with key
    for i in range(beta):
        """
        Length beta inner product with the auto key.
        Need to read in the auto key.
        Assumes we have the space for an accumulator to store the
        two result limbs in memory.
        Only write out the result of the inner product.
        """
        ## no multiply by 2 because the a term is pseudorandom
        stats.auto_rd += N * raised_limbs  ## auto key read

        stats += poly_mult(logN, raised_limbs)
        stats += poly_mult(logN, raised_limbs)
        if i != 0:
            stats += poly_add(logN, raised_limbs)
            stats += poly_add(logN, raised_limbs)
    stats.limb_wr += 2 * N * raised_limbs

    # print("remove this")
    # return stats

    ## reduce both polys in the ciphertext
    stats.limb_rd += 2 * N * raised_limbs
    stats += mod_reduce(logN, L, dnum)
    stats += mod_reduce(logN, L, dnum)

    # print("remove this")
    # return stats

    stats.limb_rd += N * L  ## read the message
    rotate(logN, L)  ## rotate and add the message
    stats += poly_add(logN, L)
    stats.limb_wr += 2 * N * L

    return stats


def baby_step_giant_step(logN: int, L: int, dnum: int, dim: int, chip_mem_size: str):
    t = int(ceil(sqrt(dim)))

    abs_rot = (dim - 1) // 2

    start = -1 * t * ceil(abs_rot / t)
    end = t * (abs_rot // t)
    # if (d - 1) % (2 * t) == 0:
    # end -= t
    min_rot, max_rot = -1 * abs_rot, abs_rot
    print("min_rot, max_rot", min_rot, max_rot)
    print("start, end", start, end)

    print("d, t", dim, t)

    stats = Stats(logN=logN)

    stats.limb_rd += (1 << logN) * L

    stats += mod_raise(logN, L, dnum)

    # print("remove this")
    # return stats

    for _ in range(t - 1):
        stats += rotate_digits(logN, L, dnum)

    # print("remove this")
    # return stats

    for base in range(start, end + t, t):
        added = False
        for index in range(base, base + t):
            if min_rot <= index <= max_rot:
                stats.limb_rd += 2 * (1 << logN) * L
                stats.plain_rd += (1 << logN) * L
                stats += multiply_plain(logN, L)
                if added:
                    stats += poly_add(logN, L)
                    stats += poly_add(logN, L)
                added = True
        stats.limb_wr += 2 * (1 << logN) * L

        if base != 0:
            stats.limb_rd += 2 * (1 << logN) * L
            stats += rotation(logN, L, dnum, chip_mem_size)
            stats.limb_rd += 2 * (1 << logN) * L
            stats += poly_add(logN, L)
            stats += poly_add(logN, L)
            stats.limb_wr += 2 * (1 << logN) * L

    # print("remove this")
    # return stats

    ## rescale
    stats.limb_rd += (1 << logN) * L
    stats += mod_reduce_rescale(logN, L)
    stats.limb_wr += (1 << logN) * L

    return stats


def fft_bsgs(logN: int, dnum: int, u: int, chip_mem_size: str):
    stats = Stats()

    assert logN == 17

    if u == 1:
        bsgs_iters = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        levels_required = 16
    elif u == 2:
        bsgs_iters = [2, 2, 2, 2, 2, 2, 2, 2]
        levels_required = 8
    elif u == 3:
        bsgs_iters = [3, 3, 3, 3, 4]
        levels_required = 5
    elif u == 4:
        bsgs_iters = [4, 4, 4, 4]
        levels_required = 4
    elif u == 5:
        bsgs_iters = [5, 5, 6]
        levels_required = 3

    fresh_limbs = 23
    L = (2 * levels_required) + 11 + fresh_limbs

    for log_radix in bsgs_iters:
        dim = pow(2, log_radix + 1) - 1
        stats += baby_step_giant_step(logN, L, dnum, dim, chip_mem_size)
        # drop a limb
        L -= 1
        # print("remove this")
        # return stats

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
    chip_mem_size = "large"
    logq = 50

    u = 2

    # for u in range(1, 6):
    stats.append(fft_bsgs(logN, dnum, u, chip_mem_size))

    # K = 1
    for val in stats:
        # if K == 2:
        # print("K: %s" % K)
        print(val)
        # K += 1

    # stacked_plot_compute(stats)
    # stacked_plot_dram_transfers(stats)

    return 0


if __name__ == "__main__":
    main()
