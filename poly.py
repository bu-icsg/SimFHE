import params
from perf_counter import PerfCounter
from math import ceil


def add(poly_ctxt: params.PolyContext, arch_param: params.ArchParam):
    """
    This function does not handle any of the reading of inputs or
    writing of the result.
    """
    stats = PerfCounter()
    stats.sw.add = poly_ctxt.N * poly_ctxt.limbs
    stats.arch.add_cyc_fm = stats.sw.add / (arch_param.funits * arch_param.sets)
    stats.arch.add_cyc_sm = arch_param.add_lat + stats.arch.add_cyc_fm
    return stats


def mult(poly_ctxt: params.PolyContext, arch_param: params.ArchParam):
    """
    This function does not handle any of the reading of inputs or
    writing of the result.
    """
    stats = PerfCounter()
    stats.sw.mult = poly_ctxt.N * poly_ctxt.limbs
    stats.arch.mult_cyc_fm = stats.sw.mult / (arch_param.funits * arch_param.sets)
    stats.arch.mult_cyc_sm = arch_param.mult_lat + stats.arch.mult_cyc_fm
    return stats


def ntt_common(poly_ctxt: params.PolyContext, arch_param: params.ArchParam):
    """
    Common NTT and iNTT performance tracker.

    This function does not handle the reading of the input, but it does handle
    the reading in of the NTT coefficients.
    """

    stats = PerfCounter()

    logN = poly_ctxt.logN
    N = poly_ctxt.N
    # logq = poly_ctxt.logq

    num_limbs = poly_ctxt.limbs

    stats.sw.ntt = num_limbs
    stats.sw.mult = (N // 2) * logN * stats.sw.ntt
    stats.sw.add = 2 * stats.sw.mult

    stats.arch.add_cyc_fm = stats.sw.add / (arch_param.funits * arch_param.sets)
    stats.arch.add_cyc_sm = arch_param.add_lat + stats.arch.add_cyc_fm
    stats.arch.mult_cyc_fm = stats.sw.mult / (arch_param.funits * arch_param.sets)
    stats.arch.mult_cyc_sm = arch_param.mult_lat + stats.arch.mult_cyc_fm

    """Reading twiddle factor(worth half a limb)
    Reading phi(worth a limb) 
    Total 1.5 limb worth of data read for NTT parameters
    Division by 8 accounts for converting bits to bytes 
    """
    # stats.arch.dram_ntt_rd = int(ceil((1.5 * N * num_limbs * logq) / 8))
    # stats.arch.dram_ntt = stats.arch.dram_ntt_rd
    stats.arch.ntt_cyc_fm = (N * logN * num_limbs) / (
        2 * arch_param.funits * arch_param.sets
    )
    stats.arch.ntt_cyc_sm = (
        arch_param.ntt_lat
        + ((N * num_limbs) / (2 * arch_param.funits * arch_param.sets))
    ) * logN

    return stats


def compute_phi(poly_ctxt: params.PolyContext, arch_params: params.ArchParam):
    """
    Computes on-the-fly phi and phi inverse values for NTT.
    Assuming all N values can be generated with N multiplications.
    Assumes there is space on the chip for these phi values.
    """
    stats = PerfCounter()

    stats.sw.mult = poly_ctxt.N * poly_ctxt.limbs
    stats.arch.ntt_otf_cyc_fm = (poly_ctxt.N * poly_ctxt.limbs) / (
        arch_params.funits * arch_params.sets
    )
    stats.arch.ntt_otf_cyc_sm = arch_params.mult_lat + stats.arch.ntt_otf_cyc_fm

    return stats


def compute_tf(poly_ctxt: params.PolyContext, arch_params: params.ArchParam):
    """
    Computes on-the-fly twiddle-factor and twiddle-factor inverse values for NTT.
    Assuming all N/2 values can be generated with N/2 multiplications.
    Assumes there is space on the chip for these twiddle factors.
    """
    stats = PerfCounter()

    stats.sw.mult = int(ceil(poly_ctxt.N / 2)) * poly_ctxt.limbs
    stats.arch.mult_cyc_fm = stats.sw.mult / (arch_params.funits * arch_params.sets)
    stats.arch.mult_cyc_sm = arch_params.mult_lat + stats.arch.mult_cyc_fm
    stats.arch.ntt_otf_cyc_fm = stats.arch.mult_cyc_fm
    stats.arch.ntt_otf_cyc_sm = stats.arch.mult_cyc_sm

    return stats


def ntt(poly_ctxt: params.PolyContext, arch_params: params.ArchParam):
    """
    Generates phi and twiddle factors then runs NTT.
    Assumes input polynomial is already in memory.
    """
    compute_phi(poly_ctxt, arch_params)
    compute_tf(poly_ctxt, arch_params)
    ntt_common(poly_ctxt, arch_params)
    return PerfCounter()


def intt(poly_ctxt: params.PolyContext, arch_params: params.ArchParam):
    """
    Generates phi and twiddle factors then runs NTT.
    Assumes input polynomial is already in memory.
    """
    compute_phi(poly_ctxt, arch_params)
    compute_tf(poly_ctxt, arch_params)
    ntt_common(poly_ctxt, arch_params)
    return PerfCounter()


def automorph(poly_ctxt: params.PolyContext, arch_param: params.ArchParam):
    """
    Assumes polynomial is already in memory.
    """
    stats = PerfCounter()
    stats.arch.auto_cyc_fm_wc = poly_ctxt.N * poly_ctxt.limbs
    stats.arch.auto_cyc_fm_bc = (poly_ctxt.N * poly_ctxt.limbs) / (
        arch_param.funits * arch_param.sets
    )
    stats.arch.auto_cyc_sm_wc = arch_param.auto_lat + stats.arch.auto_cyc_fm_wc
    stats.arch.auto_cyc_sm_bc = arch_param.auto_lat + stats.arch.auto_cyc_fm_bc
    return stats
