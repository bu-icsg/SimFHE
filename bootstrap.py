from math import ceil, log2
import logging

from perf_counter import PerfCounter
import evaluator
import fft
import eval_sine
import params


def conj_and_add(poly_ctxt: params.PolyContext, arch_params: params.ArchParam):
    """
    The memory of this function is self-contained. Assumes that memory is empty at the
    start and finishes with empty memory.
    """
    stats = PerfCounter()

    ## TODO: large cache check
    stats.arch.dram_limb_rd += 2 * poly_ctxt.size_in_bytes

    evaluator.rotate(poly_ctxt, arch_params)

    ## read in the input again for the addition and subtraction
    ## TODO: large cache check
    stats.arch.dram_limb_rd += 2 * poly_ctxt.size_in_bytes

    evaluator.add(poly_ctxt, arch_params)
    evaluator.add(poly_ctxt, arch_params)

    ## TODO: large cache check
    stats.arch.dram_limb_wr += 2 * poly_ctxt.size_in_bytes

    return stats


def coeff_to_slot(input_ctxt: params.PolyContext, scheme_params: params.SchemeParams):
    """
    The memory of this function is self-contained. Assumes that memory is empty at the
    start and finishes with empty memory.
    """
    stats = PerfCounter()

    fft.fft(input_ctxt, scheme_params)

    ## write out FFT result

    """
    This is the correct number of output limbs as enforced by the assertion at 
    the end of the fft function
    """

    conj_ctxt = input_ctxt.drop(scheme_params.he_fft_limbs)
    conj_and_add(conj_ctxt, scheme_params.arch_param)

    ## the multiplication by i is absorbed in the
    ## next multiplication in the non-linear approximation.

    return stats


def slot_to_coeff(poly_ctxt: params.PolyContext, scheme_params: params.SchemeParams):
    stats = PerfCounter()

    ## handle the last multiplication by i in the polynomial evaluation

    evaluator.add(poly_ctxt, scheme_params.arch_param)

    fft.fft(poly_ctxt, scheme_params)

    return stats


def bootstrap(scheme_params: params.SchemeParams):
    stats = PerfCounter()

    ## nominal modulus raising
    coeff_to_slot(scheme_params.mod_raise_ctxt, scheme_params)

    if scheme_params.arch_param.cache_style >= params.CacheStyle.HUGE:
        ## cache the multiplication key
        key_ctxt = scheme_params.cts_ctxt.key_switch_context()
        stats.arch.dram_auto_rd += scheme_params.arch_param.key_sz(
            key_ctxt.size_in_bytes
        )

    for _ in range(2):
        eval_sine.eval_sine(scheme_params.cts_ctxt, scheme_params)
    ## write out eval sine result
    ## TODO: large cache check
    stats.arch.dram_limb_wr += 2 * 2 * scheme_params.eval_sine_ctxt.size_in_bytes
    slot_to_coeff(scheme_params.eval_sine_ctxt, scheme_params)

    return stats
