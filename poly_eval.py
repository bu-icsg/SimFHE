from math import ceil, log2
import logging

import params
import evaluator
from perf_counter import PerfCounter


def baby_step_basis(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    bs_depth: int,
):
    stats = PerfCounter()

    for idx in range(1, bs_depth):
        ## from 2^i +1 to 2^{i+1}
        ## perform all multiplications
        for exp in range(2 ** (idx), 2 ** (idx + 1)):
            sqr = exp % 2 == 0
            evaluator.multiply(poly_ctxt, arch_params, sqr, rd_in=True, wr_out=True)

        ## rescale all previous ciphertexts
        for _ in range(1, 2**idx):
            evaluator.mod_reduce_rescale(
                poly_ctxt, arch_params, rd_in=True, wr_out=True
            )

        poly_ctxt = poly_ctxt.drop()

    logging.debug("baby step computation result limbs", poly_ctxt.limbs)

    return stats


def giant_step_basis(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    gs_depth: int,
):
    stats = PerfCounter()

    print("in giant_step_basis")

    for idx in range(gs_depth):
        key_cached = arch_params.cache_style >= params.CacheStyle.HUGE and (idx > 0)
        evaluator.double_multiply(
            poly_ctxt, arch_params, sqr=True, rd_in=True, wr_out=True, key_cached=key_cached
        )
        poly_ctxt = poly_ctxt.drop()

    return stats


def baby_step_leafs(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    bs_depth: int,
    gs_depth: int,
):
    stats = PerfCounter()

    bs_size = pow(2, bs_depth)
    gs_size = pow(2, gs_depth)

    for nleaf in range(gs_size):
        for i in range(1, bs_size):
            ## reading in inner product input
            if arch_params.cache_style < params.CacheStyle.ALPHA and nleaf == 0:
                ## reuse this across all limbs
                stats.arch.dram_limb_rd += 2 * poly_ctxt.size_in_bytes
            evaluator.multiply_plain(poly_ctxt, arch_params)
            if i == 1:
                evaluator.add_plain(poly_ctxt, arch_params)
            else:
                evaluator.add(poly_ctxt, arch_params)
        ## TODO: Check if this write is necessary with a large cache
        stats.arch.dram_limb_wr += 2 * poly_ctxt.size_in_bytes

        evaluator.mod_reduce_rescale(poly_ctxt, arch_params, rd_in=True, wr_out=True)
    return stats


def giant_step_accumulate(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    gs_depth: int,
):
    stats = PerfCounter()

    gs_size = pow(2, gs_depth)
    print(gs_depth, gs_size)

    for i in range(gs_depth - 1, -1, -1):
        num_nodes = pow(2, i)
        print(num_nodes)
        for _ in range(num_nodes):
            evaluator.multiply(poly_ctxt, arch_params, rd_in=True, wr_out=False)
            red_ctxt = poly_ctxt.drop()

            # Merge this multiply-plain and addition with correction in multiply
            # Avoids extra mod_reduce_rescale
            ## TODO: check if these reads and writes are necessary with a large cache
            stats.arch.dram_limb_rd += 2 * poly_ctxt.size_in_bytes
            evaluator.multiply_plain(poly_ctxt, arch_params)
            evaluator.add(red_ctxt, arch_params)

            stats.arch.dram_limb_wr += 2 * red_ctxt.size_in_bytes

        poly_ctxt = poly_ctxt.drop()

    return stats


def exp(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    exponent: int,
):
    stats = PerfCounter()

    for _ in range(exponent):
        evaluator.double_multiply(
            poly_ctxt, arch_params, sqr=True, rd_in=True, wr_out=True
        )
        poly_ctxt = poly_ctxt.drop()

    return stats


def poly_eval(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    degree: int,
    exponent: int,
    bs_depth: int = 0,
):
    """
    The memory of this function is self-contained. Assumes that memory is empty at the
    start and finishes with empty memory.
    """
    stats = PerfCounter()

    total_depth = int(ceil(log2(degree + 1)))
    if bs_depth == 0:
        bs_depth = total_depth // 2
    gs_depth = total_depth - bs_depth  ## [ell, m-1] inclusive

    start_limbs = poly_ctxt.limbs
    baby_step_basis(poly_ctxt, arch_params, bs_depth)
    poly_ctxt = poly_ctxt.drop(bs_depth)

    giant_step_basis(poly_ctxt, arch_params, gs_depth)

    baby_step_leafs(poly_ctxt, arch_params, bs_depth, gs_depth)
    poly_ctxt = poly_ctxt.drop()

    giant_step_accumulate(poly_ctxt, arch_params, gs_depth)
    poly_ctxt = poly_ctxt.drop(gs_depth)
    print("done")

    exp(poly_ctxt, arch_params, exponent)
    poly_ctxt = poly_ctxt.drop(exponent)

    assert poly_ctxt.limbs == start_limbs - total_depth - 1 - exponent

    return stats
