from math import floor, ceil, sqrt

import params
from params import CacheStyle
import poly
import evaluator
from perf_counter import PerfCounter


def fft_inner_hoisted_unrolled(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    num_diag: int = 7,
):
    """
    The memory of this function is self-contained. Assumes that memory is empty at the
    start and finishes with empty memory.
    """
    stats = PerfCounter()

    key_ctxt = poly_ctxt.key_switch_context()
    limb_rdwr = arch_params.cache_style < CacheStyle.CONST
    reorder_rdwr = limb_rdwr or not arch_params.mod_down_reorder

    evaluator.key_switch_hoisting(poly_ctxt, key_ctxt, arch_params, rd_in=True)
    for idx in range(num_diag - 1):
        rd_a = arch_params.cache_style < CacheStyle.BETA or idx == 0
        # fmt: off
        evaluator.key_switch_inner_product(
            poly_ctxt, key_ctxt, arch_params, automorph=True, rd_in=rd_a, wr_out=limb_rdwr
        )
        # fmt: on

        ## TODO: Check if idx==0 is necessary if cache is large
        rd_b = limb_rdwr or idx == 0
        if rd_b:
            ## read in b to perform correction and raise it's modulus
            stats.arch.dram_limb_rd += poly_ctxt.size_in_bytes
            poly.mult(poly_ctxt, arch_params)  ## multiply by P
        ## all P limbs are zero
        poly.automorph(poly_ctxt, arch_params)

        ## read in the switched limb, fix-up and write-out
        if limb_rdwr:
            stats.arch.dram_limb_rd += poly_ctxt.size_in_bytes
        poly.add(poly_ctxt, arch_params)
        if limb_rdwr:
            stats.arch.dram_limb_wr += poly_ctxt.size_in_bytes

    ## multiplications in the inner product
    for idx in range(num_diag):
        if limb_rdwr:
            stats.arch.dram_limb_rd += 2 * key_ctxt.size_in_bytes
        evaluator.multiply_plain(key_ctxt, arch_params)
        if idx != 0:
            evaluator.add(key_ctxt, arch_params)
    if limb_rdwr:
        stats.arch.dram_limb_wr += 2 * key_ctxt.size_in_bytes

    ## scale down the resulting sum
    poly_ctxt = poly_ctxt.drop()  ## combine with rescale
    # fmt: off
    ## TODO: large cache check
    evaluator.mod_down(key_ctxt, poly_ctxt, arch_params, rd_in=reorder_rdwr, wr_out=True)
    evaluator.mod_down(key_ctxt, poly_ctxt, arch_params, rd_in=reorder_rdwr, wr_out=True)
    # fmt: on

    return stats


def fft_inner_bsgs(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    num_diag: int,
    num_gs: int = 0,
):
    """Generic algorithm to multiply matrix with encrypted vector"""
    stats = PerfCounter()

    if not num_gs:
        num_gs = int(floor(sqrt(num_diag)))
    num_bs = int(ceil(num_diag / num_gs))
    assert num_diag <= (num_bs * num_gs) < (num_diag + num_gs)

    key_switch_ctxt = poly_ctxt.key_switch_context()
    limb_rdwr = arch_params.cache_style < CacheStyle.CONST

    # Perform the hoisted baby step rotations
    evaluator.key_switch_hoisting(poly_ctxt, key_switch_ctxt, arch_params, rd_in=True)
    for idx in range(num_bs - 1):  ## no need to count the rotation by zero
        rd_in = arch_params.cache_style < CacheStyle.BETA or idx == 0
        evaluator.rotate_digits(poly_ctxt, key_switch_ctxt, arch_params, rd_in=rd_in)

    # Perform the accumulation and the giant-step rotations
    min_rot = -1 * (num_diag // 2)
    for start in range(min_rot, min_rot + num_diag, num_bs):
        inner_sz = min(min_rot + num_diag - start, num_bs)
        ## Maintain an accumulator for this outer loop.
        ## Don't need to write until the end of the inner product.
        for idx in range(inner_sz):
            ## TODO: Check if this read is always necessary with large cache
            stats.arch.dram_limb_rd += 2 * poly_ctxt.size_in_bytes
            # stats.arch.dram_plain_rd += poly_ctxt.size_in_bytes
            evaluator.multiply_plain(poly_ctxt, arch_params)
            if idx != 0:
                evaluator.add(poly_ctxt, arch_params)
        if limb_rdwr:
            stats.arch.dram_limb_wr += 2 * poly_ctxt.size_in_bytes

        no_rot = start <= 0 < start + num_bs
        if not no_rot:
            ## same size because rescaling at the end
            evaluator.rotate(poly_ctxt, arch_params, rd_in=limb_rdwr, wr_out=limb_rdwr)

        # optionally read-in rotated value
        if limb_rdwr:
            stats.arch.dram_limb_rd += 2 * poly_ctxt.size_in_bytes
        ## read in and write out accumulator
        ## TODO: check if this write is necessary with large cache
        stats.arch.dram_limb_rd += 2 * poly_ctxt.size_in_bytes
        evaluator.add(poly_ctxt, arch_params, wr_out=True)

    ## rescale
    ## TODO: large cache check
    evaluator.mod_reduce_rescale(poly_ctxt, arch_params, rd_in=True, wr_out=True)
    evaluator.mod_reduce_rescale(poly_ctxt, arch_params, rd_in=True, wr_out=True)

    return stats


def fft_inner_bsgs_hoisted(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    num_diag: int,
    num_gs: int = 0,
):
    """Lattigo double-hoisting algorithm to multiply matrix with encrypted vector"""

    stats = PerfCounter()

    if not num_gs:
        num_gs = int(floor(sqrt(num_diag)))
    num_bs = int(ceil(num_diag / num_gs))
    assert num_diag <= (num_bs * num_gs) < (num_diag + num_gs)

    key_ctxt = poly_ctxt.key_switch_context()
    gs_ctxt = poly_ctxt.drop()  ## combine with rescale
    limb_rdwr = arch_params.cache_style < CacheStyle.CONST
    reorder_rdwr = limb_rdwr or not arch_params.mod_down_reorder

    # Perform the hoisted baby step rotations
    ## TODO: large cache check
    evaluator.key_switch_hoisting(poly_ctxt, key_ctxt, arch_params, rd_in=True)
    for idx in range(num_bs - 1):
        ## TODO: check if idx == 0 is necessary with large cache
        rd_a = arch_params.cache_style < CacheStyle.BETA or idx == 0
        ## TODO: check if num_gs>1 is necessary with a large cache
        wr_a = limb_rdwr or num_gs > 1
        evaluator.key_switch_inner_product(
            poly_ctxt, key_ctxt, arch_params, automorph=True, rd_in=rd_a
        )
        if wr_a:
            stats.arch.dram_limb_wr += key_ctxt.size_in_bytes  # the 'a' limb
        if limb_rdwr:
            stats.arch.dram_limb_wr += key_ctxt.size_in_bytes  # the 'b' limb

        ## TODO: check if idx==0 is necessary with large cache
        rd_b = limb_rdwr or idx == 0
        if rd_b:
            ## read in b to perform correction and raise it's modulus
            stats.arch.dram_limb_rd += poly_ctxt.size_in_bytes
            poly.mult(poly_ctxt, arch_params)  ## multiply by P
        ## all P limbs are zero
        poly.automorph(poly_ctxt, arch_params)

        ## read in the switched limb, fix-up and write-out
        if limb_rdwr:
            stats.arch.dram_limb_rd += poly_ctxt.size_in_bytes
        poly.add(poly_ctxt, arch_params)
        if limb_rdwr:
            stats.arch.dram_limb_wr += poly_ctxt.size_in_bytes

    # Perform the accumulation and the giant-step rotations
    min_rot = -1 * (num_diag // 2)
    for start in range(min_rot, min_rot + num_diag, num_bs):
        inner_sz = min(min_rot + num_diag - start, num_bs)
        for idx in range(inner_sz):
            ## TODO: large cache check
            if limb_rdwr or start != min_rot:
                stats.arch.dram_limb_rd += 2 * key_ctxt.size_in_bytes
            # stats.arch.dram_plain_rd += poly_ctxt.size_in_bytes
            evaluator.multiply_plain(key_ctxt, arch_params)
            if idx != 0:
                evaluator.add(key_ctxt, arch_params)
        if limb_rdwr:
            stats.arch.dram_limb_wr += 2 * key_ctxt.size_in_bytes

        if num_gs > 1:
            ## scale down the resulting sum
            # fmt: off
            ## TODO: Large cache check
            evaluator.mod_down(key_ctxt, gs_ctxt, arch_params, rd_in=reorder_rdwr, wr_out=True)
            evaluator.mod_down(key_ctxt, gs_ctxt, arch_params, rd_in=reorder_rdwr, wr_out=True)
            # fmt: on

            no_rot = start <= 0 < start + num_bs
            if no_rot:
                evaluator.multiply_plain(gs_ctxt, arch_params)
            else:
                # fmt: off
                evaluator.key_switch_hoisting(gs_ctxt, key_ctxt, arch_params, rd_in=True)
                evaluator.key_switch_inner_product(gs_ctxt, key_ctxt, arch_params)
                # fmt: on

                ## read in b to perform correction and raise it's modulus
                ## TODO: large cache check
                stats.arch.dram_limb_rd += gs_ctxt.size_in_bytes
                poly.mult(gs_ctxt, arch_params)  ## multiply by P

                ## all P limbs are zero
                poly.automorph(gs_ctxt, arch_params)
                poly.add(gs_ctxt, arch_params)

            if start != min_rot:
                ## TODO: large cache check
                stats.arch.dram_limb_rd += 2 * key_ctxt.size_in_bytes
                evaluator.add(key_ctxt, arch_params)
            ## TODO: large cache check
            stats.arch.dram_limb_wr += 2 * key_ctxt.size_in_bytes

    ## scale down the resulting sum
    # fmt: off
    evaluator.mod_down(key_ctxt, gs_ctxt, arch_params, rd_in=reorder_rdwr, wr_out=True)
    evaluator.mod_down(key_ctxt, gs_ctxt, arch_params, rd_in=reorder_rdwr, wr_out=True)
    # fmt: on

    return stats


def fft(
    poly_ctxt: params.PolyContext,
    scheme_params: params.SchemeParams,
):
    """
    The memory of this function is self-contained. Assumes that memory is empty at the
    start and finishes with empty memory.
    """
    stats = PerfCounter()

    start_limbs = poly_ctxt.limbs

    for log_radix in scheme_params.bsgs_iters:
        dim = pow(2, log_radix + 1) - 1  ## 2r-1
        if scheme_params.fft_style == params.FFTStyle.UNROLLED_HOISTED:
            fft_inner_hoisted_unrolled(poly_ctxt, scheme_params.arch_param, dim)
        elif scheme_params.fft_style == params.FFTStyle.BSGS:
            fft_inner_bsgs(poly_ctxt, scheme_params.arch_param, dim)
        elif scheme_params.fft_style == params.FFTStyle.BSGS_HOISTED:
            fft_inner_bsgs_hoisted(poly_ctxt, scheme_params.arch_param, dim)
        else:
            raise ValueError("unknown FFT style")
        poly_ctxt = poly_ctxt.drop()  ## rescaling in bsgs

    """
    We need to ensure that the `claimed` number of limbs required for the FFT is 
    the actual number of limbs we used. This assertion should catch discrepancies.
    """
    assert poly_ctxt.limbs == start_limbs - scheme_params.he_fft_limbs

    # print("fft end limbs", poly_ctxt.limbs)

    return stats
