from math import ceil, sqrt
from os import stat

import poly
import params
from params import CacheStyle
from perf_counter import PerfCounter
from profiler import Profiler

"""
These operations are over cipehrtexts. The poly context inputs
represent the size of one of the polynomials in the ciphertext.
"""


def add(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    rd_in: bool = False,
    wr_out: bool = False,
):
    """Add two ciphertexts"""
    stats = PerfCounter()

    if rd_in:
        stats.arch.dram_limb_rd += 4 * poly_ctxt.size_in_bytes

    ## two logq scalars
    stats.arch.min_bytes = 2 * int(ceil(poly_ctxt.logq / 8))

    for _ in range(2):
        poly.add(poly_ctxt, arch_params)

    if wr_out:
        stats.arch.dram_limb_wr += 2 * poly_ctxt.size_in_bytes

    return stats


def add_plain(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    rd_in: bool = False,
    wr_out: bool = False,
):
    """Add a ciphertext to a plain-text vector"""
    stats = PerfCounter()

    if rd_in:
        stats.arch.dram_limb_rd += 2 * poly_ctxt.size_in_bytes

    ## two logq scalars
    stats.arch.min_bytes = 2 * int(ceil(poly_ctxt.logq / 8))

    poly.add(poly_ctxt, arch_params)

    if wr_out:
        stats.arch.dram_limb_wr += poly_ctxt.size_in_bytes

    return stats


def multiply_plain(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    rd_in: bool = False,
    wr_out: bool = False,
):
    """Multiply a ciphertext to a plain-text vector"""
    stats = PerfCounter()

    if rd_in:
        stats.arch.dram_limb_rd += 3 * poly_ctxt.size_in_bytes

    ## two logq scalars
    stats.arch.min_bytes = 2 * int(ceil(poly_ctxt.logq / 8))

    for _ in range(2):
        poly.mult(poly_ctxt, arch_params)

    if wr_out:
        stats.arch.dram_limb_wr += 2 * poly_ctxt.size_in_bytes

    return stats


def rotate_inner(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    rd_in: bool = False,
    wr_out: bool = False,
):
    """Rotate both polynomials in a ciphertext"""
    stats = PerfCounter()

    if rd_in:
        stats.arch.dram_limb_rd += 2 * poly_ctxt.size_in_bytes

    ## one limb
    stats.arch.min_bytes = poly_ctxt.logq * poly_ctxt.N / 8

    for _ in range(2):
        poly.automorph(poly_ctxt, arch_params)

    if wr_out:
        stats.arch.dram_limb_wr += 2 * poly_ctxt.size_in_bytes

    return stats


def decompose(poly_ctxt: params.PolyContext):
    """
    Decompose a Q basis polynomial into dnum alpha-limb digits.
    Requires a pair of modular multiplications per slot
    rd_in: handled by caller (includes constants), wr_out: handled by caller
    """
    stats = PerfCounter()
    stats.sw.mult = 2 * poly_ctxt.N * poly_ctxt.limbs
    return stats


def basis_convert(
    input_ctxt: params.PolyContext,
    output_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
):
    """
    Generate output_ctxt.limbs from input_ctxt.limbs
    This models the algorithm that takes in limbs x_1, ... , x_L and outputs
    x_j = sum_{i=1}^L [x_i * q_i'^{-1}] * q_i' mod p_j
    for each j from 1 to num_new_limbs.
    rd_in: handled by caller (includes constants), wr_out: handled by caller
    """

    ## No reads or writes in this function...

    stats = PerfCounter()

    stats.sw.add = input_ctxt.N * input_ctxt.limbs * output_ctxt.limbs
    stats.sw.mult = input_ctxt.N * input_ctxt.limbs * (1 + output_ctxt.limbs)

    stats.arch.add_cyc_fm = stats.sw.add / (arch_params.funits * arch_params.sets)
    stats.arch.add_cyc_sm = arch_params.add_lat + stats.arch.add_cyc_fm
    stats.arch.mult_cyc_fm = stats.sw.mult / (arch_params.funits * arch_params.sets)
    stats.arch.mult_cyc_sm = arch_params.mult_lat + stats.arch.mult_cyc_fm
    stats.arch.bsc_cyc_fm = stats.arch.mult_cyc_fm + stats.arch.add_cyc_fm
    stats.arch.bsc_cyc_sm = stats.arch.mult_cyc_sm + stats.arch.add_cyc_sm

    return stats


def mod_raise(
    input_ctxt: params.PolyContext,
    output_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    rd_in: bool = False,
    wr_out: bool = False,
):
    """
    Raises the modulus of a single polynomial from the input basis to the output basis.
    Handles the intermediate reads and writes to switch from limb-wise reads to slot-wise reads.
    """
    stats = PerfCounter()

    delta_ctxt = output_ctxt.drop(input_ctxt.limbs)

    if rd_in:
        stats.arch.dram_limb_rd += input_ctxt.size_in_bytes

    poly.intt(input_ctxt, arch_params)
    if arch_params.cache_style < CacheStyle.ALPHA:
        ## write out limbs (all slot of a given limb)
        stats.arch.dram_limb_wr += input_ctxt.size_in_bytes

        ## Reads in the input data limb-wise (all limbs of a single element)
        ## to perform the basis conversion
        stats.arch.dram_limb_rd += input_ctxt.size_in_bytes

    basis_convert(input_ctxt, delta_ctxt, arch_params)
    ## Writes out the resulting limbs
    if arch_params.cache_style < CacheStyle.ALPHA:
        ## this writes out the limb in coeff form to make space for the new limbs.
        ## if we have alpha limbs of cache, we have space for the entire new limb,
        ## so we can immediately perform the forward ntt
        stats.arch.dram_limb_wr += delta_ctxt.size_in_bytes

        ## Reads in the new limbs slot-wise (all slots for a single limb) for the NTT
        stats.arch.dram_limb_rd += delta_ctxt.size_in_bytes

    ## only the new limbs need to be taken to Eval rep
    poly.ntt(delta_ctxt, arch_params)

    if wr_out:
        # only write out the new limbs since the old limbs are in memory already
        stats.arch.dram_limb_wr += delta_ctxt.size_in_bytes

    return stats


def mod_down_reduce(
    input_ctxt: params.PolyContext,
    output_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
):
    """
    Given x in PQ basis, return x mod P in the Q basis
    input rd: handled by caller, output wr: handled by caller
    """
    stats = PerfCounter()

    basis_convert_input_ctxt = input_ctxt.drop(output_ctxt.limbs)
    poly.intt(basis_convert_input_ctxt, arch_params)

    if arch_params.cache_style < CacheStyle.ALPHA:
        ## write limb-wise and read slot-wise
        stats.arch.dram_limb_wr += basis_convert_input_ctxt.size_in_bytes
        stats.arch.dram_limb_rd += basis_convert_input_ctxt.size_in_bytes

    basis_convert(basis_convert_input_ctxt, output_ctxt, arch_params)

    if arch_params.cache_style < CacheStyle.ALPHA:
        ## read slot-wise and write limb-wise
        stats.arch.dram_limb_wr += output_ctxt.size_in_bytes
        stats.arch.dram_limb_rd += output_ctxt.size_in_bytes

    poly.ntt(output_ctxt, arch_params)

    return stats


def mod_down_divide(poly_ctxt: params.PolyContext, arch_params: params.ArchParam):
    """
    Given x mod Q and x mod P in the Q basis, return x/P in the Q basis
    input rd: handled by caller, output wr: handled by caller
    """
    stats = PerfCounter()
    poly.add(poly_ctxt, arch_params)
    poly.mult(poly_ctxt, arch_params)

    return stats


def mod_down(
    input_ctxt: params.PolyContext,
    output_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    rd_in: bool = False,
    wr_out: bool = False,
):
    """
    Given x in PQ basis, return x/P in the Q basis
    """

    stats = PerfCounter()
    if rd_in:
        # read the limbs to be reduced
        basis_convert_input_ctxt = input_ctxt.drop(output_ctxt.limbs)
        stats.arch.dram_limb_rd += basis_convert_input_ctxt.size_in_bytes
    mod_down_reduce(input_ctxt, output_ctxt, arch_params)

    if rd_in:
        ## read in the original limbs to finish mod_down
        stats.arch.dram_limb_rd += output_ctxt.size_in_bytes
    mod_down_divide(output_ctxt, arch_params)

    if wr_out:
        stats.arch.dram_limb_wr += output_ctxt.size_in_bytes

    return stats


def key_switch_hoisting(
    poly_ctxt: params.PolyContext,
    key_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    rd_in: bool = False,
    wr_out: bool = True,
):
    """
    Decompose the input limb and mod-raise each resulting digit
    input rd: handled by caller, output wr: handled here
    """

    stats = PerfCounter()

    alpha = poly_ctxt.alpha
    beta = poly_ctxt.dnum

    new_limbs = key_ctxt.limbs - alpha
    assert new_limbs >= 0

    new_limb_ctxt = poly_ctxt.basis_convert(new_limbs)

    if rd_in:
        stats.arch.dram_limb_rd += poly_ctxt.size_in_bytes

    digit_context = poly_ctxt.basis_convert(alpha)
    for _ in range(beta):
        decompose(digit_context)
        ## Don't write because we immediately do inverse NTT. The
        ## inverse NTT performs the write
        ## input is the output of decompose
        mod_raise(digit_context, key_ctxt, arch_params)
        if wr_out:
            stats.arch.dram_limb_wr += new_limb_ctxt.size_in_bytes

    return stats


def key_switch_inner_product(
    poly_ctxt: params.PolyContext,
    key_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    automorph: bool = False,
    rd_in: bool = True,
    wr_out: bool = False,
    key_cached: bool = False,  ## True if the key-switch key is already in cache
):
    """
    Multiply a mod-raised limb with a key-switching key
    input rd: handled here, output wr: handled by caller
    """
    stats = PerfCounter()

    beta = poly_ctxt.dnum

    for i in range(beta):
        ## reading in a
        if rd_in:
            stats.arch.dram_limb_rd += key_ctxt.size_in_bytes
        if automorph:
            poly.automorph(key_ctxt, arch_params)
        ## reading in key
        ## TODO: large cache check
        if not key_cached:
            stats.arch.dram_auto_rd += arch_params.key_sz(key_ctxt.size_in_bytes)
        multiply_plain(key_ctxt, arch_params)
        if i != 0:
            add(key_ctxt, arch_params)

    if wr_out:
        stats.arch.dram_limb_wr += 2 * key_ctxt.size_in_bytes

    return stats


def key_switch(
    poly_ctxt: params.PolyContext,
    key_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    rd_in: bool = False,
    wr_out: bool = False,
    key_cached: bool = False,
):
    """
    Switch the key of a ciphertext (ref: Han and Ki '19)
    """
    stats = PerfCounter()
    limb_rdwr = arch_params.cache_style < CacheStyle.CONST

    key_switch_hoisting(poly_ctxt, key_ctxt, arch_params, rd_in=rd_in, wr_out=True)
    key_switch_inner_product(
        poly_ctxt,
        key_ctxt,
        arch_params,
        automorph=False,
        rd_in=True,
        wr_out=limb_rdwr,
        key_cached=key_cached,
    )

    ## one mod reduction for each polynomial in the key switch result
    ## mod_down writes the a term immediately and the b term after fix-up
    mod_down(key_ctxt, poly_ctxt, arch_params, rd_in=limb_rdwr, wr_out=wr_out)
    mod_down(key_ctxt, poly_ctxt, arch_params, rd_in=limb_rdwr, wr_out=False)
    if rd_in:
        stats.arch.dram_limb_rd += poly_ctxt.size_in_bytes
    poly.add(poly_ctxt, arch_params)
    if wr_out:
        stats.arch.dram_limb_wr += poly_ctxt.size_in_bytes

    return stats


def mod_reduce_rescale(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    rd_in: bool = False,
    wr_out: bool = False,
):
    """
    Rescale a polymonials by dropping the final limb.
    """
    stats = PerfCounter()

    delta_ctxt = poly_ctxt.basis_convert(1)
    output_ctxt = poly_ctxt.drop()

    if rd_in:
        stats.arch.dram_limb_rd += delta_ctxt.size_in_bytes
    poly.intt(delta_ctxt, arch_params)

    stats.sw.add = (poly_ctxt.limbs - 1) * poly_ctxt.N
    stats.sw.mult = (poly_ctxt.limbs - 1) * poly_ctxt.N

    poly.ntt(output_ctxt, arch_params)
    if rd_in:
        stats.arch.dram_limb_rd += output_ctxt.size_in_bytes
    if wr_out:
        stats.arch.dram_limb_wr += output_ctxt.size_in_bytes

    return stats


def rotate(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    rd_in: bool = False,
    wr_out: bool = False,
):
    """
    Performs a full automorphism on a ciphertext.
    input rd: handled by caller, output wr: handled by caller
    """
    stats = PerfCounter()

    # when a small constant number of limbs can be cached:
    # read the a-component one-limb at a time, call rotate inner on that limb
    # pass the limbs to key-switch and generate the output limbs
    # note that b-limbs are read in and rotated during the fix-up in key-switch
    # if not enabled: rotate_inner, write-back and read again for key-switch
    limb_rdwr = arch_params.cache_style < CacheStyle.CONST

    rotate_inner(poly_ctxt, arch_params, rd_in=rd_in, wr_out=limb_rdwr)

    key_ctxt = poly_ctxt.key_switch_context()
    key_switch(poly_ctxt, key_ctxt, arch_params, rd_in=limb_rdwr, wr_out=wr_out)

    return stats


def rotate_digits(
    poly_ctxt: params.PolyContext,
    key_switch_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    rd_in: bool = True,
    wr_out: bool = True,
):
    """
    Single rotation of a hoisted ciphertext.
    """

    ## TODO: Writes are always happending here.
    ## Check if this is necessary with a very big cache.

    stats = PerfCounter()

    beta = poly_ctxt.dnum

    for i in range(beta):
        ## NOTE: We're reading in the same limb for each rotation
        ## We could pass the responsibility of reading in the input
        ## to the caller to handle the reuse of these limbs.
        ## We're not including this optimization so that this
        ## function acts as a baseline
        if rd_in:
            stats.arch.dram_limb_rd += key_switch_ctxt.size_in_bytes
        poly.automorph(key_switch_ctxt, arch_params)  ## rotate the digits
        ## TODO: large cache check
        stats.arch.dram_auto_rd += arch_params.key_sz(key_switch_ctxt.size_in_bytes)
        multiply_plain(key_switch_ctxt, arch_params)
        if i != 0:
            add(key_switch_ctxt, arch_params)
    ## write out accumulated result
    ## TODO: large cache check
    stats.arch.dram_limb_wr += 2 * key_switch_ctxt.size_in_bytes

    mod_down(key_switch_ctxt, poly_ctxt, arch_params, rd_in=True, wr_out=wr_out)
    mod_down(key_switch_ctxt, poly_ctxt, arch_params, rd_in=True)
    if rd_in:
        stats.arch.dram_limb_rd += poly_ctxt.size_in_bytes
    poly.automorph(poly_ctxt, arch_params)  ## rotate the message
    poly.add(poly_ctxt, arch_params)
    if wr_out:
        stats.arch.dram_limb_wr += poly_ctxt.size_in_bytes

    return stats


def partial_multiply_inner(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    sqr: bool = False,
):
    """
    Compute (a0b1 + a1b0, b0b1) from (a0, a1) and (b0, b1)
    input rd: handled by caller, output wr: handled by caller
    """

    if sqr:
        poly.mult(poly_ctxt, arch_params)  ## a * b
        poly.add(poly_ctxt, arch_params)  ## 2 * a * b
        poly.mult(poly_ctxt, arch_params)  ## b * b
        return PerfCounter()

    if arch_params.karatsuba:
        poly.mult(poly_ctxt, arch_params)  ## b_0 * b_1

        poly.add(poly_ctxt, arch_params)  ## a_0 + b_0
        poly.add(poly_ctxt, arch_params)  ## a_1 + b_1
        ## (a_0 + b_0)(a_1 + b_1) = a_0a_1 + a_0b_1 + a_1b_0 + b_0b_1
        poly.mult(poly_ctxt, arch_params)

        poly.add(poly_ctxt, arch_params)  ## (a_0 + b_0)(a_1 + b_1) - a_0a_1
        poly.add(poly_ctxt, arch_params)  ## (a_0 + b_0)(a_1 + b_1) - b_0b_1
    else:
        poly.mult(poly_ctxt, arch_params)  ## b_0 * b_1
        poly.mult(poly_ctxt, arch_params)  ## a_0 * b_1
        poly.mult(poly_ctxt, arch_params)  ## a_1 * b_0

        poly.add(poly_ctxt, arch_params)  ## a_0 * b_1 + a_1 * b_0

    return PerfCounter()


def multiply(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    sqr: bool = False,
    rd_in: bool = False,
    wr_out: bool = False,
    key_cached: bool = False,
):
    """
    Computes x*y from input x and y.
    input rd: handled here, output wr: handled by caller
    """

    stats = PerfCounter()

    key_ctxt = poly_ctxt.key_switch_context()
    limb_rdwr = arch_params.cache_style < CacheStyle.CONST
    reorder_rdwr = limb_rdwr or not arch_params.mod_down_reorder
    sq_mult = 1 if sqr else 2

    if rd_in:
        ## read in a0 and a1 term
        stats.arch.dram_limb_rd += sq_mult * poly_ctxt.size_in_bytes
    poly.mult(poly_ctxt, arch_params)  ## a0a1
    if limb_rdwr:
        ## write out all limbs of a0a1
        stats.arch.dram_limb_wr += poly_ctxt.size_in_bytes

    key_switch_hoisting(poly_ctxt, key_ctxt, arch_params, rd_in=limb_rdwr, wr_out=True)
    key_switch_inner_product(
        poly_ctxt,
        key_ctxt,
        arch_params,
        automorph=False,
        rd_in=True,
        wr_out=limb_rdwr,
        key_cached=key_cached,
    )

    if arch_params.rescale_fusion:
        ## read in a0 and a1 and b0 and b1 for the same limb and
        ## compute key_switch(a0a1) + P * (a0b1 + a1b0, b0b1)
        if rd_in:
            stats.arch.dram_limb_rd += 2 * sq_mult * poly_ctxt.size_in_bytes
        partial_multiply_inner(poly_ctxt, arch_params, sqr)
        multiply_plain(poly_ctxt, arch_params)

        ## add to inner_product result
        if limb_rdwr:
            ## read the inner_product output limbwise (only first poly_ctxt.limbs)
            stats.arch.dram_limb_rd += 2 * poly_ctxt.size_in_bytes
        add(poly_ctxt, arch_params, wr_out=reorder_rdwr)

        ## scale down the resulting sum
        poly_ctxt = poly_ctxt.drop()  ## combine with rescale
        mod_down(key_ctxt, poly_ctxt, arch_params, rd_in=reorder_rdwr, wr_out=wr_out)
        mod_down(key_ctxt, poly_ctxt, arch_params, rd_in=reorder_rdwr, wr_out=wr_out)
    else:
        ## force inner_product write out if forwarded in wrong order
        ## TODO: large cache check
        if not limb_rdwr and reorder_rdwr:
            stats.arch.dram_limb_wr = 2 * key_ctxt.size_in_bytes

        # scale down the inner_product output
        # fmt: off
        mod_down(key_ctxt, poly_ctxt, arch_params, rd_in=reorder_rdwr, wr_out=limb_rdwr)
        mod_down(key_ctxt, poly_ctxt, arch_params, rd_in=reorder_rdwr, wr_out=limb_rdwr)
        # fmt: on

        ## read in a0 and a1 and b0 and b1 for the same limb and
        ## compute (a0b1 + a1b0, b0b1)
        if rd_in:
            stats.arch.dram_limb_rd += 2 * sq_mult * poly_ctxt.size_in_bytes
        partial_multiply_inner(poly_ctxt, arch_params, sqr)
        if limb_rdwr:
            stats.arch.dram_limb_rd += 2 * poly_ctxt.size_in_bytes
        add(poly_ctxt, arch_params, wr_out=reorder_rdwr)

        mod_reduce_rescale(poly_ctxt, arch_params, rd_in=reorder_rdwr, wr_out=wr_out)
        mod_reduce_rescale(poly_ctxt, arch_params, rd_in=reorder_rdwr, wr_out=wr_out)

    return stats


def double_multiply(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    sqr: bool = False,
    rd_in: bool = True,
    wr_out: bool = True,
    key_cached: bool = False,
):
    """Computes 2*x*y from input x and y"""
    multiply(poly_ctxt, arch_params, sqr=sqr, rd_in=rd_in, wr_out=False, key_cached=key_cached)
    red_ctxt = poly_ctxt.drop()
    add(red_ctxt, arch_params, rd_in=False, wr_out=wr_out)
    return PerfCounter()


def double_square(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    rd_in=True,
    wr_out=True,
):
    """Computes 2x^2 from input x"""
    multiply(poly_ctxt, arch_params, sqr=True, rd_in=rd_in, wr_out=False)
    red_ctxt = poly_ctxt.drop()
    add(red_ctxt, arch_params, rd_in=False, wr_out=wr_out)
    return PerfCounter()
