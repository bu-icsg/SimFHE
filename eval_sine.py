from math import ceil, log2
import logging
from urllib.request import CacheFTPHandler

import params
import evaluator
from perf_counter import PerfCounter


def baby_step_basis(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    degree: int,
):
    stats = PerfCounter()

    m = int(ceil(log2(degree + 1)))
    ell = m // 2

    max_baby_step_index = pow(2, ell)

    index_reached = 1
    current_level = 1

    ## initial read performed at bootstrapping level
    key_cached = arch_params.cache_style >= params.CacheStyle.HUGE

    while index_reached < max_baby_step_index:
        reachable_index = min(pow(2, current_level), max_baby_step_index)

        ## from 2^i +1 to 2^{i+1}
        ## perform all multiplications
        for i in range(index_reached, reachable_index):
            ## perform all multiplications for the new ciphertexts
            current_index = i + 1
            if current_index % 2 == 0:
                evaluator.double_multiply(poly_ctxt, arch_params, sqr=True, key_cached=key_cached)
            else:
                evaluator.double_multiply(poly_ctxt, arch_params, key_cached=key_cached)
            

        ## memory is empty

        red_ctxt = poly_ctxt.drop()

        ## rescale all previous ciphertexts
        for _ in range(index_reached):
            ## TODO: large cache check
            stats.arch.dram_limb_rd += 2 * poly_ctxt.size_in_bytes
            evaluator.multiply_plain(poly_ctxt, arch_params)
            evaluator.mod_reduce_rescale(poly_ctxt, arch_params)
            stats.arch.dram_limb_wr += 2 * red_ctxt.size_in_bytes

        ## memory is empty

        ## from 2^i +1 to 2^{i+1}
        ## perform subtraction
        for i in range(index_reached, reachable_index):
            """
            Don't need to read in the ciphertext being subtracted away,
            since this ciphertext is always T_1. We can perform these
            subtractions while the limbs of T_1 are in memory from
            the rescaling loop above.
            """
            ## TODO: large cache check
            stats.arch.dram_limb_rd += 2 * red_ctxt.size_in_bytes
            current_index = i + 1
            if current_index % 2:
                evaluator.add_plain(red_ctxt, arch_params)
            else:
                evaluator.add(red_ctxt, arch_params)

            ## write out result
            ## TODO: large cache check
            stats.arch.dram_limb_wr += 2 * red_ctxt.size_in_bytes

        index_reached = reachable_index
        current_level += 1
        poly_ctxt = red_ctxt

    logging.debug("baby step computation result limbs", poly_ctxt.limbs)

    return stats


def giant_step_basis(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    degree: int,
):
    stats = PerfCounter()

    m = int(ceil(log2(degree + 1)))
    ell = m // 2

    ## initial read handled at bootstrapping level
    key_cached = arch_params.cache_style >= params.CacheStyle.HUGE

    ## start from 2^ell
    ## drop limb to account for constant multiplication
    giant_step_ctxt = poly_ctxt.drop()
    curr_index = pow(2, ell)
    while curr_index < pow(2, m - 1):
        ## square and double
        evaluator.double_multiply(giant_step_ctxt, arch_params, sqr=True, key_cached=key_cached)
        giant_step_ctxt = giant_step_ctxt.drop()
        ## no need to read in for add because we can just add before
        ## the write in the double_square
        evaluator.add_plain(giant_step_ctxt, arch_params)
        curr_index *= 2

    return stats


def leaf_prods(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    degree: int,
):
    stats = PerfCounter()

    m = int(ceil(log2(degree + 1)))
    ell = m // 2

    logging.debug("leaf multiplication input limbs", poly_ctxt.limbs)
    num_giant_steps = m - ell  ## [ell, m-1] inclusive
    num_leaves = pow(2, num_giant_steps)

    ## poly_ctxt is the context of the starting baby steps
    ## begin by multiplying the baby step polynomials by the Chebyshev constants
    num_baby_steps = pow(2, ell)
    logging.debug("num leaves, num baby steps", num_leaves, num_baby_steps)
    for _ in range(num_leaves):
        for i in range(num_baby_steps):
            ## reading in inner product input
            ## TODO: try to reuse this across all limbs
            ## TODO: large cache check
            stats.arch.dram_limb_rd += 2 * poly_ctxt.size_in_bytes
            # pass
            evaluator.multiply_plain(poly_ctxt, arch_params)
            if i == 0:
                evaluator.add_plain(poly_ctxt, arch_params)
            else:
                evaluator.add(poly_ctxt, arch_params)
            ## no need to write out result
            ## we are computing the last limb first so that we
            ## can immediately begin the rescale once that limb
            ## is computed. We then finish the rescale operation on
            ## all subsequent completed limbs.
        evaluator.mod_reduce_rescale(poly_ctxt, arch_params)
        red_ctxt = poly_ctxt.drop()
        ## write out result
        ## TODO: large cache check
        stats.arch.dram_limb_wr += 2 * red_ctxt.size_in_bytes
    poly_ctxt = poly_ctxt.drop()

    logging.debug("leaf multiplication result limbs", poly_ctxt.limbs)

    return stats


def tree_up(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    degree: int,
):
    stats = PerfCounter()

    m = int(ceil(log2(degree + 1)))
    ell = m // 2

    num_giant_steps = m - ell  ## [ell, m-1] inclusive

    ## initial key read handled at bootstrapping level
    key_cached = arch_params.cache_style >= params.CacheStyle.HUGE

    for i in range(num_giant_steps - 1, -1, -1):
        num_nodes = pow(2, i)
        for _ in range(num_nodes):
            """
            Everything except the alpha+1 inner product for the mod reduction limb generation
            is over the same limb index.
            This means that we don't need any writes in the multiplication until the end
            of this loop.
            The only intermediate reads and writes would be if we don't have alpha-caching, which
            would require us to write out the L-1 correction terms to DRAM in coefficient.
            Refer to image...
            """
            evaluator.multiply(poly_ctxt, arch_params, wr_out=False, key_cached=key_cached)
            red_ctxt = poly_ctxt.drop()

            ## rescale addition operand
            ## TODO: large cache check
            stats.arch.dram_limb_rd += 2 * poly_ctxt.size_in_bytes
            evaluator.multiply_plain(poly_ctxt, arch_params)
            evaluator.mod_reduce_rescale(poly_ctxt, arch_params)

            evaluator.add(red_ctxt, arch_params)

            ## TODO: large cache check
            stats.arch.dram_limb_wr += 2 * red_ctxt.size_in_bytes

        poly_ctxt = poly_ctxt.drop()

    return stats


def r_square(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    r: int,
):
    stats = PerfCounter()

    ## initial read handled at the bootstrapping level
    key_cache = arch_params.cache_style >= params.CacheStyle.HUGE

    ## now do the r squaring

    for _ in range(r):
        evaluator.double_multiply(poly_ctxt, arch_params, sqr=True, key_cached=key_cache)
        ## do this add before the final write in double_square
        evaluator.add_plain(poly_ctxt, arch_params)
        poly_ctxt = poly_ctxt.drop()

    logging.debug("eval_sine result limbs", poly_ctxt.limbs)

    return stats


def eval_sine(
    poly_ctxt: params.PolyContext,
    scheme_params: params.SchemeParams,
):
    """
    The memory of this function is self-contained. Assumes that memory is empty at the
    start and finishes with empty memory.
    """
    stats = PerfCounter()

    d = scheme_params.poly_degree
    m = int(ceil(log2(d + 1)))
    ell = m // 2

    start_limbs = poly_ctxt.limbs
    logging.debug("d, m, ell", d, m, ell)
    logging.debug("eval_sine start limbs", start_limbs)

    ## Compute all T_i for i in [0, 2^{ell-1}]
    ## and i = 2^ell, 2^{ell+1}, .... , 2^{m-1}

    ## Two phases. In the first phase, compute all
    ## indices in the set [0, 2^{ell}] depth-optimally
    ## In the second phase square 2^ell until the rest
    ## of the indices are computed

    ## Chebyshev addition operation is
    ## T_{m+n} = 2 * T_m * T_n - T_{|m-n|}

    """
    Handling different scaling factors.

    Addition with different scaling factors. 
    
    Refer to image...
    """

    baby_step_basis(poly_ctxt, scheme_params.arch_param, scheme_params.poly_degree)

    max_baby_step_index = pow(2, ell)

    index_reached = 1
    current_level = 1

    while index_reached < max_baby_step_index:
        reachable_index = min(pow(2, current_level), max_baby_step_index)

        red_ctxt = poly_ctxt.drop()

        index_reached = reachable_index
        current_level += 1
        poly_ctxt = red_ctxt

    # logging.debug("baby step computation result limbs", poly_ctxt.limbs)

    giant_step_basis(poly_ctxt, scheme_params.arch_param, scheme_params.poly_degree)

    leaf_prods(poly_ctxt, scheme_params.arch_param, scheme_params.poly_degree)

    ## now we compute the bsgs tree
    ## every giant step must be used
    ## depth equals the number of giant steps
    # logging.debug("leaf multiplication input limbs", poly_ctxt.limbs)
    num_giant_steps = m - ell  ## [ell, m-1] inclusive

    # ## poly_ctxt is the context of the starting baby steps
    # ## begin by multiplying the baby step polynomials by the Chebyshev constants
    poly_ctxt = poly_ctxt.drop()

    # logging.debug("leaf multiplication result limbs", poly_ctxt.limbs)

    ## memory is empty

    tree_up(poly_ctxt, scheme_params.arch_param, scheme_params.poly_degree)

    ## now we multiply by the giant steps
    ## the first multiplication starts with the current poly_ctxt
    ## each multiplication then simply removes one level.
    ## the number of multiplications is just the number of leaves at
    ## the next level of the tree
    for _ in range(num_giant_steps - 1, -1, -1):
        poly_ctxt = poly_ctxt.drop()

    # ## now do the r squaring

    r_square(poly_ctxt, scheme_params.arch_param, scheme_params.r)

    for _ in range(scheme_params.r):
        #     evaluator.double_multiply(poly_ctxt, scheme_params.arch_param, sqr=True)
        #     ## do this add before the final write in double_square
        #     evaluator.add_plain(poly_ctxt, scheme_params.arch_param)
        poly_ctxt = poly_ctxt.drop()

    # logging.debug("eval_sine result limbs", poly_ctxt.limbs)
    # logging.debug(start_limbs - scheme_params.eval_sine_limbs, poly_ctxt.limbs)
    assert start_limbs - scheme_params.eval_sine_limbs == poly_ctxt.limbs

    return stats
