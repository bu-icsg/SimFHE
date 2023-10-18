from math import ceil, log2

import params
from params import CacheStyle
import poly
import evaluator
import bootstrap
from perf_counter import PerfCounter


def rotate_sum(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    n_rot: int,
    n_iter: int = 1,
    rescale: bool = False,
):
    depth = int(log2(n_rot))
    baseline = depth // n_iter
    res = [baseline] * n_iter
    for idx in range(depth - sum(res)):
        res[idx] += 1
    rot_list = [pow(2, val) for val in res]

    stats = PerfCounter()

    key_ctxt = poly_ctxt.key_switch_context()
    limb_rdwr = arch_params.cache_style < CacheStyle.CONST
    reorder_rdwr = limb_rdwr or not arch_params.mod_down_reorder

    for rot_per_iter in rot_list:
        evaluator.key_switch_hoisting(poly_ctxt, key_ctxt, arch_params, rd_in=True)
        for idx in range(rot_per_iter - 1):
            rd_a = arch_params.cache_style < CacheStyle.BETA or idx == 0
            # fmt: off
            evaluator.key_switch_inner_product(
                poly_ctxt, key_ctxt, arch_params, automorph=True, rd_in=rd_a, wr_out=limb_rdwr
            )
            # fmt: on

            ## TODO: check if idx == 0 is necessary with large cache
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

        ## sum the values
        for idx in range(rot_per_iter):
            if limb_rdwr:
                stats.arch.dram_limb_rd += 2 * key_ctxt.size_in_bytes
            if idx != 0:
                evaluator.add(key_ctxt, arch_params)
        if limb_rdwr:
            stats.arch.dram_limb_wr += 2 * key_ctxt.size_in_bytes

        ## scale down the resulting sum
        if rescale:
            poly_ctxt = poly_ctxt.drop()
        # fmt: off
        evaluator.mod_down(key_ctxt, poly_ctxt, arch_params, rd_in=reorder_rdwr, wr_out=True)
        evaluator.mod_down(key_ctxt, poly_ctxt, arch_params, rd_in=reorder_rdwr, wr_out=True)
        # fmt: on

    return stats


def inner_product(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    dim: int,
):
    evaluator.multiply(poly_ctxt, arch_params, rd_in=True, wr_out=True)
    poly_ctxt.drop()
    rotate_sum(poly_ctxt, arch_params, dim, 4)  # left rotate to accumulate
    evaluator.multiply_plain(poly_ctxt, arch_params, wr_out=True)  # puncture
    rotate_sum(poly_ctxt, arch_params, dim, 4, rescale=True)  # right rotate to repeat
    return PerfCounter()


def sigmoid_product(
    poly_ctxt: params.PolyContext, arch_params: params.ArchParam,
):
    """Compute (ay^3+by+c)*x with a 2-level circuit"""
    x_ctxt = poly_ctxt.basis_convert(poly_ctxt.limbs + 1)
    lvl1_ctxt = poly_ctxt.drop()
    lvl2_ctxt = lvl1_ctxt.drop()

    # Compute y^2
    evaluator.multiply(poly_ctxt, arch_params, rd_in=True, wr_out=True)
    # Compute y^2 + b/a. Combine this addition with write-out on the previous step
    evaluator.add_plain(lvl1_ctxt, arch_params)
    # Compute x*c*delta^3 and x*a*delta
    evaluator.multiply_plain(x_ctxt, arch_params, rd_in=True)
    evaluator.multiply_plain(x_ctxt, arch_params)
    # Compute x*a
    evaluator.mod_reduce_rescale(x_ctxt, arch_params, wr_out=True)
    # Compute x*c
    evaluator.mod_down_reduce(x_ctxt, lvl2_ctxt, arch_params)
    # Compute x*y*a
    evaluator.multiply(poly_ctxt, arch_params, rd_in=True, wr_out=True)
    # Compute (ay^2+by)*x
    evaluator.multiply(lvl1_ctxt, arch_params, rd_in=True, wr_out=True)
    # Compute (ay^2+by+c)*x
    evaluator.add(lvl2_ctxt, arch_params, rd_in=True, wr_out=True)

    return PerfCounter()


def nesterov_update(
    poly_ctxt: params.PolyContext, arch_params: params.ArchParam,
):
    delta_ctxt = poly_ctxt.drop(4)
    stats = PerfCounter()
    # drop v and w down to delta number of limbs
    ## TODO: check if these reads and writes are necessary with large cache
    stats.arch.dram_limb_rd = 4 * poly_ctxt.size_in_bytes
    evaluator.mod_down_reduce(poly_ctxt, delta_ctxt, arch_params)
    evaluator.mod_down_reduce(poly_ctxt, delta_ctxt, arch_params)
    stats.arch.dram_limb_rd = 2 * delta_ctxt.size_in_bytes
    # compute w_nxt
    stats.arch.dram_limb_rd = 2 * delta_ctxt.size_in_bytes
    evaluator.add(delta_ctxt, arch_params, rd_in=True)
    # compute v_nxt
    evaluator.multiply_plain(delta_ctxt, arch_params)
    evaluator.multiply_plain(delta_ctxt, arch_params)
    evaluator.add(delta_ctxt, arch_params, wr_out=True)
    # drop a limb from v_nxt and w_nxt
    evaluator.mod_reduce_rescale(delta_ctxt, arch_params, rd_in=True, wr_out=True)
    evaluator.mod_reduce_rescale(delta_ctxt, arch_params, rd_in=True, wr_out=True)
    return stats


def iteration(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    n_features: int = 196,
    batch_sz: int = 1024,
):
    n_eff_features = pow(2, int(ceil(log2(n_features + 1))))
    samples_per_ct = poly_ctxt.N // (2 * n_eff_features)
    n_ct = batch_sz // samples_per_ct

    for _ in range(n_ct):
        inner_product(poly_ctxt, arch_params, n_eff_features)
        sigmoid_product(poly_ctxt, arch_params)
    delta_ctxt = poly_ctxt.drop(4)
    rotate_sum(delta_ctxt, arch_params, samples_per_ct, 4)
    nesterov_update(poly_ctxt, arch_params)
    poly_ctxt.drop(5)
    return PerfCounter()


def logistic_regression(
    poly_ctxt: params.PolyContext,
    arch_params: params.ArchParam,
    n_iter: int = 3,
    n_features: int = 196,
    batch_sz: int = 1024,
):
    for _ in range(n_iter):
        iteration(poly_ctxt, arch_params, n_features, batch_sz)
        poly_ctxt = poly_ctxt.drop(5)
    return PerfCounter()


def bootstrap_regression(
    scheme_params: params.SchemeParams,
    n_iter: int = 3,
    n_features: int = 196,
    batch_sz: int = 1024,
):
    bootstrap.bootstrap(scheme_params)
    fresh_ctxt = scheme_params.fresh_ctxt
    logistic_regression(
        fresh_ctxt, scheme_params.arch_param, n_iter, n_features, batch_sz
    )
    return PerfCounter()
