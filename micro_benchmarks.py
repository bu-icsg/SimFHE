from perf_counter import PerfCounter
import params
import evaluator


def mod_up(poly_ctxt: params.PolyContext, scheme_params: params.SchemeParams):
    digit_ctxt = poly_ctxt.basis_convert(poly_ctxt.alpha)
    key_ctxt = poly_ctxt.key_switch_context()
    evaluator.mod_raise(digit_ctxt, key_ctxt, scheme_params.arch_param, True, True)
    return PerfCounter()


def mod_down(poly_ctxt: params.PolyContext, scheme_params: params.SchemeParams):
    key_ctxt = poly_ctxt.key_switch_context()
    evaluator.mod_down(key_ctxt, poly_ctxt, scheme_params.arch_param, True, True)
    return PerfCounter()


def decomp(poly_ctxt: params.PolyContext):
    stats = PerfCounter()
    ## TODO: large cache check
    stats.arch.dram_limb_rd += poly_ctxt.size_in_bytes
    evaluator.decompose(poly_ctxt)
    stats.arch.dram_limb_wr += poly_ctxt.size_in_bytes
    return stats


def inner_product(poly_ctxt: params.PolyContext, scheme_params: params.SchemeParams):
    key_ctxt = poly_ctxt.key_switch_context()
    evaluator.key_switch_inner_product(
        poly_ctxt, key_ctxt, scheme_params.arch_param, True, True
    )
    return PerfCounter()


def automorph(poly_ctxt: params.PolyContext, scheme_params: params.SchemeParams):
    evaluator.rotate_inner(poly_ctxt, scheme_params.arch_param, True, True)
    return PerfCounter()


def pt_add(poly_ctxt: params.PolyContext, scheme_params: params.SchemeParams):
    evaluator.add_plain(poly_ctxt, scheme_params.arch_param, True, True)
    return PerfCounter()


def add(poly_ctxt: params.PolyContext, scheme_params: params.SchemeParams):
    evaluator.add(poly_ctxt, scheme_params.arch_param, True, True)
    return PerfCounter()


def pt_mult(poly_ctxt: params.PolyContext, scheme_params: params.SchemeParams):
    evaluator.multiply_plain(poly_ctxt, scheme_params.arch_param, True, True)
    evaluator.mod_reduce_rescale(poly_ctxt, scheme_params.arch_param, True, True)
    evaluator.mod_reduce_rescale(poly_ctxt, scheme_params.arch_param, True, True)
    return PerfCounter()


def mult(poly_ctxt: params.PolyContext, scheme_params: params.SchemeParams):
    evaluator.multiply(poly_ctxt, scheme_params.arch_param, False, True, True)
    return PerfCounter()


def rotate(poly_ctxt: params.PolyContext, scheme_params: params.SchemeParams):
    evaluator.rotate(poly_ctxt, scheme_params.arch_param, True, True)
    return PerfCounter()


def hoisted_rotate(
    poly_ctxt: params.PolyContext, scheme_params: params.SchemeParams, n_rot: int = 8
):
    stats = PerfCounter()
    arch_params = scheme_params.arch_param
    key_ctxt = poly_ctxt.key_switch_context()
    evaluator.key_switch_hoisting(poly_ctxt, key_ctxt, arch_params)
    for _ in range(n_rot):  ## no need to count the rotation by zero
        rd_in = arch_params.cache_style < params.CacheStyle.BETA
        evaluator.rotate_digits(poly_ctxt, key_ctxt, arch_params, rd_in=rd_in)
    return stats
