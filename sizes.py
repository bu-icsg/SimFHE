"""
Compute the sizes of various elements
"""

from numpy import poly
from evaluator import key_switch
from params import SchemeParams, ArchParam, CacheStyle, FFTStyle, PolyContext

logq = 50
logN = 17
dnum = 2
limbs = 40
fft_iters = 6

BEST_ARCH_PARAMS = ArchParam(
    karatsuba=True,
    key_compression=True,
    rescale_fusion=True,
    cache_style=CacheStyle.ALPHA,
    mod_down_reorder=True,
)


params = SchemeParams(
    logq=logq, 
    logN=logN, 
    # override_Q0_limbs=0, 
    dnum=dnum, 
    fft_iters=fft_iters,
    fft_style=FFTStyle.UNROLLED_HOISTED,
    arch_param=BEST_ARCH_PARAMS,
)

print("Q0 limbs", params.bootstrapping_Q0_limbs)
key_ctxt = params.cts_ctxt.key_switch_context()
key_switch_size = params.arch_param.key_sz(key_ctxt.size_in_bytes)
print("key switch key size", key_switch_size/1000000, "MB")
