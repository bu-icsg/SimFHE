import copy
from dataclasses import dataclass, field
import dataclasses
from functools import total_ordering
from math import ceil, log2
from enum import Enum
from multiprocessing.sharedctypes import Value

@dataclass
class LimbsContext:
    limbs : int

class ConfigParam:
    def __init__(self,**kwargs):
        self.N = 2**16
        self.limbs = 47
        self.exp_img_ctxt = LimbsContext(limbs=32)
        self.slot_coeff_ctxt = LimbsContext(limbs=19)
        for k,v in kwargs.items():
            setattr(self,k,v)


@total_ordering
class CacheStyle(Enum):
    NONE = 0
    CONST = 1
    BETA = 2
    ALPHA = 3
    # ALPHA_ROT = 4
    HUGE = 5  ## roughly 250 MB. Big enough to cache key-switch keys

    def __lt__(self, other):
        if self.__class__ == other.__class__:
            return self.value < other.value
        return NotImplemented


class FFTStyle(Enum):
    BSGS = 1
    BSGS_HOISTED = 2
    UNROLLED_HOISTED = 3


@dataclass
class PolyContext:
    logq: int
    logN: int
    dnum: int
    limbs: int

    @property
    def N(self):
        return 1 << self.logN

    @property
    def alpha(self):
        return int(ceil(self.limbs / self.dnum))

    @property
    def size_in_bytes(self):
        return self.N * self.limbs * self.logq / 8

    def _copy(self, new_limbs: int):
        return PolyContext(self.logq, self.logN, self.dnum, new_limbs)

    def basis_convert(self, new_limbs):
        return self._copy(new_limbs)

    def key_switch_context(self, new_limbs=-1):
        if new_limbs == -1:
            new_limbs = self.alpha + 1
        return self._copy(self.limbs + new_limbs)

    def __lt__(self, other):
        assert isinstance(other, PolyContext)
        return self.limbs < other.limbs

    def drop(self, drop_limbs: int = 1):
        return self._copy(self.limbs - drop_limbs)

    def min_ctxt(c1, c2):
        if c1 < c2:
            return c1
        else:
            return c2


@dataclass
class ArchParam:
    ## each set has funits number of functional units
    ## functional units operate over single integers
    sets: int = 8
    funits: int = 32  ## number of modular multiplication and addition units.
    ## cycles needed for each operation
    add_lat: int = 4
    mult_lat: int = 16
    ntt_lat: int = 16
    auto_lat: int = 2

    # karatsuba multiplication
    karatsuba: bool = False

    ## compress the pseudo random component of the key
    key_compression: bool = False

    # fuse dropping a limb after pt-mult or mult with a preceding mod-reduce
    rescale_fusion: bool = False

    ## pre mod-down re-ordering and fusion
    mod_down_reorder: bool = False

    cache_style: CacheStyle = CacheStyle.CONST

    # fuse the mod_up in rotate and mult with preceding operation
    # fuse key_switch_inner_product output computation with subsequent computation

    ## cache beta limbs to perform the inner product in
    ## memory for each rotation

    ## cache alpha limbs for the modulus raising.
    ## no need to write out limbs to read them in a different order.
    ## all new limbs can be created without any further reads

    def key_sz(self, sz):
        return sz if self.key_compression else 2 * sz

    def key_sz(self, sz):
        return sz if self.key_compression else 2 * sz


@dataclass
class SchemeParams:
    """
    Static top-level parameters.
    Contains starting parameters for polynomials and bootstrapping parameters
    """

    ## Base parameters
    logq: int = 50  ## size of one limb
    logN: int = 17  ## poly degree
    override_Q0_limbs: int = 0  ## 0 indicates calculate, else use fixed val
    dnum: int = 2
    fft_iters: int = 6  ## number of iterations of the fft inner loop

    ## number of limbs we want to add after bootstrapping is finished
    # fresh_limbs: int = 19

    ## Modular reduction parameters
    poly_degree: int = 63
    r: int = 2  ## number of squarings after poly eval

    ## Homomorphic FFT style
    # fft_style: FFTStyle = FFTStyle.BASELINE
    # fft_style: FFTStyle = FFTStyle.HOISTED
    fft_style: FFTStyle = FFTStyle.UNROLLED_HOISTED
    # fft_style: FFTStyle = FFTStyle.BSGS

    arch_param: ArchParam = field(default_factory=ArchParam)

    @property
    def bsgs_iters(self):
        # res = [5, 5, 5]
        # res = [5, 5, 6]
        n_iter = self.fft_iters
        baseline = (self.logN - 1) // n_iter
        diff = (self.logN - 1) - baseline * n_iter
        res = [baseline] * n_iter
        for i in range(diff):
            res[-1 - i] += 1
        assert sum(res) == self.logN - 1
        return res  ## need this to add up to logN-1

    @property
    def N(self):
        return 1 << self.logN

    @property
    def eval_sine_limbs(self):
        """
        number of limbs needed to evaluate approximate modular reduction circuit
        """
        return int(ceil(log2(self.poly_degree))) + self.r + 1

    @property
    def he_fft_limbs(self):
        """
        number of limbs needed to evaluate homomorphic FFT or homomorphic inverse FFT
        """
        if self.fft_style == FFTStyle.UNROLLED_HOISTED:
            return self.fft_iters
        elif self.fft_style == FFTStyle.BSGS or self.fft_style == FFTStyle.BSGS_HOISTED:
            return len(self.bsgs_iters)
        else:
            raise ValueError("unknown FFT style")

    @property
    def bootstrapping_Q0_limbs(self):
        """
        computes the necessary number of limbs needed at the start of bootstrapping
        start limbs is the number of limbs in the input ciphertext
        fresh limbs is the number of limbs we want to add after bootstrapping is finished
        """
        if not self.override_Q0_limbs:
            return 2 * self.he_fft_limbs + self.eval_sine_limbs + self.fresh_limbs
        else:
            return self.override_Q0_limbs

    @property
    def mod_raise_ctxt(self):
        num_limbs = self.bootstrapping_Q0_limbs
        return PolyContext(self.logq, self.logN, self.dnum, num_limbs)

    @property
    def cts_ctxt(self):
        num_limbs = self.bootstrapping_Q0_limbs - self.he_fft_limbs
        return PolyContext(self.logq, self.logN, self.dnum, num_limbs)

    @property
    def eval_sine_ctxt(self):
        num_limbs = (
            self.bootstrapping_Q0_limbs - self.he_fft_limbs - self.eval_sine_limbs
        )
        return PolyContext(self.logq, self.logN, self.dnum, num_limbs)

    @property
    def fresh_ctxt(self):
        num_limbs = self.fresh_limbs
        return PolyContext(self.logq, self.logN, self.dnum, num_limbs)

    @property
    def limbs_PQ(self):
        poly_ctxt = PolyContext(
            self.logq, self.logN, self.dnum, self.bootstrapping_Q0_limbs
        )
        key_switch_ctxt = poly_ctxt.key_switch_context()
        return key_switch_ctxt.limbs

    @property
    def max_limbs(self):
        # logNtoMaxLogQ = {16: 1900, 17: 3800}
        logNtoMaxLogQ = {16: 1533, 17: 3069}  ## secret key density 192
        return logNtoMaxLogQ[self.logN] // self.logq

    @property
    def fresh_limbs(self):
        if not self.override_Q0_limbs:
            key_switch_limbs = self.max_limbs  ## = L + ceil(L/dnum) + 1
            L = ((key_switch_limbs - 1) * self.dnum) // (self.dnum + 1)
        else:
            L = self.override_Q0_limbs
        poly_ctxt = PolyContext(self.logq, self.logN, self.dnum, L)
        key_switch_ctxt = poly_ctxt.key_switch_context()
        # print(
        #     L,
        #     key_switch_ctxt.limbs,
        #     key_switch_limbs,
        #     self.dnum,
        #     poly_ctxt.dnum,
        # )
        if not self.override_Q0_limbs:
            assert key_switch_ctxt.limbs <= key_switch_limbs
        fL = L - 2 * self.he_fft_limbs - self.eval_sine_limbs
        if fL <= 0:
            print("L", L)
            if self.override_Q0_limbs:
                print("Q0 limbs have been overwritten.", self.override_Q0_limbs)
            print("max limbs", self.max_limbs)
            raise ValueError("not enough fresh limbs")
        return fL

    def __str__(self):
        return (
            f"({self.logN}, {self.dnum}, {self.fft_iters}, {self.fft_style}, "
            + f"{self.arch_param.cache_style}, "
            + f"{self.bootstrapping_Q0_limbs}, {self.limbs_PQ})"
            + f" -> {self.fresh_limbs}"
        )

    def get_max_cache_size(self, constant=1):
        """
        Returns the maximum size of the cache in bytes
        """
        cache_type = self.arch_param.cache_style
        num_limbs = 1  ## number of limbs that can fit in the cache
        if cache_type == CacheStyle.NONE:
            num_limbs = 0
        elif cache_type == CacheStyle.CONST:
            assert False
            num_limbs = (
                4  ## TODO: this is arbitrary. update with something more justifiable
            )
        elif cache_type == CacheStyle.BETA:
            num_limbs = self.dnum
        elif cache_type == CacheStyle.ALPHA:
            num_limbs = ceil((self.bootstrapping_Q0_limbs + 1) / self.dnum)
        elif cache_type == CacheStyle.HUGE:
            return 250000000
        else:
            raise ValueError("unknown cache style")

        num_limbs *= constant

        return int(ceil(num_limbs * self.N * self.logq / 8))


GPU_ARCH_PARAMS = ArchParam(
    karatsuba=True,
    key_compression=False,
    rescale_fusion=False,
    cache_style=CacheStyle.NONE,
    # cache_style=CacheStyle.BETA,
    mod_down_reorder=False,
)


GPU_PARAMS = SchemeParams(
    # logq=50,
    ## This is to model the inefficiency in the word transfer of the GPU.
    ## Even though their limbs are 50 bits, they transfer 64 bits.
    # logq=38.5,
    logq=64,
    logN=17,
    # override_Q0_limbs=35,
    # override_Q0_limbs=44,
    override_Q0_limbs=38,
    dnum=3,
    poly_degree=31,
    r=3,
    fft_iters=3,
    fft_style=FFTStyle.BSGS,
    arch_param=GPU_ARCH_PARAMS
    # arch_param=ArchParam(
    #     karatsuba=True,
    #     key_compression=False,
    #     rescale_fusion=False,
    #     cache_style=CacheStyle.NONE,
    #     mod_down_reorder=False,
    # ),
)

Mem_benchmark_O_1_cache = dataclasses.replace(GPU_PARAMS)
Mem_benchmark_O_1_cache.arch_param = dataclasses.replace(GPU_PARAMS.arch_param)
Mem_benchmark_O_1_cache.arch_param.cache_style = CacheStyle.CONST

Mem_benchmark_beta_cache = dataclasses.replace(GPU_PARAMS)
Mem_benchmark_beta_cache.arch_param = dataclasses.replace(GPU_PARAMS.arch_param)
Mem_benchmark_beta_cache.arch_param.cache_style = CacheStyle.BETA

Mem_benchmark_alpha_cache = dataclasses.replace(GPU_PARAMS)
Mem_benchmark_alpha_cache.arch_param = dataclasses.replace(GPU_PARAMS.arch_param)
Mem_benchmark_alpha_cache.arch_param.cache_style = CacheStyle.ALPHA

Mem_benchmark_reorder = dataclasses.replace(GPU_PARAMS)
Mem_benchmark_reorder.arch_param = dataclasses.replace(GPU_PARAMS.arch_param)
Mem_benchmark_reorder.arch_param.cache_style = CacheStyle.ALPHA
Mem_benchmark_reorder.arch_param.mod_down_reorder = True

LATTIGO_PARAMS = SchemeParams(
    logq=30,
    logN=16,
    override_Q0_limbs=31,
    dnum=5,
    poly_degree=63,
    r=2,
    fft_iters=4,
    fft_style=FFTStyle.BSGS_HOISTED,
    arch_param=ArchParam(
        karatsuba=True,
        key_compression=False,
        rescale_fusion=False,
        cache_style=CacheStyle.NONE,
        mod_down_reorder=False,
    ),
)


# def get_params():
#     num_sets = 7
#     params = [copy.deepcopy(GPU_PARAMS) for _ in range(num_sets)]
#     for idx in range(1, num_sets):
#         params[idx].arch_param.key_compression = True
#     for idx in range(2, num_sets):
#         params[idx].arch_param.rescale_fusion = True
#     for idx in range(3, num_sets):
#         params[idx].arch_param.cache_style = CacheStyle.CONST
#     for idx in range(4, num_sets):
#         params[idx].arch_param.mod_down_reorder = True
#     for idx in range(5, num_sets):
#         params[idx].arch_param.cache_style = CacheStyle.BETA
#     for idx in range(6, num_sets):
#         params[idx].arch_param.cache_style = CacheStyle.ALPHA
#     return params


# def get_params():
#     num_sets = 7
#     params = [copy.deepcopy(GPU_PARAMS) for _ in range(num_sets)]
#     for idx in range(1, num_sets):
#         params[idx].arch_param.key_compression = True
#     for idx in range(2, num_sets):
#         params[idx].arch_param.rescale_fusion = True
#     for idx in range(3, num_sets):
#         params[idx].arch_param.cache_style = CacheStyle.CONST
#     for idx in range(4, num_sets):
#         params[idx].arch_param.cache_style = CacheStyle.BETA
#     for idx in range(5, num_sets):
#         params[idx].arch_param.cache_style = CacheStyle.ALPHA
#     for idx in range(6, num_sets):
#         params[idx].arch_param.mod_down_reorder = True
#     return params


def get_mem_params():
    # num_sets = 7
    # num_sets = 6
    num_sets = 5
    params = [copy.deepcopy(GPU_PARAMS) for _ in range(num_sets)]

    for idx in range(1, num_sets):
        params[idx].arch_param.cache_style = CacheStyle.CONST
    for idx in range(2, num_sets):
        params[idx].arch_param.cache_style = CacheStyle.BETA
    for idx in range(3, num_sets):
        params[idx].arch_param.cache_style = CacheStyle.ALPHA
    for idx in range(4, num_sets):
        params[idx].arch_param.mod_down_reorder = True
    # for idx in range(1, num_sets):
    #     params[idx].arch_param.rescale_fusion = True

    return params


def get_alg_params():
    params = []

    params.append(copy.deepcopy(BEST_PARAMS))
    params[-1].fft_style = FFTStyle.BSGS
    params[-1].arch_param.key_compression = False
    params[-1].arch_param.rescale_fusion = False

    params.append(copy.deepcopy(BEST_PARAMS))
    params[-1].fft_style = FFTStyle.BSGS
    params[-1].arch_param.key_compression = False
    # params[-1].arch_param.rescale_fusion = False

    params.append(copy.deepcopy(BEST_PARAMS))
    # params[-1].fft_style = FFTStyle.BSGS
    params[-1].arch_param.key_compression = False
    # params[-1].arch_param.rescale_fusion = False

    params.append(copy.deepcopy(BEST_PARAMS))
    # params[-1].fft_style = FFTStyle.BSGS
    # params[-1].arch_param.key_compression = False
    # params[-1].arch_param.rescale_fusion = False

    # for idx in range(6, num_sets):
    # params[idx].arch_param.key_compression = True

    # params.append(copy.deepcopy(BEST_PARAMS))
    # params[-1].arch_param.key_compression = False
    # params[-1].fft_style = FFTStyle.BSGS

    # params.append(copy.deepcopy(BEST_PARAMS))
    # params[-1].arch_param.key_compression = False

    # params.append(copy.deepcopy(BEST_PARAMS))

    return params


BEST_ARCH_PARAMS = ArchParam(
    karatsuba=True,
    key_compression=True,
    rescale_fusion=True,
    cache_style=CacheStyle.ALPHA,
    mod_down_reorder=True,
)

HUGE_ARCH_PARAMS = ArchParam(
    karatsuba=True,
    key_compression=True,
    rescale_fusion=True,
    cache_style=CacheStyle.HUGE,
    mod_down_reorder=True,
)


ALG_OPT_PARAMS = SchemeParams(
    logq=50,
    logN=17,
    override_Q0_limbs=35,
    dnum=3,
    fft_iters=3,
    fft_style=FFTStyle.UNROLLED_HOISTED,
    arch_param=BEST_ARCH_PARAMS,
)


BEST_PARAMS = SchemeParams(
    logq=50,
    logN=17,
    dnum=2,
    fft_iters=6,
    fft_style=FFTStyle.UNROLLED_HOISTED,
    arch_param=BEST_ARCH_PARAMS,
)

Alg_benchmark_baseline = dataclasses.replace(BEST_PARAMS)
Alg_benchmark_baseline.arch_param = dataclasses.replace(BEST_ARCH_PARAMS)
Alg_benchmark_baseline.fft_style = FFTStyle.BSGS
Alg_benchmark_baseline.arch_param.key_compression = False
Alg_benchmark_baseline.arch_param.rescale_fusion = False

Alg_benchmark_mod_down_merge = dataclasses.replace(Alg_benchmark_baseline)
Alg_benchmark_mod_down_merge.arch_param = dataclasses.replace(
    Alg_benchmark_baseline.arch_param
)
Alg_benchmark_mod_down_merge.arch_param.rescale_fusion = True

Alg_benchmark_mod_down_hoist = dataclasses.replace(Alg_benchmark_mod_down_merge)
Alg_benchmark_mod_down_hoist.arch_param = dataclasses.replace(
    Alg_benchmark_mod_down_merge.arch_param
)
Alg_benchmark_mod_down_hoist.fft_style = FFTStyle.UNROLLED_HOISTED


HUGE_PARAMS = SchemeParams(
    logq=50,
    logN=17,
    dnum=2,
    fft_iters=6,
    fft_style=FFTStyle.UNROLLED_HOISTED,
    arch_param=HUGE_ARCH_PARAMS,
)
