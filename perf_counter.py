from dataclasses import dataclass, field
from enum import Enum


class OpType(Enum):
    READ = 0
    WRITE = 1
    AUTO_READ = 2
    KICK_OUT = 3


@dataclass
class ArchCounter:
    cyc: int = 0

    """
    sm tag is short for slow model (FPGA model)
    fm tag is short for fast model (real chip model)
    """

    add_cyc_sm: int = 0
    mult_cyc_sm: int = 0
    add_cyc_fm: int = 0
    mult_cyc_fm: int = 0

    ## ntt is total ntt cycles for all of bootstrapping
    ## ntt_oft is the overhead from computing twiddle factors on the fly

    ntt_cyc_sm: int = 0
    ntt_otf_cyc_sm: int = 0
    ntt_cyc_fm: int = 0
    ntt_otf_cyc_fm: int = 0

    bsc_cyc_sm: int = 0  ## basis conversion
    bsc_cyc_fm: int = 0

    ## worst-case and best-case automorphism
    ## worst-case is two cycles
    ## best-case is zero cycles (additional latency)

    auto_cyc_sm_wc: int = 0
    auto_cyc_sm_bc: int = 0
    auto_cyc_fm_wc: int = 0
    auto_cyc_fm_bc: int = 0

    """
    Reads are reads in from external memory
    Writes are writes out to external memory
    Untagged dram tracks local storage needed in external memory

    The default model is that nothing is stored on chip. 
    """

    op_list: list = field(default_factory=list)
    min_bytes: int = 0

    # dram_limb_rd: int = 0
    _dram_limb_rd: int = 0

    @property
    def dram_limb_rd(self):
        return self._dram_limb_rd

    @dram_limb_rd.setter
    def dram_limb_rd(self, value):
        self.op_list.append((OpType.READ, value - self._dram_limb_rd))
        self._dram_limb_rd = value

    # dram_limb_wr: int = 0
    _dram_limb_wr: int = 0

    @property
    def dram_limb_wr(self):
        return self._dram_limb_wr

    @dram_limb_wr.setter
    def dram_limb_wr(self, value):
        self.op_list.append((OpType.WRITE, value - self._dram_limb_wr))
        self._dram_limb_wr = value

    # dram_limb: int = 0

    # dram_ntt_rd: int = 0
    # dram_ntt: int = 0

    # dram_auto_rd: int = 0
    _dram_auto_rd: int = 0

    @property
    def dram_auto_rd(self):
        return self._dram_auto_rd

    @dram_auto_rd.setter
    def dram_auto_rd(self, value):
        self.op_list.append((OpType.AUTO_READ, value - self._dram_auto_rd))
        self._dram_auto_rd = value

    # dram_auto: int = 0

    # dram_plain_rd: int = 0
    # dram_plain: int = 0

    """
    Keeps track of number of read and write ports required to keep all funits busy
    """
    rd_ports: int = 0
    wr_ports: int = 0

    def __add__(self, other):
        return ArchCounter(
            _dram_limb_rd=self.dram_limb_rd + other.dram_limb_rd,
            _dram_limb_wr=self.dram_limb_wr + other.dram_limb_wr,
            _dram_auto_rd=self.dram_auto_rd + other.dram_auto_rd,
            # op_list=self.op_list + other.op_list,
            add_cyc_fm=self.add_cyc_fm + other.add_cyc_fm,
            add_cyc_sm=self.add_cyc_sm + other.add_cyc_sm,
            mult_cyc_fm=self.mult_cyc_fm + other.mult_cyc_fm,
            mult_cyc_sm=self.mult_cyc_sm + other.mult_cyc_sm,
        )

    @property
    def dram_total_rdwr_small(self):
        return (
            self.dram_limb_rd
            + self.dram_limb_wr
            # + self.dram_ntt_rd
            + self.dram_auto_rd
            # + self.dram_plain_rd
        )

    # @property
    # def dram_total_rdwr_large(self):
    #     return self.dram_ntt_rd + self.dram_auto_rd + self.dram_plain_rd

    # @property
    # def dram_total_small(self):
    #     return self.dram_limb + self.dram_ntt + self.dram_auto + self.dram_plain

    # @property
    # def dram_total_large(self):
    #     return self.dram_ntt + self.dram_auto + self.dram_plain

    @property
    def total_cycle_sm_bc(self):
        return self.add_cyc_sm + self.mult_cyc_sm + self.auto_cyc_sm_bc

    @property
    def total_cycle_fm_bc(self):
        return self.add_cyc_fm + self.mult_cyc_fm + self.auto_cyc_fm_bc

    @property
    def total_cycle_sm_wc(self):
        return self.add_cyc_sm + self.mult_cyc_sm + self.auto_cyc_sm_wc

    @property
    def total_cycle_fm_wc(self):
        return self.add_cyc_fm + self.mult_cyc_fm + self.auto_cyc_fm_wc


@dataclass
class SWCounter:
    mult: int = 0
    add: int = 0
    ntt: int = 0

    def __add__(self, other):
        return SWCounter(
            mult=self.mult + other.mult,
            add=self.add + other.add,
            ntt=self.ntt + other.ntt,
        )

    @property
    def total_ops(self):
        return self.mult + self.add


@dataclass
class PerfCounter:
    sw: SWCounter = field(default_factory=SWCounter)
    arch: ArchCounter = field(default_factory=ArchCounter)

    def __add__(self, other):
        return PerfCounter(
            sw=self.sw + other.sw,
            arch=self.arch + other.arch,
        )
