# SimFHE

SimFHE is a python-based custom simulator that can be used to benchmark the compute and memory requirements of a CKKS-based application.
SimFHE models the primitive operations in CKKS FHE scheme by keeping track of both individual compute operations as well as the DRAM transfers for a given cache size.
This enables one to benchmark the compute and memory requirements of individual complex operations as well as the CKKS-based application as a whole.

An execution of primitive operations and bootstrapping in SimFHE is parameterized by the CKKS scheme parameters, the number of functional units, the size of the on-chip memory, and the MAD optimizations to include in bootstrapping. 
SimFHE tracks compute at the modular arithmetic level, i.e., in terms of modular multiplications and additions.
SimFHE tracks DRAM transfers based on the data size and the available cache size instead of directly tracking cache hits or misses through actual data reads.
SimFHE implements the proposed memory-aware design optimizations in a modular fashion, allowing to toggle between each optimization independently so as to isolate the benefit of each optimization. 
In addition, many of these optimizations are memory-aware, and so for a large enough on-chip memory, SimFHE will automatically deploy the applicable optimization. 

SimFHE also helps in optimal parameter selection by combining the simulation of algorithmic optimizations and hardware constraints. 
Given the on-chip memory size, SimFHE searches the CKKS parameter space using a brute-force approach to find the optimal parameters that maximize the bootstrapping throughput for a given underlying compute system. 
The parameters computed by SimFHE include the high-level ring parameters such as the polynomial degree and coefficient modulus as well as the low-level, internal bootstrapping parameters such as the number of FFT iterations (fftIter) and the number of digits (dnum) in the KeySwitch operation. 
SimFHE can help answer questions like how changing a specific CKKS algorithm parameter or system constraint such as on-chip memory size would affect the overall bootstrapping performance. 
This, in turn, helps save significant design and development time.
Moreover, the security level constraints limit the number of possible parameter sets, so the optimal parameter search takes only a few minutes.
