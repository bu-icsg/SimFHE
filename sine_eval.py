"""

parameters  

q/delta = 1024 (2^10)
degree of sine approximation = 31
r = 2 
K = 12

"""

from math import log2, ceil, pi


def ct_ct_mult(ct1, ct2):
    """
    homomorphic ciphertext product
    """
    pass


def ct_ct_add(ct1, ct2):
    """
    homomorphic ciphertext addition
    """
    pass


def ct_pt_mult(ct, pt):
    """
    Multiplies a plaintext with a ciphertext
    """
    pass


def ct_pt_add(ct, pt):
    """
    Adds a plaintext to a ciphertext
    """
    pass


def rescale(ct):
    """
    rescales ct
    """
    pass


def level(ct):
    """
    returns levels in ciphertext
    """
    pass


class CoeffPoly:
    """
    Polynomial class for plaintext polynomial arithmetic
    """

    def __init__(self, coeffs):
        self.coeffs = coeffs.copy()
        ## prune leading zeros
        if len(self.coeffs) == 0:
            self.coeffs = [0]
        else:
            while self.coeffs[-1] == 0 and len(self.coeffs) > 1:
                self.coeffs = self.coeffs[:-1]

    @property
    def degree(self):
        return len(self.coeffs) - 1

    def zero():
        return CoeffPoly([0])

    def __call__(self, x):
        res = 0
        x_pow = 1
        for i in range(len(self.coeffs)):
            res += self.coeffs[i] * x_pow
            x_pow *= x
        return res

    def scalar_add(self, other):
        res_coeffs = self.coeffs.copy()
        if len(res_coeffs) > 0:
            res_coeffs[0] += other
        else:
            res_coeffs.append(other)
        return CoeffPoly(res_coeffs)

    def __add__(self, other):
        if not isinstance(other, CoeffPoly):
            return self.scalar_add(other)

        if self.degree < other.degree:
            return other + self
        ## we have self.degree >= other.degree
        res_coeffs = []
        for i in range(other.degree + 1):
            res_coeffs.append(self.coeffs[i] + other.coeffs[i])
        for i in range(other.degree + 1, self.degree + 1):
            res_coeffs.append(self.coeffs[i])
        return CoeffPoly(res_coeffs)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (other * (-1))

    def scalar_mult(self, other):
        res_coeffs = [x * other for x in self.coeffs]
        return CoeffPoly(res_coeffs)

    def __mul__(self, other):
        if not isinstance(other, CoeffPoly):
            return self.scalar_mult(other)

        res_degree = self.degree + other.degree
        res_coeffs = [0] * (res_degree + 1)
        for i in range(self.degree + 1):
            for j in range(other.degree + 1):
                res_coeffs[i + j] += self.coeffs[i] * other.coeffs[j]
        return CoeffPoly(res_coeffs)

    def __rmul__(self, other):
        return self * other

    def __eq__(self, other):
        if not isinstance(other, CoeffPoly):
            return False
        if self.degree != other.degree:
            return False
        for (c, o) in zip(self.coeffs, other.coeffs):
            if c != o:
                return False
        return True

    def __neq__(self, other):
        return not (self == other)

    def __str__(self):
        res = ""
        for c in self.coeffs:
            res += str(c) + ", "
        res = res[:-2]
        return res


## common inputs
ct = " input ciphertext "
r = 2
K = 12
d = 31
delta = 50
m = int(ceil(log2(d + 1)))
ell = m // 2

T_polys = [CoeffPoly([1]), CoeffPoly([0, 1])]
T_prod = CoeffPoly([0, 2])
for i in range(2, pow(2, m) + 1):
    new_T = (T_polys[i - 1] * T_prod) - T_polys[i - 2]
    T_polys.append(new_T)

for i in range(0, pow(2, m - 1) + 1):
    should_be_2i = (2 * T_polys[i] * T_polys[i]) - 1
    assert should_be_2i == T_polys[2 * i]

print("T polys computed")

c_coeffs = []
for _ in T_polys:
    c_coeffs.append(1)

p = CoeffPoly.zero()
for (c_i, T_i) in zip(c_coeffs, T_polys):
    p = p + (c_i * T_i)
assert p == sum(T_polys)  ## only correct bc all c_i = 1

## compute the u polynomials
upper_giant_step = d // ell

u_polys = []
should_be_p = CoeffPoly.zero()
for i in range(upper_giant_step + 1):
    u = CoeffPoly([0])
    for j in range(pow(2, ell)):
        u = u + c_coeffs[i * pow(2, ell) + j] * T_polys[j]
    u_polys.append(u)

T_0 = 1
T_1 = ct_pt_add(ct, -0.5 / (pow(2, r + 1) * K))

## parameters


u_max_t = u_polys[-1]
q_T = [" modulus at each level. not quite sure what this is "]
delta_T = [" These are the values of delta at each level "]


def eval_recurse(target_delta, m, ell, p_t, T):
    c = [" these are the coefficients of p_t for the basis T "]
    if d < pow(2, ell):
        if p_t == u_max_t and ell > pow(2, m) - pow(2, ell - 1) and ell > 1:
            new_m = ceil(log2(d + 1))
            new_ell = new_m // 2
            return eval_recurse(target_delta, new_m, new_ell, p_t, T)
        else:
            ## seems to start as a plaintext, then becomes a ciphertext after the first round
            ## this is probably just removing the T_0 product
            ct_new = round(c[0] * delta * q_T[d])
            for i in range(d, 0, -1):
                ct_new = ct_ct_add(
                    ct_new, ct_pt_mult(T[i], (c[i] * delta * q_T[d]) // delta_T[i])
                )
            return rescale(ct_new)
    ## express p(t) = q(t) * T_{2^{m-1}}+ r(t)
    q_t = " quotient term "
    r_t = " remainder term "

    left_target_scale = (target_delta * q_T[pow(2, m - 2)]) / delta_T[pow(2, m - 1)]
    ct_0 = eval_recurse(left_target_scale, m - 1, ell, q_t, T)
    ct_1 = eval_recurse(target_delta, m - 1, ell, r_t, T)
    ct_0 = ct_ct_mult(ct_0, T[pow(2, m - 1)])
    if level(ct_0) > level(ct_1):
        ct_0 = ct_ct_add(rescale(ct_0), ct_1)
    else:
        ct_0 = rescale(ct_ct_add(ct_0, ct_1))

    return ct_0


## reassign delta

T_baby_step = [T_0, T_1]
"""
This is not optimal. This should be done in depth ell.
"""
for i in range(2, pow(2, ell) + 1):
    T_i = ct_ct_mult(ct, T_baby_step[i - 1])
    T_i = ct_pt_mult(T_i, 2)
    T_i = ct_ct_add(T_i, -1 * T_baby_step[i - 2])
    T_baby_step.append(T_i)


T_giant_step = []
T_prev = T_baby_step[-1]  ## T_{2^ell}
index = pow(2, ell)
while index <= pow(2, m):
    T_next = ct_pt_add(ct_pt_mult(ct_ct_mult(T_prev, T_prev), 2), -1)
    T_giant_step.append(T_next)
    index *= 2

T = T_baby_step + T_giant_step

## compute target delta
target_delta = " placeholder "
p_T = " representation of p(t) in the basis T "
ct_prime = eval_recurse(target_delta, m, ell, p_T, T)

for i in range(r):
    ct_prime = ct_pt_add(
        ct_pt_mult(2, ct_ct_mult(ct_prime, ct_prime)),
        -1 * pow(1 / (2 * pi), 1 / pow(2, r - i)),
    )
    ct_prime = rescale(ct_prime)

## resent scaling factor to the original delta

## return ct_prime
