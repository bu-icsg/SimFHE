import params


def test_N():
    config = params.ConfigParam()
    assert config.N == (2 ** 16)


def test_limbs():
    config = params.ConfigParam()
    assert config.limbs == 47


def test_eval_img_limbs():
    config = params.ConfigParam()
    assert config.exp_img_ctxt.limbs == 32


def test_slot_to_coeff_limbs():
    config = params.ConfigParam()
    assert config.slot_coeff_ctxt.limbs == 19
