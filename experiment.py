from math import ceil, exp, log
from dataclasses import dataclass, field
from typing import List, Dict
import csv
import tqdm
import tabulate

import profiler
import params
import post_process

RPT_ATTR = {
    "total ops": "sw.total_ops",
    "total mult": "sw.mult",
    "dram total": "arch.dram_total_rdwr_small",
    "dram limb rd": "arch.dram_limb_rd",
    "dram limb wr": "arch.dram_limb_wr",
    "dram key rd": "arch.dram_auto_rd",
    # "total cycles (slow, worst case)": "arch.total_cycle_sm_wc",
    # "total cycles (slow, best case)": "arch.total_cycle_sm_bc",
    # "total cycles (fast, worst case)": "arch.total_cycle_fm_wc",
    # "total cycles (fast, best case)": "arch.total_cycle_fm_bc"
}


@dataclass
class Target:
    name: str
    depth: int
    args: List = field(default_factory=list)
    kwargs: List = field(default_factory=dict)


def generate_profile(target: Target):
    experiment = profiler.Profiler(target.name)
    experiment.profile(target.name, *target.args, **target.kwargs)
    return experiment


def generate_flamegraph(experiment: profiler.Profiler, attr, suffix=""):
    graph_name = experiment.name + f"_{attr}"
    if suffix:
        graph_name += f"_{suffix}"
    post_process.flamegraph(graph_name, experiment.data, attr)


def get_table(data, attr_dict, depth):
    table = post_process.get_table(data, attr_dict.values(), depth)
    # transpose
    ttable = []
    nrow, ncol = len(table[0]), len(table)
    for row_idx in range(nrow):
        ttable.append([table[col_idx][row_idx] for col_idx in range(ncol)])
    return ttable


def save_csv(headers, data, filepath):
    headers = ["logN", "dnum", "fft_iters", "fresh_limbs", "op_count", "total_mem"]
    with open(filepath, "w") as csvfile:
        csvwriter = csv.writer(csvfile, dialect="excel")
        csvwriter.writerow(headers)
        csvwriter.writerows(data)


def get_headers(attr_dict):
    return ["fn"] + list(attr_dict.keys())


def run_single(target, attr_dict=RPT_ATTR):
    experiment = generate_profile(target)
    acc_data = post_process.accumulate(experiment.data)
    data = get_table(acc_data, attr_dict, target.depth)
    headers = get_headers(attr_dict)
    return (headers, data)


def run_mutiple(targets, attr_dict=RPT_ATTR):
    cum_data = []
    headers = get_headers(attr_dict)
    for target in targets:
        experiment = generate_profile(target)
        acc_data = post_process.accumulate(experiment.data)
        data = get_table(acc_data, attr_dict, target.depth)
        cum_data += data
    return (headers, cum_data)


def compare_bootstrap(schemes, attr_dict=RPT_ATTR):
    cum_data = []
    headers = get_headers(attr_dict)
    for scheme_params in schemes:
        target = Target("bootstrap.bootstrap", 1, [scheme_params])
        experiment = generate_profile(target)
        acc_data = post_process.accumulate(experiment.data)
        data = get_table(acc_data, attr_dict, target.depth)
        cum_data += data
    return (headers, cum_data)


def print_table(headers, data):
    tabulate.PRESERVE_WHITESPACE = True
    print(tabulate.tabulate(data, headers=headers))
    tabulate.PRESERVE_WHITESPACE = False


def aux_subroutine_benchmarks(scheme_params: params.SchemeParams):
    micro_args = [scheme_params.mod_raise_ctxt, scheme_params]
    targets = [
        Target("micro_benchmarks.mod_up", 1, micro_args),
        Target("micro_benchmarks.mod_down", 1, micro_args),
        Target("micro_benchmarks.decomp", 1, micro_args),
        Target("micro_benchmarks.inner_product", 1, micro_args),
        Target("micro_benchmarks.automorph", 1, micro_args),
    ]
    headers, data = run_mutiple(targets)
    print_table(headers, data)
    save_csv(headers, data, "data/aux_subroutine.csv")


def low_level_benchmark(scheme_params: params.SchemeParams):
    micro_args = [scheme_params.mod_raise_ctxt, scheme_params]
    targets = [
        Target("micro_benchmarks.pt_add", 1, micro_args),
        Target("micro_benchmarks.add", 1, micro_args),
        Target("micro_benchmarks.pt_mult", 1, micro_args),
        Target("micro_benchmarks.mult", 1, micro_args),
        Target("micro_benchmarks.rotate", 1, micro_args),
        Target("micro_benchmarks.hoisted_rotate", 1, micro_args),
    ]
    headers, data = run_mutiple(targets)
    print_table(headers, data)
    save_csv(headers, data, "data/low_level.csv")


def high_level_benchmark(scheme_params: params.SchemeParams):
    targets = [
        Target("fft.fft", 2, [scheme_params.mod_raise_ctxt, scheme_params]),
        Target("eval_sine.eval_sine", 2, [scheme_params.cts_ctxt, scheme_params]),
    ]
    headers, data = run_mutiple(targets)
    print_table(headers, data)
    save_csv(headers, data, "data/high_level.csv")


def bootstrap_benchmark(scheme_params: params.SchemeParams, rpt_depth=3):
    targets = [Target("bootstrap.bootstrap", rpt_depth, [scheme_params])]
    headers, data = run_mutiple(targets)
    print_table(headers, data)
    save_csv(headers, data, "data/bootstrap.csv")


def fft_best_params():
    """
    Sweep for each logN, 16 and 17
    for each dnum from 1 to 6
    for each squashing 1 to 5
    """

    logNVals = [16, 17]
    dnum_vals = range(1, 7)
    squashing_vals = range(1, 7)
    total_runs = len(logNVals) * len(dnum_vals) * len(squashing_vals)

    table = []
    with tqdm.tqdm(total=total_runs) as pbar:
        for logN in logNVals:
            for dnum in dnum_vals:
                fft_iter_vals = [int(ceil((logN - 1) / x)) for x in squashing_vals]
                for fft_iters in fft_iter_vals:
                    scheme_params = params.SchemeParams(
                        logN=logN,
                        dnum=dnum,
                        fft_iters=fft_iters,
                        fft_style=params.FFTStyle.UNROLLED_HOISTED,
                        arch_param=params.BEST_ARCH_PARAMS,
                    )

                    try:
                        start_limbs = scheme_params.bootstrapping_Q0_limbs
                    except ValueError:
                        pbar.update(1)
                        continue

                    # target = Target(
                    #     "bootstrap.fft", [scheme_params.mod_raise_ctxt, scheme_params]
                    # )
                    target = Target("bootstrap.bootstrap", 1, [scheme_params])
                    experiment = generate_profile(target)
                    acc_data = post_process.accumulate(experiment.data)

                    op_count = post_process.get_attr(acc_data, "sw.total_ops", 1)
                    total_mem = post_process.get_attr(
                        acc_data, "arch.dram_total_rdwr_small", 1
                    )

                    table.append(
                        [
                            logN,
                            dnum,
                            fft_iters,
                            scheme_params.fresh_limbs,
                            op_count,
                            total_mem,
                        ]
                    )

                    pbar.update(1)

    headers = ["logN", "dnum", "fft_iters", "fresh_limbs", "op_count", "total_mem"]

    print_table(headers, table)
    save_csv(headers, table, "data/fft.csv")


if __name__ == "__main__":
    # scheme_params = params.BEST_PARAMS
    # micro_args = [scheme_params.mod_raise_ctxt, scheme_params]
    # targets = [
    #     Target("micro_benchmarks.mod_up", 3, micro_args),
    #     Target("micro_benchmarks.mod_down", 3, micro_args),
    #     Target("micro_benchmarks.rotate", 4, micro_args),
    # ]
    # headers, data = run_mutiple(targets)
    # print_table(headers, data)

    # for scheme_params in [params.GPU_PARAMS, params.BEST_PARAMS]:
    #     print(scheme_params)
    #     aux_subroutine_benchmarks(scheme_params)
    #     low_level_benchmark(scheme_params)
    #     print()

    # for scheme_params in [params.BEST_PARAMS]:
    #     targets = [
    #         Target(
    #             "poly_eval.poly_eval",
    #             1,
    #             [scheme_params.mod_raise_ctxt, scheme_params.arch_param, 63, 2],
    #         )
    #     ]
    #     headers, data = run_mutiple(targets)
    #     print_table(headers, data)
    #     print()

    targets = []
    # for scheme_params in [params.GPU_PARAMS, params.LATTIGO_PARAMS, params.BEST_PARAMS]:
    # for scheme_params in [params.LATTIGO_PARAMS]:
    # for scheme_params in [params.BEST_PARAMS, params.HUGE_PARAMS]:
    # for scheme_params in [
    #     params.GPU_PARAMS,
    #     params.Mem_benchmark_O_1_cache,
    #     params.Mem_benchmark_beta_cache,
    #     params.Mem_benchmark_alpha_cache,
    #     params.Mem_benchmark_reorder,
    # ]:
    for scheme_params in [
        params.Alg_benchmark_baseline,
        params.Alg_benchmark_mod_down_merge,
        params.Alg_benchmark_mod_down_hoist,
        params.BEST_PARAMS,
    ]:
        # for scheme_params in [params.BEST_PARAMS]:
        print(scheme_params)
        targets.append(
            #         Target(
            #             "logistic_regression.inner_product",
            #             1,
            #             [scheme_params.fresh_ctxt, scheme_params.arch_param, 256],
            #         ),
            #         Target(
            #             "logistic_regression.sigmoid_product",
            #             1,
            #             [scheme_params.fresh_ctxt, scheme_params.arch_param],
            #         ),
            #         Target(
            #             "logistic_regression.iteration",
            #             1,
            #             [scheme_params.fresh_ctxt, scheme_params.arch_param, 256],
            #         ),
            Target(
                "bootstrap.bootstrap",
                1,
                [scheme_params],
            ),
            # Target(
            #     "logistic_regression.logistic_regression",
            #     2,
            #     [scheme_params.fresh_ctxt, scheme_params.arch_param],
            # ),
            #         Target(
            #             "logistic_regression.bootstrap_regression",
            #             1,
            #             [scheme_params],
            #         ),
        )
    headers, data = run_mutiple(targets)
    print_table(headers, data)
    print()

    # for scheme_params in [params.BEST_PARAMS]:
    #     targets = [
    #         Target(
    #             "fft.fft_inner_hoisted_unrolled",
    #             1,
    #             [scheme_params.mod_raise_ctxt, scheme_params.arch_param, 7],
    #         ),
    #         Target(
    #             "fft.fft_inner_bsgs_hoisted",
    #             1,
    #             [scheme_params.mod_raise_ctxt, scheme_params.arch_param, 7, 1],
    #         ),
    #         Target(
    #             "fft.fft_inner_hoisted_unrolled",
    #             1,
    #             [scheme_params.mod_raise_ctxt, scheme_params.arch_param, 63],
    #         ),
    #         Target(
    #             "fft.fft_inner_bsgs",
    #             1,
    #             [scheme_params.mod_raise_ctxt, scheme_params.arch_param, 63],
    #         ),
    #         Target(
    #             "fft.fft_inner_bsgs_hoisted",
    #             1,
    #             [scheme_params.mod_raise_ctxt, scheme_params.arch_param, 63, 1],
    #         ),
    #         Target(
    #             "fft.fft_inner_bsgs_hoisted",
    #             1,
    #             [scheme_params.mod_raise_ctxt, scheme_params.arch_param, 63, 2],
    #         ),
    #         Target(
    #             "fft.fft_inner_bsgs_hoisted",
    #             1,
    #             [scheme_params.mod_raise_ctxt, scheme_params.arch_param, 63, 4],
    #         ),
    #         Target(
    #             "fft.fft_inner_bsgs_hoisted",
    #             1,
    #             [scheme_params.mod_raise_ctxt, scheme_params.arch_param, 63],
    #         ),
    #     ]
    #     headers, data = run_mutiple(targets)
    #     print_table(headers, data)

    # scheme_params_list = params.get_params()
    # scheme_params_list = params.get_mem_params()
    # scheme_params_list = params.get_alg_params()
    # for scheme_params in scheme_params_list:
    # i=0
    # for scheme_params in [params.GPU_PARAMS,params.GPU_PARAMS, params.BEST_PARAMS]:
    #     if i == 1:
    #         scheme_params.arch_param.rescale_fusion=True

    #     print(scheme_params)
    #     print(scheme_params.arch_param)
    #     low_level_benchmark(scheme_params)
    #     # high_level_benchmark(scheme_params)
    #     # bootstrap_benchmark(scheme_params, rpt_depth=1)
    #     print()
    #     i += 1

    # headers, data = compare_bootstrap(scheme_params_list)
    # print_table(headers, data)

    # run_benchmark(cts_fft)
    # for lvl_squashed in range(1, 6):
    #     fft_iter = Target(
    #         "bootstrap.fft_inner_hoisted_unrolled",
    #         [scheme_params.mod_raise_ctxt, lvl_squashed],
    #     )
    #     print(f"lvl squashed: {lvl_squashed}")
    #     run_benchmark(fft_iter, rpt_depth=1)
    #     print()

    # fft_best_params()
