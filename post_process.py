import operator
import subprocess

import profiler


def accumulate(data):
    acc_frame = profiler.Frame(name=data.name, stats=data.stats)
    acc_frame.children = [accumulate(child) for child in data.children]
    for child in acc_frame.children:
        acc_frame.stats += child.stats
    return acc_frame


def walk_tree(data, gen_func, attr=None, depth=0):
    stack = [(data, 0)]
    out_list = []
    while stack:
        frame, ptr = stack.pop()
        valid_depth = not depth or len(stack) < depth
        if ptr == 0 and valid_depth:
            out_list.append(gen_func(frame, stack, attr))

        if ptr < len(frame.children):
            stack.append((frame, ptr + 1))
            stack.append((frame.children[ptr], 0))
    return out_list


def prettyprint(data, attr, depth=0):
    def gen_func(frame, stack, attr=None):
        attr_val = operator.attrgetter(attr)(frame.stats)
        attr_val /= 1e9
        leading_space = "  " * len(stack)
        leaf_node = (depth and len(stack) == depth - 1) or not frame.children
        colon = "" if leaf_node else ":"
        out_str = f"{attr_val:.5g} G  {leading_space}{frame.name}{colon}"
        return out_str

    return "\n".join(walk_tree(data, gen_func, attr, depth))


def get_table(data, attr_list, depth):
    def gen_label(frame, stack, attr=None):
        leading_space = "  " * len(stack)
        out_str = f"{leading_space}{frame.name}"
        return out_str

    def gen_val(frame, stack, attr):
        return operator.attrgetter(attr)(frame.stats) / 1e9

    table = [walk_tree(data, gen_label, depth=depth)] + [
        walk_tree(data, gen_val, attr, depth=depth) for attr in attr_list
    ]

    return table


def get_attr(data, attr, depth=0):
    data_str = prettyprint(data, attr, depth)
    vals = data_str.split(" ")
    return float(vals[0])


def flamegraph(name, data, attr, dir="data", gen_svg=True):
    def gen_func(frame, stack, attr=None):
        attr_val = operator.attrgetter(attr)(frame.stats)
        path = ";".join([parent[0].name for parent in stack] + [frame.name])
        out_str = f"{path} {attr_val}"
        return out_str

    print("in flamegraph generator")
    subprocess.call(["mkdir", "-p", dir])
    print("finished mkdir call", dir)
    flame_txt = f"{dir}/{name}.log"
    print("flame text file", flame_txt)
    with open(flame_txt, "w") as txt_file:
        flame_data = "\n".join(walk_tree(data, gen_func, attr))
        txt_file.write(flame_data)

    print("finished writing file")
    if gen_svg:
        print("in gen_svg")
        flame_svg = f"{dir}/{name}.svg"
        chart_units = attr.split(".")[-1]
        print(flame_svg, chart_units)
        with open(flame_svg, "w") as svg_file:
            print(
                "calling command flamgraph.pl --title",
                name,
                "--countname",
                chart_units,
                flame_txt,
            )
            command = "flamegraph.pl"

            subprocess.call(
                [
                    command,
                    "--title",
                    name,
                    # "--width", "2400",
                    "--countname",
                    chart_units,
                    flame_txt,
                ],
                stdout=svg_file,
            )


def stack_trace(data):
    def gen_func(frame, stack, attr=None):
        leading_space = "  " * len(stack)
        out_str = f"{leading_space}{frame.name}:"
        return out_str

    return "\n".join(walk_tree(data, gen_func, None))
