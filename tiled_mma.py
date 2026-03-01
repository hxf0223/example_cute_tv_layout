from cutlass import cute
from cute_viz import render_tv_layout_svg, render_layout_svg
import cutlass

# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py


@cute.jit
def print_tv_layout():
    # layout of B for SM80_16x8x16_F16F16F16F16_TN
    layout = cute.make_layout(((4, 8), (2, 2)), stride=((16, 1), (8, 64)))  # TV-layout (T, V) -> (M, K)
    render_layout_svg(layout, "out/layout.svg")
    inv_layout = cute.right_inverse(layout)
    print("inv layout", inv_layout)


print_tv_layout()
