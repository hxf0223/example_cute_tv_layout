from cutlass import cute
from cute_viz import render_tv_layout_svg


@cute.jit
def display_tv_layout():
    thr_layout = cute.make_layout((16, 4), stride=(1, 16))
    val_layout = cute.make_layout((4, 1), stride=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    render_tv_layout_svg(tv_layout, tiler_mn, "out/tv_layout.svg")


display_tv_layout()
