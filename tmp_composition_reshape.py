from cutlass import cute
from cute_viz import render_layout_svg, display_layout


@cute.jit
def test_composition_reshape():
    layout_1d = cute.make_layout(20, stride=2)
    tiler = cute.make_layout((5, 4), stride=(4, 1))
    result = cute.composition(layout_1d, tiler)
    render_layout_svg(layout_1d, "out/layout_1d.svg")
    render_layout_svg(tiler, "out/tiler.svg")
    render_layout_svg(result, "out/reshaped_2d.svg")


test_composition_reshape()
