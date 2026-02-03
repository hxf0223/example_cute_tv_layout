from cutlass import cute
from cute_viz import render_layout_svg, render_swizzle_layout_svg


@cute.jit
def test_swizzle_layout():
    layout_2d = cute.make_layout((32, 128), stride=(128, 1))
    sw = cute.make_swizzle(5, 0, 7)
    swizzled_layout = cute.make_composed_layout(sw, 0, layout_2d)
    render_layout_svg(layout_2d, "out/original_layout.svg")
    render_swizzle_layout_svg(swizzled_layout, "out/swizzled_layout.svg")


test_swizzle_layout()
