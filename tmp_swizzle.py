from cutlass import cute
from cute_viz import render_layout_svg, render_swizzle_layout_svg


@cute.jit
def test_swizzle_layout():
    layout_2d = cute.make_layout((32, 32), stride=(1, 32))
    sw = cute.make_swizzle(5, 0, 5)
    swizzled_layout = cute.make_composed_layout(sw, 0, layout_2d)
    render_layout_svg(layout_2d, "out/original_layout.svg")
    render_swizzle_layout_svg(swizzled_layout, "out/swizzled_layout.svg")


test_swizzle_layout()


@cute.jit
def test_swizzle_layout2():  # 该swizzle不起作用
    """该swizzle不起作用, 因为fast dimension长度不满足要求"""
    layout_3d = cute.make_layout((32, 8))  # 列主序布局
    sw = cute.make_swizzle(3, 3, 4)  # 要求fast dimension 长度为 2^(4+3) = 128
    swizzled_layout = cute.make_composed_layout(sw, 0, layout_3d)
    render_layout_svg(layout_3d, "out/original_layout2.svg")
    render_swizzle_layout_svg(swizzled_layout, "out/swizzled_layout2.svg")


test_swizzle_layout2()


@cute.jit
def test_swizzle_layout3():
    """修正: fast dimension长度满足要求, swizzle生效"""
    """MBase=3: 8个元素组成一组. """
    layout_3d = cute.make_layout((128, 8))  # 列主序布局
    sw = cute.make_swizzle(3, 3, 4)  # 要求fast dimension 长度为 2^(4+3) = 128
    swizzled_layout = cute.make_composed_layout(sw, 0, layout_3d)
    render_layout_svg(layout_3d, "out/original_layout3.svg")
    render_swizzle_layout_svg(swizzled_layout, "out/swizzled_layout3.svg")


test_swizzle_layout3()
