import cutlass
from cutlass import cute
from cute_viz import render_layout_svg, display_layout


@cute.jit
def test_composition_extract_subtile():
    layout_2d = cute.make_layout((10, 2), stride=(16, 4))
    tiler = cute.make_layout((5, 4), stride=(1, 5))
    subtile = cute.composition(layout_2d, tiler)
    render_layout_svg(layout_2d, "out/layout_2d.svg")
    render_layout_svg(tiler, "out/tiler.svg")
    render_layout_svg(subtile, "out/extracted_subtile.svg")


test_composition_extract_subtile()
