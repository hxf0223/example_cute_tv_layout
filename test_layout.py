from cutlass import cute
from cute_viz import render_layout_svg, display_layout


@cute.jit
def main():
    # Create and render a layout to file
    layout = cute.make_layout((8, 8), stride=(8, 1))
    render_layout_svg(layout, "out/layout.svg")

    # Or display directly in Jupyter notebook
    display_layout(layout)

    # For hierarchical layouts, you can choose between flattened (default) or nested visualization
    hierarchical_layout = cute.make_layout(((2, 2), (3, 4)), stride=((1, 6), (2, 12)))
    render_layout_svg(hierarchical_layout, "out/layout_flat.svg", flatten_hierarchical=True)  # Flattened (default)
    render_layout_svg(hierarchical_layout, "out/layout_nested.svg", flatten_hierarchical=False)  # With tile boundaries


main()
