import cutlass
from cutlass import cute
from cute_viz import render_layout_svg


@cute.jit
def test_coalesce_layout():
    layout = cute.make_layout((2, (1, 6)), stride=(1, (6, 2)))
    coalesced_layout = cute.coalesce(layout)

    assert cute.size(layout) == cute.size(coalesced_layout)
    assert cute.depth(coalesced_layout) <= 1, f"Depth is {cute.depth(coalesced_layout)}"
    for i in cutlass.range_constexpr(cute.size(layout)):
        original_value = layout(i)
        coalesced_value = coalesced_layout(i)
        assert original_value == coalesced_value, f"Value mismatch at index {i}"

    render_layout_svg(layout, "out/original_layout.svg")
    render_layout_svg(coalesced_layout, "out/coalesced_layout.svg")

    print("Original layout:", layout)
    print("Coalesced layout:", coalesced_layout)


test_coalesce_layout()


@cute.jit
def bymode_coalesce_example():
    layout = cute.make_layout((2, (1, 6)), stride=(1, (6, 2)))

    # Coalesce with mode-wise profile (1,1) = coalesce both modes
    result = cute.coalesce(layout, target_profile=(1, 1))

    render_layout_svg(layout, "out/bymode_original_layout.svg")
    render_layout_svg(result, "out/bymode_coalesced_layout.svg")

    # Print results
    print(">>> Original: ", layout)
    print(">>> Coalesced Result: ", result)


bymode_coalesce_example()
