import cutlass
from cutlass import cute


@cute.jit
def test_layout_properties():
    layout = cute.make_layout((4, 8), stride=(8, 1))
    print("layout rank:", cute.rank(layout))
    print("layout depth:", cute.depth(layout))


test_layout_properties()
