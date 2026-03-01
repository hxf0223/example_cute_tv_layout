import cutlass
from cutlass import cute


@cute.jit
def compose_verify():
    A = cute.make_layout((6, 2), stride=(8, 2))
    B = cute.make_layout((4, 3), stride=(3, 1))
    C = cute.composition(A, B)

    flat = cute.coalesce(B)
    for i in cutlass.range_constexpr(cute.size(flat)):
        print(f"C({i}) = {C(i)}, \tflat({i}) = {flat(i)}, \tA(flat({i})) = {A(flat(i))}")


compose_verify()
