import numpy as np


def decode(data: str, shift):
    return "".join(map(lambda ch: chr((ord(ch) + shift - 33) % 100 + 33), data))


def decode_for_all():
    for i in range(1010):
        print(decode(base_data, i))
        print("\n\n")


def numbers(data: str):
    return list(map(lambda ch: ord(ch), data))


base_data = """C)+;2@-!#(;O!=]AR'TWMVPHBTMG>2*%W(NL#A)GL`7<.S*3:OKVC(+I0VU[F].?+3]:;@YO4.Z<\!J]AM6=6@V)M@1]]K>&VWBP=06<A;FH)G#4?.L&EN;]8O/)&4H3R,,:NBXK.&020^M(F8SL*-<A<B^5+J0'$D)4X^1S!OUC*NZOL?T5P.:F:;:60UI9^]`M<=8-E_6M#ETW&GX*9N_8RXD5R`=A;]DN8BIL2<9RC6HI]2P:;##/)9V18JP*-N!IHM7UM1/]?BXZ%+FA"8G/\WQUG-:7@]TV:)YQZ\['R%9+!9]>_M#@[,O:#X!``#_M_P,4$L#!!0!@`(````(0!\.Y<Y(@$``+D#```<``@!=V]R9"]?<F5L<R]DM;V-U;65N="YX;6PN<F5L<R"B!`$HH
"""

if __name__ == '__main__':
    # numbers = numbers(base_data)[:-1]
    # print(chr(95))
    # print(numbers)
    # print(np.max(numbers))
    # print(np.min(numbers))
    decode_for_all()
