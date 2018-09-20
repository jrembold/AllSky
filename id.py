"""
Script to generate unique 6 character
ids in an incrementing fashion.
"""

import string

def gen_id(prev_id:string):
    old = base36decode(prev_id)
    return base36encode(old+1)

def base36encode(number:int):
    if not isinstance(number, int):
        raise TypeError('Number must be an integer')

    if number < 0:
        raise ValueError('Number must be positive')

    alphabet = string.digits + string.ascii_uppercase
    base36 = ''

    if 0 <= number < len(alphabet):
        return alphabet[number].rjust(3,'0')

    while number != 0:
        number, i = divmod(number, len(alphabet))
        base36 = alphabet[i] + base36

    return base36.rjust(3,'0')

def base36decode(input:string):
    return int(input, 36)
