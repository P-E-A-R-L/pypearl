from typing import get_origin

def is_generic_type(typ) -> bool:
    return get_origin(typ) is not None

if __name__ == "__main__":
    print(is_generic_type(list[int]))
    print(is_generic_type(int))