import ml_dtypes
from ml_dtypes import float8_e5m2

def ml_add(a, b):
    return a+b

def ml_multiply(a, b):
    return a*b

def main():
    a = float8_e5m2(30.5)
    b = float8_e5m2(20.5)
    print("Addition of two numbers using ml_add function: ", ml_add(a, b))
    print("Multiplication of two numbers using ml_multiply function: ", ml_multiply(a, b))
    return

if __name__ == "__main__":
    main()
