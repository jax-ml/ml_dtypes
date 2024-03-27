#include <iostream>
#include "float8.h"

using float8_e5m2 = ml_dtypes::float8_internal::float8_e5m2;

float8_e5m2 ml_add(float8_e5m2 a, float8_e5m2 b){
    return a+b;
}

float8_e5m2 ml_multiply(float8_e5m2 a, float8_e5m2 b){
    return a*b;
}

int main()
{
    float8_e5m2 a(2.2);
    float8_e5m2 b(3.3);

    std::cout << "a: 0x" << a << std::endl;
    std::cout << "b: 0x" << b << std::endl;

    float8_e5m2 result = ml_add(a, b); // Storing the result of ml_add in a new variable
    int res = static_cast<int>(result);
    std::cout << "Addition: " << result << std::endl;

    result = ml_multiply(a, b); // Storing the result of ml_multiply in the same variable
    std::cout << "Multiplication: " << result << std::endl;
}

// Compilation and Execution steps:
// g++ -o output_file main.cc -I/work/vatsal/ml_dtypes/ml_dtypes/include -I/work/vatsal/ml_dtypes/third_party/eigen/
// ./output_file
