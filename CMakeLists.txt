cmake_minimum_required(VERSION 3.15...3.29)
project(ml_dtypes LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NumPy_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

# Add Eigen as a submodule. This will run Eigen's configuration step and generate headers.
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/eigen EXCLUDE_FROM_ALL)

Python_add_library(_ml_dtypes_ext MODULE WITH_SOABI
    ml_dtypes/_src/dtypes.cc
    ml_dtypes/_src/numpy.cc
)

target_include_directories(_ml_dtypes_ext PRIVATE
    "${NumPy_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(_ml_dtypes_ext PRIVATE Eigen3::Eigen)

if(MSVC)
    target_compile_options(_ml_dtypes_ext PRIVATE
        /std:c++17
        /DEIGEN_MPL2_ONLY
        /EHsc
        /bigobj
    )
else()
    target_compile_options(_ml_dtypes_ext PRIVATE
        -std=c++17
        -DEIGEN_MPL2_ONLY
        -fvisibility=hidden
        # -ftrapping-math is necessary because NumPy looks at floating point
        # exception state to determine whether to emit, e.g., invalid value
        # warnings. Without this setting, on Mac ARM we see spurious "invalid
        # value" warnings when running the tests.
        -ftrapping-math
    )
endif()

set_target_properties(_ml_dtypes_ext PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

install(TARGETS _ml_dtypes_ext DESTINATION ml_dtypes)
