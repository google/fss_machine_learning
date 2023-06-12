# Contains helper functions for computing on fixed-point elements
package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "fixed_point_arithmetic",
    srcs = [
        "fixed_point_arithmetic.cc",
    ],
    hdrs = [
        "fixed_point_arithmetic.h",
    ],
    deps = ["//poisson_regression:fixed_point_element_util"],
)

cc_test(
    name = "fixed_point_arithmetic_test",
    size = "small",
    srcs = [
        "fixed_point_arithmetic_test.cc",
    ],
    deps = [
        ":fixed_point_arithmetic",
        "//poisson_regression/prng:basic_rng",
        "@com_github_google_googletest//:gtest_main",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

cc_library(
    name = "matrix_arithmetic",
    srcs = [
        "matrix_arithmetic.cc",
    ],
    hdrs = [
        "matrix_arithmetic.h",
    ],
    deps = [
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

cc_test(
    name = "matrix_arithmetic_test",
    size = "small",
    srcs = [
        "matrix_arithmetic_test.cc",
    ],
    deps = [
        ":matrix_arithmetic",
        "@com_github_google_googletest//:gtest_main",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)
