# Contains implementations of various gates for secure multiparty computation performed over
# secret shares.

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "scalar_vector_product",
    srcs = [
        "scalar_vector_product.cc",
    ],
    hdrs = [
        "scalar_vector_product.h",
    ],
    deps = [
        "//poisson_regression:fixed_point_element_util",
        "@com_google_absl//absl/numeric:int128",
    ],
)

cc_test(
    name = "scalar_vector_product_test",
    size = "small",
    srcs = [
        "scalar_vector_product_test.cc",
    ],
    deps = [
        ":scalar_vector_product",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "//poisson_regression/prng:basic_rng",
        "@com_github_google_googletest//:gtest_main",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

cc_library(
    name = "vector_addition",
    srcs = [
        "vector_addition.cc",
    ],
    hdrs = [
        "vector_addition.h",
    ],
    deps = [
        "//poisson_regression:beaver_triple_utils",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

cc_test(
    name = "vector_addition_test",
    size = "small",
    srcs = [
        "vector_addition_test.cc",
    ],
    deps = [
        ":vector_addition",
        "@com_github_google_googletest//:gtest_main",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

cc_library(
    name = "vector_subtraction",
    srcs = [
        "vector_subtraction.cc",
    ],
    hdrs = [
        "vector_subtraction.h",
    ],
    deps = [
        "//poisson_regression:beaver_triple_utils",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

cc_test(
    name = "vector_subtraction_test",
    size = "small",
    srcs = [
        "vector_subtraction_test.cc",
    ],
    deps = [
        ":vector_subtraction",
        "@com_github_google_googletest//:gtest_main",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

proto_library(
    name = "correlated_beaver_triple_messages_proto",
    srcs = ["correlated_beaver_triple_messages.proto"],
)

cc_proto_library(
    name = "correlated_beaver_triple_messages_cc_proto",
    deps = [":correlated_beaver_triple_messages_proto"],
)

cc_library(
    name = "correlated_matrix_product",
    srcs = [
        "correlated_matrix_product.cc",
    ],
    hdrs = [
        "correlated_matrix_product.h",
    ],
    deps = [
        ":correlated_beaver_triple_messages_cc_proto",
        "//poisson_regression:beaver_triple_utils",
    ],
)

cc_test(
    name = "correlated_matrix_product_test",
    size = "small",
    srcs = [
        "correlated_matrix_product_test.cc",
    ],
    deps = [
        ":correlated_matrix_product",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "@com_github_google_googletest//:gtest_main",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

cc_library(
    name = "matrix_product",
    srcs = [
        "matrix_product.cc",
    ],
    hdrs = [
        "matrix_product.h",
    ],
    deps = [
        "//poisson_regression:beaver_triple_utils",
    ],
)

cc_test(
    name = "matrix_product_test",
    size = "small",
    srcs = [
        "matrix_product_test.cc",
    ],
    deps = [
        ":matrix_product",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "//secret_sharing_mpc/arithmetic:fixed_point_arithmetic",
        "@com_github_google_googletest//:gtest_main",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

cc_library(
    name = "hadamard_product",
    srcs = [
        "hadamard_product.cc",
    ],
    hdrs = [
        "hadamard_product.h",
    ],
    deps = [
        "//poisson_regression:beaver_triple_utils",
    ],
)

cc_test(
    name = "hadamard_product_test",
    size = "small",
    srcs = [
        "hadamard_product_test.cc",
    ],
    deps = [
        ":hadamard_product",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "@com_github_google_googletest//:gtest_main",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

cc_library(
    name = "vector_exponentiation",
    srcs = [
        "vector_exponentiation.cc",
    ],
    hdrs = [
        "vector_exponentiation.h",
    ],
    deps = [
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:secure_exponentiation",
        "//poisson_regression:secure_exponentiation_cc_proto",
    ],
)

cc_test(
    name = "vector_exponentiation_test",
    size = "small",
    srcs = ["vector_exponentiation_test.cc"],
    deps = [
        ":vector_exponentiation",
        "@com_github_google_googletest//:gtest_main",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

proto_library(
    name = "powers_messages_proto",
    srcs = ["powers_messages.proto"],
)

cc_proto_library(
    name = "powers_messages_cc_proto",
    deps = [":powers_messages_proto"],
)

cc_library(
    name = "powers",
    srcs = [
        "powers.cc",
    ],
    hdrs = [
        "hadamard_product.h",
        "powers.h",
    ],
    deps = [
        ":powers_messages_cc_proto",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
    ],
)

cc_test(
    name = "powers_test",
    size = "small",
    srcs = [
        "powers_test.cc",
    ],
    deps = [
        ":powers",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "@com_github_google_googletest//:gtest_main",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

proto_library(
    name = "polynomial_messages_proto",
    srcs = [
        "polynomial_messages.proto",
    ],
)

cc_proto_library(
    name = "polynomial_messages_cc_proto",
    deps = [":polynomial_messages_proto"],
)

cc_library(
    name = "polynomial",
    srcs = [
        "polynomial.cc",
    ],
    hdrs = [
        "hadamard_product.h",
        "polynomial.h",
        "powers.h",
        "scalar_vector_product.h",
    ],
    deps = [
        ":polynomial_messages_cc_proto",
        ":powers",
        ":powers_messages_cc_proto",
        ":scalar_vector_product",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
    ],
)

cc_test(
    name = "polynomial_test",
    size = "small",
    srcs = [
        "polynomial_test.cc",
    ],
    deps = [
        ":polynomial",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "@com_github_google_googletest//:gtest_main",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)
