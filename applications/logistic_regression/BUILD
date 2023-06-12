load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

# This package implements logistic regression gradient descent (SGD).

package(
    default_visibility = ["//visibility:public"],
)

proto_library(
    name = "gradient_descent_messages_proto",
    srcs = [":gradient_descent_messages.proto"],
    deps = [
        "//secret_sharing_mpc/gates:correlated_beaver_triple_messages_proto",
    ],
)

cc_proto_library(
    name = "gradient_descent_messages_cc_proto",
    deps = [":gradient_descent_messages_proto"],
)

cc_library(
    name = "gradient_descent",
    srcs = [
        "gradient_descent.cc",
        "gradient_descent_utils.cc",
    ],
    hdrs = [
        "gradient_descent.h",
        "gradient_descent_utils.h",
    ],
    deps = [
        ":gradient_descent_messages_cc_proto",
        "//applications/secure_sigmoid",
        "//applications/secure_sigmoid_new_mic",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "//poisson_regression:ring_utils",
        "//poisson_regression:secure_exponentiation",
        "//secret_sharing_mpc/arithmetic:matrix_arithmetic",
        "//secret_sharing_mpc/gates:correlated_matrix_product",
        "//secret_sharing_mpc/gates:scalar_vector_product",
        "//secret_sharing_mpc/gates:vector_addition",
        "//secret_sharing_mpc/gates:vector_subtraction",
        "@com_github_google_glog//:glog",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

cc_test(
    name = "gradient_descent_test",
    size = "small",
    srcs = [
        "gradient_descent_test.cc",
    ],
    data = [
        "balloonX.csv",
        "balloony.csv",
    ],
    deps = [
        ":gradient_descent",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:ring_utils",
        "//secret_sharing_mpc/gates:correlated_matrix_product",
        "//secret_sharing_mpc/gates:matrix_product",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

proto_library(
    name = "gradient_descent_rpc_proto",
    srcs = [":gradient_descent_rpc.proto"],
    deps = [
        ":gradient_descent_messages_proto",
        "//applications/secure_sigmoid:secure_sigmoid_proto",
        "//applications/secure_sigmoid_new_mic:secure_sigmoid_new_mic_proto",
    ],
)

cc_proto_library(
    name = "gradient_descent_rpc_cc_proto",
    deps = [
        ":gradient_descent_rpc_proto",
    ],
)

cc_grpc_library(
    name = "gradient_descent_rpc_grpc_proto",
    srcs = [":gradient_descent_rpc_proto"],
    grpc_only = True,
    deps = [":gradient_descent_rpc_cc_proto"],
)

cc_library(
    name = "gradient_descent_rpc_impl",
    srcs = ["gradient_descent_rpc_impl.cc"],
    hdrs = [
        "gradient_descent_rpc_impl.h",
    ],
    deps = [
        ":gradient_descent",
        ":gradient_descent_messages_cc_proto",
        ":gradient_descent_rpc_grpc_proto",
        "//applications/secure_sigmoid",
        "//applications/secure_sigmoid:secure_sigmoid_cc_proto",
        "//applications/secure_sigmoid_new_mic",
        "//applications/secure_sigmoid_new_mic:secure_sigmoid_new_mic_cc_proto",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "//poisson_regression:ring_utils",
        "//poisson_regression:secure_exponentiation",
        "//secret_sharing_mpc/arithmetic:matrix_arithmetic",
        "//secret_sharing_mpc/gates:correlated_matrix_product",
        "//secret_sharing_mpc/gates:scalar_vector_product",
        "//secret_sharing_mpc/gates:vector_addition",
        "//secret_sharing_mpc/gates:vector_subtraction",
        "@com_github_grpc_grpc//:grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

cc_binary(
    name = "gradient_descent_server",
    srcs = ["gradient_descent_server.cc"],
    deps = [
        ":gradient_descent",
        ":gradient_descent_messages_cc_proto",
        ":gradient_descent_rpc_grpc_proto",
        ":gradient_descent_rpc_impl",
        "//applications/secure_sigmoid",
        "//applications/secure_sigmoid:secure_sigmoid_cc_proto",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "//poisson_regression:ring_utils",
        "//poisson_regression:secure_exponentiation",
        "//secret_sharing_mpc/arithmetic:matrix_arithmetic",
        "//secret_sharing_mpc/gates:correlated_matrix_product",
        "//secret_sharing_mpc/gates:scalar_vector_product",
        "//secret_sharing_mpc/gates:vector_addition",
        "//secret_sharing_mpc/gates:vector_subtraction",
        "@com_github_grpc_grpc//:grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

cc_binary(
    name = "gradient_descent_client",
    srcs = ["gradient_descent_client.cc"],
    deps = [
        ":gradient_descent",
        ":gradient_descent_messages_cc_proto",
        ":gradient_descent_rpc_grpc_proto",
        "//applications/secure_sigmoid",
        "//applications/secure_sigmoid:secure_sigmoid_cc_proto",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "//poisson_regression:ring_utils",
        "//poisson_regression:secure_exponentiation",
        "//secret_sharing_mpc/arithmetic:matrix_arithmetic",
        "//secret_sharing_mpc/gates:correlated_matrix_product",
        "//secret_sharing_mpc/gates:scalar_vector_product",
        "//secret_sharing_mpc/gates:vector_addition",
        "//secret_sharing_mpc/gates:vector_subtraction",
        "@com_github_grpc_grpc//:grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

# dp version

proto_library(
    name = "gradient_descent_dp_messages_proto",
    srcs = [":gradient_descent_dp_messages.proto"],
    deps = [
        "//secret_sharing_mpc/gates:correlated_beaver_triple_messages_proto",
    ],
)

cc_proto_library(
    name = "gradient_descent_dp_messages_cc_proto",
    deps = [":gradient_descent_dp_messages_proto"],
)

proto_library(
    name = "gradient_descent_dp_rpc_proto",
    srcs = [":gradient_descent_dp_rpc.proto"],
    deps = [
        ":gradient_descent_dp_messages_proto",
        "//applications/secure_sigmoid:secure_sigmoid_proto",
        "//applications/secure_sigmoid_new_mic:secure_sigmoid_new_mic_proto",
    ],
)

cc_proto_library(
    name = "gradient_descent_dp_rpc_cc_proto",
    deps = [
        ":gradient_descent_dp_rpc_proto",
    ],
)

cc_grpc_library(
    name = "gradient_descent_dp_rpc_grpc_proto",
    srcs = [":gradient_descent_dp_rpc_proto"],
    grpc_only = True,
    deps = [":gradient_descent_dp_rpc_cc_proto"],
)

cc_library(
    name = "gradient_descent_dp",
    srcs = [
        "gradient_descent_dp.cc",
        "gradient_descent_dp_utils.cc",
    ],
    hdrs = [
        "gradient_descent_dp.h",
        "gradient_descent_dp_utils.h",
    ],
    deps = [
        ":gradient_descent_dp_messages_cc_proto",
        "//applications/secure_sigmoid",
        "//applications/secure_sigmoid_new_mic",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "//poisson_regression:ring_utils",
        "//poisson_regression:secure_exponentiation",
        "//secret_sharing_mpc/arithmetic:matrix_arithmetic",
        "//secret_sharing_mpc/gates:correlated_matrix_product",
        "//secret_sharing_mpc/gates:scalar_vector_product",
        "//secret_sharing_mpc/gates:vector_subtraction",
        "@com_github_google_glog//:glog",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

cc_test(
    name = "gradient_descent_dp_test",
    size = "small",
    srcs = [
        "gradient_descent_dp_test.cc",
    ],
    data = [
        "balloonX.csv",
        "balloony.csv",
    ],
    deps = [
        ":gradient_descent_dp",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:ring_utils",
        "//secret_sharing_mpc/gates:correlated_matrix_product",
        "//secret_sharing_mpc/gates:matrix_product",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

cc_library(
    name = "gradient_descent_dp_rpc_impl",
    srcs = ["gradient_descent_dp_rpc_impl.cc"],
    hdrs = [
        "gradient_descent_dp_rpc_impl.h",
    ],
    deps = [
        ":gradient_descent_dp",
        ":gradient_descent_dp_messages_cc_proto",
        ":gradient_descent_dp_rpc_grpc_proto",
        "//applications/secure_sigmoid",
        "//applications/secure_sigmoid:secure_sigmoid_cc_proto",
        "//applications/secure_sigmoid_new_mic",
        "//applications/secure_sigmoid_new_mic:secure_sigmoid_new_mic_cc_proto",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "//poisson_regression:ring_utils",
        "//poisson_regression:secure_exponentiation",
        "//secret_sharing_mpc/arithmetic:matrix_arithmetic",
        "//secret_sharing_mpc/gates:correlated_matrix_product",
        "//secret_sharing_mpc/gates:scalar_vector_product",
        "//secret_sharing_mpc/gates:vector_addition",
        "//secret_sharing_mpc/gates:vector_subtraction",
        "@com_github_grpc_grpc//:grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

# Gradient descent test for new mic gate

cc_test(
    name = "gradient_descent_new_mic_test",
    size = "small",
    srcs = [
        "gradient_descent_new_mic_test.cc",
    ],
    data = [
        "balloonX.csv",
        "balloony.csv",
    ],
    deps = [
        ":gradient_descent",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:ring_utils",
        "//secret_sharing_mpc/gates:correlated_matrix_product",
        "//secret_sharing_mpc/gates:matrix_product",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

cc_binary(
    name = "gradient_descent_dp_server",
    srcs = ["gradient_descent_dp_server.cc"],
    deps = [
        ":gradient_descent_dp",
        ":gradient_descent_dp_messages_cc_proto",
        ":gradient_descent_dp_rpc_grpc_proto",
        ":gradient_descent_dp_rpc_impl",
        "//applications/secure_sigmoid",
        "//applications/secure_sigmoid:secure_sigmoid_cc_proto",
        "//applications/secure_sigmoid_new_mic",
        "//applications/secure_sigmoid_new_mic:secure_sigmoid_new_mic_cc_proto",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "//poisson_regression:ring_utils",
        "//poisson_regression:secure_exponentiation",
        "//secret_sharing_mpc/arithmetic:matrix_arithmetic",
        "//secret_sharing_mpc/gates:correlated_matrix_product",
        "//secret_sharing_mpc/gates:scalar_vector_product",
        "//secret_sharing_mpc/gates:vector_addition",
        "//secret_sharing_mpc/gates:vector_subtraction",
        "@com_github_grpc_grpc//:grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

cc_binary(
    name = "gradient_descent_dp_client",
    srcs = ["gradient_descent_dp_client.cc"],
    deps = [
        ":gradient_descent_dp",
        ":gradient_descent_dp_messages_cc_proto",
        ":gradient_descent_dp_rpc_grpc_proto",
        "//applications/secure_sigmoid",
        "//applications/secure_sigmoid:secure_sigmoid_cc_proto",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "//poisson_regression:ring_utils",
        "//poisson_regression:secure_exponentiation",
        "//secret_sharing_mpc/arithmetic:matrix_arithmetic",
        "//secret_sharing_mpc/gates:correlated_matrix_product",
        "//secret_sharing_mpc/gates:scalar_vector_product",
        "//secret_sharing_mpc/gates:vector_addition",
        "//secret_sharing_mpc/gates:vector_subtraction",
        "@com_github_grpc_grpc//:grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

# Gradient descent dp test for new mic gate

cc_test(
    name = "gradient_descent_dp_new_mic_test",
    size = "small",
    srcs = [
        "gradient_descent_dp_new_mic_test.cc",
    ],
    data = [
        "balloonX.csv",
        "balloony.csv",
    ],
    deps = [
        ":gradient_descent_dp",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:ring_utils",
        "//secret_sharing_mpc/gates:correlated_matrix_product",
        "//secret_sharing_mpc/gates:matrix_product",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)
