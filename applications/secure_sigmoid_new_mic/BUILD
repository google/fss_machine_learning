load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

# This package implements the sigmoid function f(x) = 1 / (1 + e^-x).
package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "secure_sigmoid_new_mic",
    srcs = [
        "secure_sigmoid_new_mic.cc",
    ],
    hdrs = [
        "secure_sigmoid_new_mic.h",
    ],
    deps = [
        ":secure_sigmoid_new_mic_cc_proto",
        "//applications/secure_comparison",
        #        "//applications/secure_comparison:secure_comparison_cc_proto",
        "//applications/const_round_secure_comparison",
        "//applications/const_round_secure_comparison:const_round_secure_comparison_cc_proto",
        "//applications/secure_spline",
        "//applications/secure_spline:secure_spline_cc_proto",
        "//poisson_regression:beaver_triple_messages_cc_proto",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "//secret_sharing_mpc/gates:hadamard_product",
        "//secret_sharing_mpc/gates:polynomial",
        "//secret_sharing_mpc/gates:scalar_vector_product",
        "//secret_sharing_mpc/gates:vector_exponentiation",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@distributed_point_functions//dcf/fss_gates:multiple_interval_containment",
        "@distributed_point_functions//dcf/fss_gates:multiple_interval_containment_cc_proto",
        "@distributed_point_functions//dpf:status_macros",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

proto_library(
    name = "secure_sigmoid_new_mic_proto",
    srcs = ["secure_sigmoid_new_mic.proto"],
    deps = [
        #        "//applications/secure_comparison:secure_comparison_proto",
        "//applications/const_round_secure_comparison:const_round_secure_comparison_proto",
        "//applications/secure_spline:secure_spline_proto",
        "//poisson_regression:beaver_triple_messages_proto",
        "//poisson_regression:secure_exponentiation_proto",
        "//secret_sharing_mpc/gates:polynomial_messages_proto",
        "//secret_sharing_mpc/gates:powers_messages_proto",
    ],
)

cc_proto_library(
    name = "secure_sigmoid_new_mic_cc_proto",
    deps = [
        ":secure_sigmoid_new_mic_proto",
    ],
)

cc_test(
    name = "secure_sigmoid_new_mic_test",
    srcs = ["secure_sigmoid_new_mic_test.cc"],
    deps = [
        ":secure_sigmoid_new_mic",
        "//poisson_regression:fixed_point_element_util",
        "//poisson_regression:ring_utils",
        "@com_github_google_glog//:glog",
        "@com_github_google_googletest//:gtest_main",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

proto_library(
    name = "secure_sigmoid_new_mic_rpc_proto",
    srcs = [":secure_sigmoid_new_mic_rpc.proto"],
    deps = [
        ":secure_sigmoid_new_mic_proto",
    ],
)

cc_proto_library(
    name = "secure_sigmoid_new_mic_rpc_cc_proto",
    deps = [
        ":secure_sigmoid_new_mic_rpc_proto",
    ],
)

cc_grpc_library(
    name = "secure_sigmoid_new_mic_rpc_grpc_proto",
    srcs = [":secure_sigmoid_new_mic_rpc_proto"],
    grpc_only = True,
    deps = [":secure_sigmoid_new_mic_rpc_cc_proto"],
)

cc_library(
    name = "secure_sigmoid_new_mic_rpc_impl",
    srcs = ["secure_sigmoid_new_mic_rpc_impl.cc"],
    hdrs = [
        "secure_sigmoid_new_mic_rpc_impl.h",
    ],
    deps = [
        ":secure_sigmoid_new_mic",
        ":secure_sigmoid_new_mic_cc_proto",
        ":secure_sigmoid_new_mic_rpc_grpc_proto",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "//poisson_regression:ring_utils",
        "@com_github_grpc_grpc//:grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

cc_binary(
    name = "secure_sigmoid_new_mic_server",
    srcs = ["secure_sigmoid_new_mic_server.cc"],
    deps = [
        ":secure_sigmoid_new_mic",
        ":secure_sigmoid_new_mic_cc_proto",
        ":secure_sigmoid_new_mic_rpc_grpc_proto",
        ":secure_sigmoid_new_mic_rpc_impl",
        "//poisson_regression:fixed_point_element_util",
        "//poisson_regression:ring_utils",
        "@com_github_grpc_grpc//:grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base",
        "@com_github_google_glog//:glog",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

cc_binary(
    name = "secure_sigmoid_new_mic_client",
    srcs = ["secure_sigmoid_new_mic_client.cc"],
    deps = [
        ":secure_sigmoid_new_mic",
        ":secure_sigmoid_new_mic_cc_proto",
        ":secure_sigmoid_new_mic_rpc_grpc_proto",
        "//poisson_regression:fixed_point_element_util",
        "//poisson_regression:ring_utils",
        "//poisson_regression:secure_exponentiation",
        "@com_github_grpc_grpc//:grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_google_glog//:glog",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)
