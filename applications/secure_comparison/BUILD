load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

# This package implements the secure comparison function x < y.
package(
    default_visibility = ["//visibility:public"],
)

proto_library(
    name = "secure_comparison_proto",
    srcs = [
        "secure_comparison.proto",
    ],
)

cc_proto_library(
    name = "secure_comparison_cc_proto",
    deps = [
        ":secure_comparison_proto",
    ],
)

cc_library(
    name = "secure_comparison",
    srcs = [
        "secure_comparison.cc",
    ],
    hdrs = [
        "secure_comparison.h",
    ],
    deps = [
        ":secure_comparison_cc_proto",
        "//fss_gates:comparison",
        "//fss_gates:equality",
        "//secret_sharing_mpc/gates:hadamard_product",
        "//secret_sharing_mpc/gates:vector_addition",
        "//secret_sharing_mpc/gates:vector_subtraction",
    ],
)

cc_test(
    name = "secure_comparison_test",
    size = "small",
    srcs = [
        "secure_comparison_test.cc",
    ],
    deps = [
        ":secure_comparison",
        "//poisson_regression:fixed_point_element_util",
        "@com_github_google_googletest//:gtest_main",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

proto_library(
    name = "secure_comparison_rpc_proto",
    srcs = [":secure_comparison_rpc.proto"],
    deps = [
        ":secure_comparison_proto",
				"//poisson_regression:beaver_triple_messages_proto"
    ],
)

cc_proto_library(
    name = "secure_comparison_rpc_cc_proto",
    deps = [
        ":secure_comparison_rpc_proto",
		]
)

cc_grpc_library(
    name = "secure_comparison_rpc_grpc_proto",
    srcs = [":secure_comparison_rpc_proto"],
    grpc_only = True,
    deps = [":secure_comparison_rpc_cc_proto"],
)

cc_library(
    name = "secure_comparison_rpc_impl",
    srcs = ["secure_comparison_rpc_impl.cc"],
    hdrs = [
        "secure_comparison_rpc_impl.h",
    ],
    deps = [
        ":secure_comparison",
        ":secure_comparison_cc_proto",
        ":secure_comparison_rpc_grpc_proto",
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
    name = "secure_comparison_server",
    srcs = ["secure_comparison_server.cc"],
    deps = [
        ":secure_comparison",
        ":secure_comparison_cc_proto",
        ":secure_comparison_rpc_grpc_proto",
        ":secure_comparison_rpc_impl",
        "//poisson_regression:fixed_point_element_util",
        "//poisson_regression:ring_utils",
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

cc_binary(
    name = "secure_comparison_client",
    srcs = ["secure_comparison_client.cc"],
    deps = [
        ":secure_comparison",
        ":secure_comparison_cc_proto",
        ":secure_comparison_rpc_grpc_proto",
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
