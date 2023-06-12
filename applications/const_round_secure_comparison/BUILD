load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "const_round_secure_comparison",
    srcs = [
        "const_round_secure_comparison.cc",
    ],
    hdrs = [
        "const_round_secure_comparison.h",
    ],
    deps = [
        ":const_round_secure_comparison_cc_proto",
        "//fss_gates:comparison",
        "//fss_gates:equality",
        "//poisson_regression:beaver_triple_messages_cc_proto",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "//secret_sharing_mpc/gates:hadamard_product",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@distributed_point_functions//dcf:distributed_comparison_function",
        "@distributed_point_functions//dcf/fss_gates/prng:basic_rng",
        "@distributed_point_functions//dpf:distributed_point_function",
        "@distributed_point_functions//dpf:distributed_point_function_cc_proto",
        "@distributed_point_functions//dpf:status_macros",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

proto_library(
    name = "const_round_secure_comparison_proto",
    srcs = ["const_round_secure_comparison.proto"],
    deps = [
        "//poisson_regression:beaver_triple_messages_proto",
    ],
)

cc_proto_library(
    name = "const_round_secure_comparison_cc_proto",
    deps = [
        ":const_round_secure_comparison_proto",
    ],
)

cc_test(
    name = "const_round_secure_comparison_test",
    srcs = ["const_round_secure_comparison_test.cc"],
    deps = [
        ":const_round_secure_comparison",
        "//applications/secure_comparison",
        "//poisson_regression:fixed_point_element_util",
        "//poisson_regression:ring_utils",
        "@com_github_google_glog//:glog",
        "@com_github_google_googletest//:gtest_main",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)


proto_library(
    name = "const_round_secure_comparison_rpc_proto",
    srcs = [":const_round_secure_comparison_rpc.proto"],
    deps = [
        ":const_round_secure_comparison_proto",
				"//poisson_regression:beaver_triple_messages_proto"
    ],
)

cc_proto_library(
    name = "const_round_secure_comparison_rpc_cc_proto",
    deps = [
        ":const_round_secure_comparison_rpc_proto",
		]
)

cc_grpc_library(
    name = "const_round_secure_comparison_rpc_grpc_proto",
    srcs = [":const_round_secure_comparison_rpc_proto"],
    grpc_only = True,
    deps = [":const_round_secure_comparison_rpc_cc_proto"],
)

cc_library(
    name = "const_round_secure_comparison_rpc_impl",
    srcs = ["const_round_secure_comparison_rpc_impl.cc"],
    hdrs = [
        "const_round_secure_comparison_rpc_impl.h",
    ],
    deps = [
        ":const_round_secure_comparison",
        ":const_round_secure_comparison_cc_proto",
        ":const_round_secure_comparison_rpc_grpc_proto",
				"//applications/secure_comparison:secure_comparison",
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
    name = "const_round_secure_comparison_server",
    srcs = ["const_round_secure_comparison_server.cc"],
    deps = [
        ":const_round_secure_comparison",
        ":const_round_secure_comparison_cc_proto",
        ":const_round_secure_comparison_rpc_grpc_proto",
        ":const_round_secure_comparison_rpc_impl",
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
    name = "const_round_secure_comparison_client",
    srcs = ["const_round_secure_comparison_client.cc"],
    deps = [
        ":const_round_secure_comparison",
				"//applications/secure_comparison:secure_comparison",
        ":const_round_secure_comparison_cc_proto",
        ":const_round_secure_comparison_rpc_grpc_proto",
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