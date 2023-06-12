# This package contains implementation of various Function Secret Sharing (FSS)
# gates as specified in https://eprint.iacr.org/2020/1392. The implementation
# uses the Distributed Point Function (as implemented in
# distributed_point_function.cc) as a central component.

load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Equality

cc_library(
    name = "equality",
    srcs = ["equality.cc"],
    hdrs = ["equality.h"],
    deps = [
        ":equality_cc_proto",
        "@distributed_point_functions//dpf:distributed_point_function",
        "@distributed_point_functions//dcf/fss_gates/prng:basic_rng",
        "@distributed_point_functions//dpf:distributed_point_function_cc_proto",
        "@distributed_point_functions//dpf:status_macros",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

proto_library(
    name = "equality_proto",
    srcs = ["equality.proto"],
    deps = [
        "@distributed_point_functions//dpf:distributed_point_function_proto",
    ],
)

cc_proto_library(
    name = "equality_cc_proto",
    deps = [":equality_proto"],
)

cc_test(
    name = "equality_test",
    srcs = ["equality_test.cc"],
    deps = [
        ":equality",
        ":equality_cc_proto",
        "@distributed_point_functions//dcf/fss_gates/prng:basic_rng",
        "@distributed_point_functions//dpf:distributed_point_function_cc_proto",
        "@distributed_point_functions//dpf:status_macros",
        "@distributed_point_functions//dpf/internal:status_matchers",
        "@distributed_point_functions//dpf/internal:value_type_helpers",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/numeric:int128",
    ],
)

# Comparison

cc_library(
    name = "comparison",
    srcs = ["comparison.cc"],
    hdrs = ["comparison.h"],
    deps = [
        ":comparison_cc_proto",
        "@distributed_point_functions//dcf:distributed_comparison_function",
        "@distributed_point_functions//dcf/fss_gates/prng:basic_rng",
        "@distributed_point_functions//dpf:distributed_point_function_cc_proto",
        "@distributed_point_functions//dpf:status_macros",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

proto_library(
    name = "comparison_proto",
    srcs = ["comparison.proto"],
    deps = [
        "@distributed_point_functions//dcf:distributed_comparison_function_proto",
        "@distributed_point_functions//dpf:distributed_point_function_proto",
    ],
)

cc_proto_library(
    name = "comparison_cc_proto",
    deps = [":comparison_proto"],
)

cc_test(
    name = "comparison_test",
    srcs = ["comparison_test.cc"],
    deps = [
        ":comparison",
        ":comparison_cc_proto",
        "@distributed_point_functions//dcf/fss_gates/prng:basic_rng",
        "@distributed_point_functions//dpf:distributed_point_function_cc_proto",
        "@distributed_point_functions//dpf:status_macros",
        "@distributed_point_functions//dpf/internal:status_matchers",
        "@distributed_point_functions//dpf/internal:value_type_helpers",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/numeric:int128",
    ],
)