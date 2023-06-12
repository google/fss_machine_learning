# This package contains implementation of secure 2 party Spline Evaluation
# using Function Secret Sharing (https://eprint.iacr.org/2020/1392) and
# secret sharing mpc


package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "secure_spline",
    srcs = ["secure_spline.cc"],
    hdrs = ["secure_spline.h"],
    deps = [
        ":secure_spline_cc_proto",
        "//poisson_regression:beaver_triple_messages_cc_proto",
        "//poisson_regression:beaver_triple_utils",
        "//poisson_regression:fixed_point_element_util",
        "//secret_sharing_mpc/gates:hadamard_product",
        "//secret_sharing_mpc/gates:scalar_vector_product",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@distributed_point_functions//dcf/fss_gates:multiple_interval_containment",
        "@distributed_point_functions//dcf/fss_gates:multiple_interval_containment_cc_proto",
        "@distributed_point_functions//dpf:status_macros",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

proto_library(
    name = "secure_spline_proto",
    srcs = ["secure_spline.proto"],
    deps = [
        "//poisson_regression:beaver_triple_messages_proto",
    ],
)

cc_proto_library(
    name = "secure_spline_cc_proto",
    deps = [
        ":secure_spline_proto",
    ],
)

cc_test(
    name = "secure_spline_test",
    srcs = ["secure_spline_test.cc"],
    deps = [
        ":secure_spline",
        "//poisson_regression:fixed_point_element_util",
        "//poisson_regression:ring_utils",
        "@com_github_google_glog//:glog",
        "@com_github_google_googletest//:gtest_main",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)
