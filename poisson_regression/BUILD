

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "fixed_point_element_util",
    srcs = [
        "fixed_point_element.cc",
        "fixed_point_element_factory.cc",
    ],
    hdrs = [
        "fixed_point_element.h",
        "fixed_point_element_factory.h",
    ],
    deps = [
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

cc_test(
    name = "fixed_point_element_util_test",
    size = "small",
    srcs = [
        "fixed_point_element_factory_test.cc",
        "fixed_point_element_test.cc",
    ],
    deps = [
        ":fixed_point_element_util",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

proto_library(
    name = "beaver_triple_messages_proto",
    srcs = ["beaver_triple_messages.proto"],
)

cc_proto_library(
    name = "beaver_triple_messages_cc_proto",
    deps = [":beaver_triple_messages_proto"],
)

cc_library(
    name = "beaver_triple_utils",
    srcs = [
        "beaver_triple_utils.cc",
        "ring_arithmetic_utils.cc",
    ],
    hdrs = [
        "beaver_triple.h",
        "beaver_triple_utils.h",
        "ring_arithmetic_utils.h",
    ],
    deps = [
        ":beaver_triple_messages_cc_proto",
        "//poisson_regression/prng:basic_rng",
        "@com_google_absl//absl/numeric:int128",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

cc_test(
    name = "beaver_triple_test",
    size = "small",
    srcs = [
        "beaver_triple_generator_test.cc",
        "beaver_triple_test.cc",
        "ring_arithmetic_utils_test.cc",
    ],
    deps = [
        ":beaver_triple_utils",
        "//poisson_regression/prng:basic_rng",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

cc_test(
    name = "beaver_protocol_test",
    size = "small",
    srcs = [
        "beaver_protocol_test.cc",
    ],
    deps = [
        ":beaver_triple_utils",
        ":fixed_point_element_util",
        "//poisson_regression/prng:basic_rng",
        "@com_github_google_googletest//:gtest_main",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

cc_library(
    name = "ring_utils",
    srcs = ["ring_arithmetic_utils.cc"],
    hdrs = ["ring_arithmetic_utils.h"],
    deps = [
        "@com_google_absl//absl/numeric:int128",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

cc_library(
    name = "secure_exponentiation",
    srcs = [
        "secure_exponentiation.cc",
    ],
    hdrs = [
        "secure_exponentiation.h",
    ],
    deps = [
        ":fixed_point_element_util",
        ":ring_utils",
        ":secure_exponentiation_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

proto_library(
    name = "secure_exponentiation_proto",
    srcs = ["secure_exponentiation.proto"],
)

cc_proto_library(
    name = "secure_exponentiation_cc_proto",
    deps = [":secure_exponentiation_proto"],
)

cc_test(
    name = "secure_exponentiation_test",
    size = "small",
    srcs = ["secure_exponentiation_test.cc"],
    deps = [
        ":beaver_triple_utils",
        ":fixed_point_element_util",
        ":secure_exponentiation",
        "//poisson_regression/prng:basic_rng",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/crypto:bn_util",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)

proto_library(
    name = "gradient_descent_messages_proto",
    srcs = [":gradient_descent_messages.proto"],
    deps = [
        ":beaver_triple_messages_proto",
        ":secure_exponentiation_proto",
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
        ":beaver_triple_utils",
        ":fixed_point_element_util",
        ":gradient_descent_messages_cc_proto",
        ":ring_utils",
        ":secure_exponentiation",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

cc_test(
    name = "gradient_descent_test",
    size = "small",
    srcs = [
        "gradient_descent_test.cc",
    ],
    deps = [
        ":beaver_triple_utils",
        ":gradient_descent",
        ":ring_utils",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/memory",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)
