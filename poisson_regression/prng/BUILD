package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "prng",
    hdrs = ["prng.h"],
    deps = [
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/util:status_includes",
    ],
)

cc_library(
    name = "basic_rng",
    hdrs = ["basic_rng.h"],
    deps = [
        ":prng",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
        "@private_join_and_compute//private_join_and_compute/crypto:openssl_includes",
    ],
)

cc_test(
    name = "basic_rng_test",
    srcs = ["basic_rng_test.cc"],
    deps = [
        ":basic_rng",
        "@com_github_google_googletest//:gtest_main",
        "@com_google_absl//absl/numeric:int128",
        "@private_join_and_compute//private_join_and_compute/util:status_testing_includes",
    ],
)
