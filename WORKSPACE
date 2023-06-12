"""WORKSPACE file for Private Join and Compute Secret Sharing MPC code."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# gflags needed for glog.
# https://github.com/gflags/gflags
http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "017e0a91531bfc45be9eaf07e4d8fed33c488b90b58509dbd2e33a33b2648ae6",
    strip_prefix = "gflags-a738fdf9338412f83ab3f26f31ac11ed3f3ec4bd",
    urls = [
        "https://github.com/gflags/gflags/archive/a738fdf9338412f83ab3f26f31ac11ed3f3ec4bd.zip",
    ],
)

# glog needed by SHELL
# https://github.com/google/glog
http_archive(
    name = "com_github_google_glog",
    sha256 = "0f91ee6cc1edc3b1c53a286382e69a37e5d172ce208b7e5b305be8770d8c21b1",
    strip_prefix = "glog-f545ff5e7d7f3df95f6e86c8cb987d9d9d4bd481",
    urls = [
        "https://github.com/google/glog/archive/f545ff5e7d7f3df95f6e86c8cb987d9d9d4bd481.zip",
    ],
)

# abseil-cpp
# https://github.com/abseil/abseil-cpp
http_archive(
    name = "com_google_absl",
    sha256 = "0b8b355781fff489ead0704984244256c145691c5fb9e27d632aaf9914293e74",
    strip_prefix = "abseil-cpp-b19ec98accca194511616f789c0a448c2b9d40e7",
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/b19ec98accca194511616f789c0a448c2b9d40e7.zip",
    ],
)



# Update platforms for Highway.
# TODO: Remove this once gRPC doesn't cause to fail the platforms definition anymore,
# or once `grpc_extra_deps` is not needed any more.
http_archive(
    name = "platforms",
    sha256 = "54095d9e2a2c6c0d4629c99fc80ecf4f74f93771aea658c872db888c1103bb93",
    strip_prefix = "platforms-fbd0d188dac49fbcab3d2876a2113507e6fc68e9",
    urls = ["https://github.com/bazelbuild/platforms/archive/fbd0d188dac49fbcab3d2876a2113507e6fc68e9.zip"],
)

# gRPC
# must be included separately, since we need to load transitive deps of grpc.
http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "3c7b67648602aaeeb8320f5c1f2787a59aed08cae7a2b1be85bb250eee04ca42",
    strip_prefix = "grpc-16a3ce51ff7c308c9e8798f1a40b10d172731fab",
    urls = [
        "https://github.com/grpc/grpc/archive/16a3ce51ff7c308c9e8798f1a40b10d172731fab.zip",
    ],
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()

# Private Join and Compute
http_archive(
    name = "private_join_and_compute",
    sha256 = "f58aa307697c06e9749c18eae16c1338bb548f999c2981bed192fc85493738d1",
    strip_prefix = "private-join-and-compute-505ba981d66c9e5e73e18cfa647b4685f74784cb",
    urls = ["https://github.com/google/private-join-and-compute/archive/505ba981d66c9e5e73e18cfa647b4685f74784cb.zip"],
)

load("@private_join_and_compute//bazel:pjc_deps.bzl", "pjc_deps")

pjc_deps()

# Distributed Point Functions, needed for DCF
http_archive(
    name = "distributed_point_functions",
    strip_prefix = "distributed_point_functions-04179c6106784a956f996dd3ba312f7441ddeb8f",
    urls = ["https://github.com/google/distributed_point_functions/archive/04179c6106784a956f996dd3ba312f7441ddeb8f.zip"],
)

#http_archive(
#    name = "distributed_point_functions",
#    sha256 = "0ef7d1ff5084e9dcb7e2914363926d2aa37d5f40d73b92cf38f8a806c631ca98",
#    strip_prefix = "distributed_point_functions-88c73a78cd61dacba6d8258f13d0f5dc5f1fb0d2",
#    urls = ["https://github.com/google/distributed_point_functions/archive/88c73a78cd61dacba6d8258f13d0f5dc5f1fb0d2.zip"],
#)

# needed for DPF
# IREE for cc_embed_data.
http_archive(
    name = "com_github_google_iree",
    sha256 = "aa369b29a5c45ae9d7aa8bf49ea1308221d1711277222f0755df6e0a575f6879",
    strip_prefix = "iree-7e6012468cbaafaaf30302748a2943771b40e2c3",
    urls = [
        "https://github.com/google/iree/archive/7e6012468cbaafaaf30302748a2943771b40e2c3.zip",
    ],
)

# rules_license needed for license() rule
# https://github.com/bazelbuild/rules_license
http_archive(
    name = "rules_license",
    sha256 = "6157e1e68378532d0241ecd15d3c45f6e5cfd98fc10846045509fb2a7cc9e381",
    urls = [
        "https://github.com/bazelbuild/rules_license/releases/download/0.0.4/rules_license-0.0.4.tar.gz",
    ],
)


# Highway for SIMD operations.
# https://github.com/google/highway
http_archive(
    name = "com_github_google_highway",
    sha256 = "83c252c7a9b8fcc36b9774778325c689e104365114a16adec0d536d47cb99e5f",
    strip_prefix = "highway-1c8250ed008f4ca22f2bb9edb6b75a73d9c587ff",
    urls = [
        "https://github.com/google/highway/archive/1c8250ed008f4ca22f2bb9edb6b75a73d9c587ff.zip",
    ],
)


# SHELL Encryption RLWE library.
http_archive(
    name = "shell_encryption",
    sha256 = "6b524ea06a88163f253ecd1e3f8368596d891ba78a92236c166aead90d7b5660",
    strip_prefix = "shell-encryption-cd1721d1ee9e20be16954f8161b0dbc051af4399",
    urls = [
        "https://github.com/google/shell-encryption/archive/cd1721d1ee9e20be16954f8161b0dbc051af4399.zip",
    ],
)

