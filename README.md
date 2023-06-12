# Implementation of secure machine learning in C++

This repository contains the implementation of secure logistic regression based on the paper:

> Agarwal, A., Peceny, S., Raykova, M., Schoppmann, P., & Seth, K. (2022).
Communication-Efficient Secure Logistic Regression.
> ePrint Archive (2022). https://eprint.iacr.org/2022/866.pdf

This repository also contains the implementation of secure poisson regression based on the paper:

> Kelkar, M., Le, P.H., Raykova, M., & Seth, K. (2022).
Secure Poisson Regression.
> USENIX Security Symposium (USENIX Security
2022).

## Building/Running Tests

This repository requires Bazel. You can install Bazel by
following the instructions for your platform on the
[Bazel website](https://docs.bazel.build/versions/master/install.html).

Once you have installed Bazel you can clone this repository and run all tests
that are included by navigating into the root folder and running:

```bash
bazel test ... --check_visibility=false --test_timeout=300000 --dynamic_mode=off --cxxopt='-std=c++14' --host_cxxopt=-std=c++14 --test_output=streamed --cache_test_results=no
```

## Disclaimer

This is not an officially supported Google product. The code is provided as-is,
with no guarantees of correctness or security.
