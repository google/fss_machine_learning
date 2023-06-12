/*
 * Copyright 2020 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GOOGLE_CODE_FSS_GATES_COMPARISON_H_
#define GOOGLE_CODE_FSS_GATES_COMPARISON_H_

// TODO move to idpf repo
// TODO run with --check_visibility=false until moved

#include "absl/numeric/int128.h"
#include "absl/status/statusor.h"
#include "dcf/distributed_comparison_function.h"
#include "dpf/status_macros.h"
#include "fss_gates/comparison.pb.h"

namespace distributed_point_functions {
namespace fss_gates {

// Implements the comparison gate as specified in
// https://eprint.iacr.org/2020/1392. Such a gate is specified by
// two input groups Z_{2 ^ n} where n is the bit length and output group Z_{2}.
// The Key generation procedure
// produces two keys, k_0 and k_1, corresponding to Party 0 and Party 1
// respectively. Evaluating each key on any point `x, y` in the input groups results
// in an additive secret share of `1`, if
// `x<y`, and 0 otherwise.

class ComparisonGate {
 public:
  // Factory method : creates and returns a ComparisonGate
  // Initialized with appropriate parameters.
  static absl::StatusOr<std::unique_ptr<ComparisonGate>>
  Create(const CmpParameters& cmp_parameters);

  // ComparisonGate is neither copyable nor movable.
  ComparisonGate(const ComparisonGate&) =
  delete;
  ComparisonGate& operator=(
      const ComparisonGate&) = delete;

  // This method generates Comparison Gate a pair of keys
  // using `r_in0` and `r_in1` and `r_out` as the input masks and output mask respectively.
  // The implementation of this method is identical to Gen procedure specified
  // in https://eprint.iacr.org/2020/1392. Note that although the
  // datatype of r_in0, r_in1 and r_out are absl::uint128, but they will be interpreted
  // as an element in the input and output group Z_{2 ^ n}, Z_{2} respectively. This
  // reinterpretion in the group is achieved simply by taking mod of r_in and
  // r_out with the size of group i.e. 2^{log_group_size}.
  absl::StatusOr<std::pair<CmpKey, CmpKey>> Gen(
      absl::uint128 r_in0, absl::uint128 r_in1, absl::uint128 r_out);

  // This method evaluates the Comparison Gate key k
  // on input domain point s`x,y`. The output is returned as a 128 bit string
  // and needs to be interpreted as an element in the output group Z_{2}.
  absl::StatusOr<absl::uint128> Eval(absl::uint128 b, CmpKey k, absl::uint128 x, absl::uint128 y);

  // same as eval but batched (highly optimized dcf)
  absl::StatusOr<std::vector<absl::uint128>>
  BatchEval(absl::uint128 b, const std::vector<CmpKey> &ks,
                            std::vector<uint64_t> &xs, std::vector<uint64_t> &ys);

private:
  // Parameters needed for specifying an Comparison Gate.
  const CmpParameters cmp_parameters_;

  // Private constructor, called by `Create`
  ComparisonGate(
      CmpParameters cmp_parameters,
      std::unique_ptr<DistributedComparisonFunction> dcf);

  // Pointer to a Distributed Comparison Function which will be internally
  // invoked by Gen and Eval.
  std::unique_ptr<DistributedComparisonFunction> dcf_;
};

}  // namespace fss_gates
}  // namespace distributed_point_functions

#endif //GOOGLE_CODE_FSS_GATES_COMPARISON_H_
