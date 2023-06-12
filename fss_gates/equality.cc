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

#include "equality.h"

#include <utility>

#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "dpf/distributed_point_function.h"
#include "fss_gates/equality.pb.h"
#include "dcf/fss_gates/prng/basic_rng.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/status_macros.h"

namespace distributed_point_functions {
namespace fss_gates {

absl::StatusOr<std::unique_ptr<EqualityGate>>
EqualityGate::Create(const EqParameters& eq_parameters) {
  // Declaring the parameters for a Distributed Point Function (DPF).
  DpfParameters dpf_parameters;

  // Return error if log_group_size is not between 0 and 127.
  if (eq_parameters.log_group_size() < 0 ||
      eq_parameters.log_group_size() > 127) {
    return absl::InvalidArgumentError(
        "log_group_size should be in > 0 and < 128");
  }

  // Setting the `log_domain_size` of the DPF to be same as the
  // `log_group_size` of the Equality Gate.
  dpf_parameters.set_log_domain_size(
      eq_parameters.log_group_size());

  // Setting the output ValueType of the DPF so that it can store 128 bit
  // integers.
  *(dpf_parameters.mutable_value_type()) =
      ToValueType<absl::uint128>();

  // Creating a DPF with appropriate parameters.
  DPF_ASSIGN_OR_RETURN(std::unique_ptr<DistributedPointFunction> dpf,
                       DistributedPointFunction::Create(dpf_parameters));

  return absl::WrapUnique(
      new EqualityGate(eq_parameters, std::move(dpf)));
}

EqualityGate::EqualityGate(
    EqParameters eq_parameters,
    std::unique_ptr<DistributedPointFunction> dpf)
    : eq_parameters_(std::move(eq_parameters)), dpf_(std::move(dpf)) {}

absl::StatusOr<std::pair<EqKey, EqKey>> EqualityGate::Gen(
    absl::uint128 r_in0, absl::uint128 r_in1,
    absl::uint128 r_out) {

  // Setting N = 2 ^ log_group_size.
  absl::uint128 N = absl::uint128(1) << eq_parameters_.log_group_size();

  // Setting M = output group size
  absl::uint128 M = (eq_parameters_.has_output_group_modulus()) ? absl::uint128(eq_parameters_.output_group_modulus()) : absl::uint128(2);


  // Checking whether r_in is a group element.
  if (r_in0 < 0 || r_in0 >= N) {
    return absl::InvalidArgumentError(
        "Input mask should be between 0 and 2^log_group_size");
  }
  if (r_in1 < 0 || r_in1 >= N) {
    return absl::InvalidArgumentError(
        "Input mask should be between 0 and 2^log_group_size");
  }

  // Checking whether each element of r_out is a group element.
  if (r_out < 0 || r_out >= absl::uint128(M)) {
    return absl::InvalidArgumentError(
        "Output mask not a group element");
  }

  absl::uint128 alpha = (r_in0 - r_in1) % N;
  absl::uint128 beta = 1;

  DpfKey key_0, key_1;
  DPF_ASSIGN_OR_RETURN(std::tie(key_0, key_1),
                       this->dpf_->GenerateKeys(alpha, beta));

  EqKey k0, k1;

  *(k0.mutable_dpfkey()) = key_0;
  *(k1.mutable_dpfkey()) = key_1;

  // Append to each key the corresponding share of output key r_out
  // 1. secret share r_out: r_out0, r_out1
  // 2. k_0 <- k0 || r_out0
  //    k_1 <- k1 || r_out1

  // First secret share
  const absl::string_view kSampleSeed = absl::string_view();
  DPF_ASSIGN_OR_RETURN(
      auto rng, distributed_point_functions::BasicRng::Create(kSampleSeed));
  DPF_ASSIGN_OR_RETURN(absl::uint128 r_out0, rng->Rand128());

    r_out0 = r_out0 % M;

    absl::uint128 r_out1 = (r_out - r_out0) % M;

    // Now append them

    Value_Integer* k0_output_mask_share = k0.mutable_output_mask_share();

    k0_output_mask_share->mutable_value_uint128()->set_high(
        absl::Uint128High64(r_out0));
    k0_output_mask_share->mutable_value_uint128()->set_low(
        absl::Uint128Low64(r_out0));

    Value_Integer* k1_output_mask_share = k1.mutable_output_mask_share();

    k1_output_mask_share->mutable_value_uint128()->set_high(
        absl::Uint128High64(r_out1));
    k1_output_mask_share->mutable_value_uint128()->set_low(
        absl::Uint128Low64(r_out1));

  return std::pair<EqKey, EqKey>(k0, k1);
}

// k is a share between the two parties
// x,y are masked inputs and known to both parties
absl::StatusOr<absl::uint128>
EqualityGate::Eval(EqKey k, absl::uint128 x, absl::uint128 y) {
  // Setting N = 2 ^ log_group_size
  absl::uint128 N = absl::uint128(1) << eq_parameters_.log_group_size();

    // Setting M = output group size
    absl::uint128 M = (eq_parameters_.has_output_group_modulus()) ? absl::uint128(eq_parameters_.output_group_modulus()) : absl::uint128(2);


        // Checking whether x is a group element
  if (x < 0 || x >= N) {
    return absl::InvalidArgumentError(
        "Masked input should be between 0 and 2^log_group_size");
  }

  // Checking whether y is a group element
  if (y < 0 || y >= N) {
    return absl::InvalidArgumentError(
        "Masked input should be between 0 and 2^log_group_size");
  }

  absl::uint128 res; // share of res

  // Compute share of unmasked result
  DPF_ASSIGN_OR_RETURN(std::vector<absl::uint128> res_vec,
      this->dpf_->template EvaluateAt<absl::uint128>(
          k.dpfkey(), 0, {(x - y) % N}));

  res = res_vec[0] % M;

  // Retrieve mask share from EqKey and Add mask share to res

  absl::uint128 r_out =
      absl::MakeUint128(k.output_mask_share().value_uint128().high(),
                          k.output_mask_share().value_uint128().low());
  //std::cerr << "r_out retrieved from key: " >>r_out.int() << std::endl;
  res = (res + r_out) % M;

  return res;
}

}  // namespace fss_gates
}  // namespace distributed_point_functions