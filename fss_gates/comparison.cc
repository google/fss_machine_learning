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

#include "comparison.h"

#include <cmath>
#include <utility>

#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "dcf/distributed_comparison_function.h"
#include "dpf/distributed_point_function.h"
//#include "dcf/fss_gates/equality.pb.h"
#include "fss_gates/comparison.pb.h"
#include "dcf/fss_gates/prng/basic_rng.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/status_macros.h"

namespace distributed_point_functions {
namespace fss_gates {

    absl::StatusOr<std::unique_ptr<ComparisonGate>>
    ComparisonGate::Create(const CmpParameters &cmp_parameters) {
        // Declaring the parameters for a Distributed Comparison Function (DCF).
        DcfParameters dcf_parameters;

        // Return error if log_group_size is not between 0 and 127.
        if (cmp_parameters.log_group_size() < 0 ||
            cmp_parameters.log_group_size() > 127) {
            return absl::InvalidArgumentError(
                    "log_group_size should be in > 0 and < 128");
        }

        // Setting the `log_domain_size` of the DCF to be same as the
        // `log_group_size` of the Multiple Interval Containment Gate.
        dcf_parameters.mutable_parameters()->set_log_domain_size(
                cmp_parameters.log_group_size());

        // Setting the output ValueType of the DCF so that it can store 128 bit
        // integers.
        *(dcf_parameters.mutable_parameters()->mutable_value_type()) =
                ToValueType<absl::uint128>();


        // Creating a DCF with appropriate parameters.
        DPF_ASSIGN_OR_RETURN(std::unique_ptr<DistributedComparisonFunction> dcf,
                             DistributedComparisonFunction::Create(dcf_parameters));

        return absl::WrapUnique(
                new ComparisonGate(cmp_parameters, std::move(dcf)));
    }

    ComparisonGate::ComparisonGate(
            CmpParameters cmp_parameters,
            std::unique_ptr<DistributedComparisonFunction> dcf)
            : cmp_parameters_(std::move(cmp_parameters)), dcf_(std::move(dcf)) {}

    absl::StatusOr<std::pair<CmpKey, CmpKey>> ComparisonGate::Gen(
            absl::uint128 r_in0, absl::uint128 r_in1,
            absl::uint128 r_out) {

        // Setting N = 2 ^ log_group_size.
        absl::uint128 N = absl::uint128(1) << cmp_parameters_.log_group_size();

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
        if (r_out < 0 || r_out >= absl::uint128(2)) {
            return absl::InvalidArgumentError(
                    "Output mask should be 0 or 1");
        }

        absl::uint128 y = (N - ((r_in0 - r_in1) % N)) % N;

        // 2 ^ (log_group_size - 1)
        absl::uint128 h = absl::uint128(1) << (cmp_parameters_.log_group_size() - 1);
        // h - 1
        absl::uint128 h_minus_1 = h - 1;

        absl::uint128 y_msb = (y & h) >> (cmp_parameters_.log_group_size() - 1);
        absl::uint128 y_no_msb = y & h_minus_1;

        absl::uint128 alpha = y_no_msb;

        absl::uint128 beta_one = y_msb ^ 1;
        absl::uint128 beta_two = y_msb;

        DcfKey key_0, key_1;
        DPF_ASSIGN_OR_RETURN(std::tie(key_0, key_1),
                             this->dcf_->GenerateKeys(alpha, (beta_one ^ beta_two)));

        CmpKey k0, k1;

        *(k0.mutable_dcfkey()) = key_0;
        *(k1.mutable_dcfkey()) = key_1;


        // Append to each key the corresponding share of output key r_out
        // 1. secret share r_out: r_out0, r_out1
        // 2. k_0 <- k0 || r_out0
        //    k_1 <- k1 || r_out1

        // First secret share
        const absl::string_view kSampleSeed = absl::string_view();
        DPF_ASSIGN_OR_RETURN(
                auto rng, distributed_point_functions::BasicRng::Create(kSampleSeed));
        DPF_ASSIGN_OR_RETURN(absl::uint128 r_out0, rng->Rand128());

        r_out0 = r_out0 % absl::uint128(2);

        //absl::uint128 r_out1 = (r_out - r_out0) % absl::uint128(2);
        absl::uint128 r_out1 = r_out ^ r_out0;

        // Now append them

        Value_Integer *k0_output_mask_share = k0.mutable_output_mask_share();

        k0_output_mask_share->mutable_value_uint128()->set_high(
                absl::Uint128High64(r_out0));
        k0_output_mask_share->mutable_value_uint128()->set_low(
                absl::Uint128Low64(r_out0));

        Value_Integer *k1_output_mask_share = k1.mutable_output_mask_share();

        k1_output_mask_share->mutable_value_uint128()->set_high(
                absl::Uint128High64(r_out1));
        k1_output_mask_share->mutable_value_uint128()->set_low(
                absl::Uint128Low64(r_out1));

        // Now append to the key shares of beta_two

        Value_Integer *k0_beta_two_share = k0.mutable_beta_two_share();

        k0_beta_two_share->mutable_value_uint128()->set_high(
                absl::Uint128High64(absl::uint128(0)));
        k0_beta_two_share->mutable_value_uint128()->set_low(
                absl::Uint128Low64(absl::uint128(0)));

        Value_Integer *k1_beta_two_share = k1.mutable_beta_two_share();

        k1_beta_two_share->mutable_value_uint128()->set_high(
                absl::Uint128High64(beta_two));
        k1_beta_two_share->mutable_value_uint128()->set_low(
                absl::Uint128Low64(beta_two));

        return std::pair<CmpKey, CmpKey>(k0, k1);
    }

// k is a share between the two parties
// x,y are masked inputs and known to both parties
    absl::StatusOr<absl::uint128>
    ComparisonGate::Eval(absl::uint128 b, CmpKey k, absl::uint128 x, absl::uint128 y) {

        // Setting N = 2 ^ log_group_size
        absl::uint128 N = absl::uint128(1) << cmp_parameters_.log_group_size();

        // Checking whether x is a group element
        if (b < 0 || b >= 2) {
            return absl::InvalidArgumentError(
                    "party should be 0 or 1");
        }

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

        absl::uint128 z = (x - y) % N;

        // 2 ^ (log_group_size - 1)
        absl::uint128 h = absl::uint128(1) << (cmp_parameters_.log_group_size() - 1);
        // h - 1
        absl::uint128 h_minus_1 = h - 1;

        absl::uint128 z_msb = (z & h) >> (cmp_parameters_.log_group_size() - 1);
        absl::uint128 z_no_msb = z & h_minus_1;
        absl::uint128 z_prime = h - z_no_msb - 1;

        absl::uint128 s; // share of s

        DPF_ASSIGN_OR_RETURN(s, dcf_->Evaluate<absl::uint128>(k.dcfkey(), z_prime));

        s = s % absl::uint128(2);

        // Add beta_two to s to reduce dcf to ddcf
        absl::uint128 beta_two =
                absl::MakeUint128(k.beta_two_share().value_uint128().high(),
                                  k.beta_two_share().value_uint128().low());

        s ^= beta_two;

        absl::uint128 res = (b & z_msb) ^ s;

        // Retrieve mask share from EqKey and Add mask share to res

        absl::uint128 r_out =
                absl::MakeUint128(k.output_mask_share().value_uint128().high(),
                                  k.output_mask_share().value_uint128().low());
        res = res ^ r_out;

        return res;
    }

// ks are shares between the two parties
// x,y are masked inputs and known to both parties
// b is party
//    absl::StatusOr<std::vector<absl::uint128>>
//    ComparisonGate::BatchEval(absl::uint128 b, const std::vector<CmpKey> &ks,
//                              std::vector<uint64_t> &xs, std::vector<uint64_t> &ys) {
//        size_t batch_size = xs.size();
//
//        // Setting N = 2 ^ log_group_size
//        absl::uint128 N = absl::uint128(1) << cmp_parameters_.log_group_size();
//
//        if (b < 0 || b >= 2) {
//            return absl::InvalidArgumentError(
//                    "party should be 0 or 1");
//        }
//
//        if (xs.size() != ys.size() || ks.size() != xs.size()) {
//            return absl::InvalidArgumentError(
//                    "batch size inconsistent");
//        }
//
//        for (size_t idx = 0; idx < batch_size; idx++) {
//
//            // Checking whether x is a group element
//            if (xs[idx] < 0 || xs[idx] >= N) {
//                return absl::InvalidArgumentError(
//                        "Masked input should be between 0 and 2^log_group_size");
//            }
//
//            // Checking whether y is a group element
//            if (ys[idx] < 0 || ys[idx] >= N) {
//                return absl::InvalidArgumentError(
//                        "Masked input should be between 0 and 2^log_group_size");
//            }
//        }
//
//        std::vector<absl::uint128> res(batch_size);
//
//        for (size_t idx = 0; idx < batch_size; idx += 32) {
//            size_t sub_batch = 32 > (batch_size - idx) ? (batch_size - idx) : 32;
//            std::vector<absl::uint128> z_primes(sub_batch);
//            std::vector<absl::uint128> z_msbs(sub_batch);
//
//            for (size_t idx_sub = 0; idx_sub < sub_batch; idx_sub++) {
//
//                absl::uint128 z = (xs[idx + idx_sub] - ys[idx + idx_sub]) % N;
//
//                // 2 ^ (log_group_size - 1)
//                absl::uint128 h = absl::uint128(1) << (cmp_parameters_.log_group_size() - 1);
//                // h - 1
//                absl::uint128 h_minus_1 = h - 1;
//
//                z_msbs[idx_sub] = (z & h) >> (cmp_parameters_.log_group_size() - 1);
//                absl::uint128 z_no_msb = z & h_minus_1;
//                z_primes[idx_sub] = h - z_no_msb - 1;
//            }
//
//            // call eval one by one version (much slower)
//            //std::vector<absl::uint128> s (batch_size); // share of s
//            //for (size_t idx = 0; idx < batch_size; idx++) {
//            //    DPF_ASSIGN_OR_RETURN(s[idx], dcf_->Evaluate<absl::uint128>(ks[idx].dcfkey(), z_primes[idx]));
//            // }
//
//            // batched version
//            // refactor to remove this copy when you can
//            std::vector<const distributed_point_functions::DcfKey *> ks_dcfkeys;
//            ks_dcfkeys.reserve(sub_batch);
//            for (size_t idx_sub = 0; idx_sub < sub_batch; idx_sub++) {
//                ks_dcfkeys.push_back(&(ks[idx + idx_sub].dcfkey()));
//            }
//            DPF_ASSIGN_OR_RETURN(std::vector<absl::uint128> s,
//                                 dcf_->BatchEvaluate<absl::uint128>(ks_dcfkeys, z_primes));
//
//            for (size_t idx_sub = 0; idx_sub < sub_batch; idx_sub++) {
//                s[idx_sub] = s[idx_sub] % absl::uint128(2);
//
//                // Add beta_two to s to reduce dcf to ddcf
//                absl::uint128 beta_two =
//                        absl::MakeUint128(ks[idx + idx_sub].beta_two_share().value_uint128().high(),
//                                          ks[idx + idx_sub].beta_two_share().value_uint128().low());
//
//                s[idx_sub] ^= beta_two;
//
//                res[idx + idx_sub] = (b & z_msbs[idx_sub]) ^ s[idx_sub];
//
//
//                // Retrieve mask share from EqKey and Add mask share to res_sub
//
//                absl::uint128 r_out =
//                        absl::MakeUint128(ks[idx + idx_sub].output_mask_share().value_uint128().high(),
//                                          ks[idx + idx_sub].output_mask_share().value_uint128().low());
//                res[idx + idx_sub] = res[idx + idx_sub] ^ r_out;
//            }
//        }
//
//        return res;
//    }

// ks are shares between the two parties
// x,y are masked inputs and known to both parties
// b is party
    absl::StatusOr<std::vector<absl::uint128>>
    ComparisonGate::BatchEval(absl::uint128 b, const std::vector<CmpKey> &ks,
                              std::vector<uint64_t> &xs, std::vector<uint64_t> &ys) {
        size_t batch_size = xs.size();
        // Setting N = 2 ^ log_group_size
        absl::uint128 N = absl::uint128(1) << cmp_parameters_.log_group_size();
        if (b < 0 || b >= 2) {
            return absl::InvalidArgumentError(
                    "party should be 0 or 1");
        }
        if (xs.size() != ys.size() || ks.size() != xs.size()) {
            return absl::InvalidArgumentError(
                    "batch size inconsistent");
        }
        for (size_t idx = 0; idx < batch_size; idx++) {
            // Checking whether x is a group element
            if (xs[idx] < 0 || xs[idx] >= N) {
                return absl::InvalidArgumentError(
                        "Masked input should be between 0 and 2^log_group_size");
            }
            // Checking whether y is a group element
            if (ys[idx] < 0 || ys[idx] >= N) {
                return absl::InvalidArgumentError(
                        "Masked input should be between 0 and 2^log_group_size");
            }
        }
        std::vector<absl::uint128> z_primes(batch_size);
        std::vector<absl::uint128> z_msbs(batch_size);
        for (size_t idx = 0; idx < batch_size; idx++) {
            absl::uint128 z = (xs[idx] - ys[idx]) % N;
            // 2 ^ (log_group_size - 1)
            absl::uint128 h = absl::uint128(1) << (cmp_parameters_.log_group_size() - 1);
            // h - 1
            absl::uint128 h_minus_1 = h - 1;
            z_msbs[idx] = (z & h) >> (cmp_parameters_.log_group_size() - 1);
            absl::uint128 z_no_msb = z & h_minus_1;
            z_primes[idx] = h - z_no_msb - 1;
        }
        // call eval one by one version (much slower)
        //std::vector<absl::uint128> s (batch_size); // share of s
        //for (size_t idx = 0; idx < batch_size; idx++) {
        //    DPF_ASSIGN_OR_RETURN(s[idx], dcf_->Evaluate<absl::uint128>(ks[idx].dcfkey(), z_primes[idx]));
        // }
        // batched version
        // refactor to remove this copy when you can
        std::vector<const distributed_point_functions::DcfKey *> ks_dcfkeys;
        ks_dcfkeys.reserve(batch_size);
        for (size_t idx = 0; idx < batch_size; idx++) {
            ks_dcfkeys.push_back(&(ks[idx].dcfkey()));
        }
        DPF_ASSIGN_OR_RETURN(std::vector<absl::uint128> s,
                             dcf_->BatchEvaluate<absl::uint128>(ks_dcfkeys, z_primes));
        std::vector<absl::uint128> res(batch_size);
        for (size_t idx = 0; idx < batch_size; idx++) {
            s[idx] = s[idx] % absl::uint128(2);
            // Add beta_two to s to reduce dcf to ddcf
            absl::uint128 beta_two =
                    absl::MakeUint128(ks[idx].beta_two_share().value_uint128().high(),
                                      ks[idx].beta_two_share().value_uint128().low());
            s[idx] ^= beta_two;
            res[idx] = (b & z_msbs[idx]) ^ s[idx];
            // Retrieve mask share from EqKey and Add mask share to res
            absl::uint128 r_out =
                    absl::MakeUint128(ks[idx].output_mask_share().value_uint128().high(),
                                      ks[idx].output_mask_share().value_uint128().low());
            res[idx] = res[idx] ^ r_out;
        }
        return res;
    }


}  // namespace fss_gates
}  // namespace distributed_point_functions
