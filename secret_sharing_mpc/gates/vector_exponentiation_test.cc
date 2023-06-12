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

#include "secret_sharing_mpc/gates/vector_exponentiation.h"

#include "private_join_and_compute/util/status_testing.inc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace private_join_and_compute {
namespace {

const uint64_t kLargePrime = 9223372036854775783;  // 63 bit prime
const FixedPointElementFactory::Params kSampleLargeFactoryParams = {
    10, 63, 1024, 9007199254740992,
    9223372036854775808UL};  // {10, 63, 2^10, 2^53, 2^63}

const ExponentiationParams kSampleLargeExpParams = {3, kLargePrime};

const double kLog2OfE = 1.4426950408889634073599;  // log_2(e)

const absl::string_view kPrngSeed = "0123456789abcdef0123456789abcdef";

class VectorExponentiationTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Create a sample large 63-bit FixedPointElementFactory
    ASSERT_OK_AND_ASSIGN(
        FixedPointElementFactory temp_factory,
        FixedPointElementFactory::Create(fpe_params_.num_fractional_bits,
                                         fpe_params_.num_ring_bits));
    fpe_factory_ =
        absl::make_unique<FixedPointElementFactory>(std::move(temp_factory));

    ASSERT_OK_AND_ASSIGN(auto temp_prng, BasicRng::Create(kPrngSeed));
    prng_ = std::move(temp_prng);

    // Create the protocol parties.
    ASSERT_OK_AND_ASSIGN(auto temp_zero, SecureExponentiationPartyZero::Create(
                                             fpe_params_, exp_params_));
    party_zero_ = std::move(temp_zero);

    ASSERT_OK_AND_ASSIGN(auto temp_one, SecureExponentiationPartyOne::Create(
                                            fpe_params_, exp_params_));
    party_one_ = std::move(temp_one);
  }

  double Truncate(const double val, uint64_t multiplier) {
    if (val >= 0) {
      return floor(val * multiplier) / multiplier;
    } else {
      return ceil(val * multiplier) / multiplier;
    }
  }

  // Compare the output of the secure exponentiation protocol with the
  // real result.
  //
  // The acceptable_error is taken as (3 * result + 1) / fractional_multiplier.
  // Consequently, assuming -exp_bound < x < exp_bound, the maximum error
  // (as a real number) is (3 * e^(exp_bound) + 1) / fractional_multiplier.
  // This is the worst case bound, but in practice, the expected value of the
  // error is also much smaller than this bound.
  //
  // Note that the error is a function of the actual result, so larger
  // exponentiations will have a larger absolute error but a similar relative
  // error.
  // Error Document: https://eprint.iacr.org/2021/208.pdf
  bool IsComputedResultCloseToRealResult(
      double exponent, FixedPointElement& computed_result_fpe) {
    double computed_result = computed_result_fpe.ExportToDouble();

    // Check with closest real result.
    double truncated_log2_e =
        Truncate(kLog2OfE, fpe_params_.fractional_multiplier);
    double closest_representable_base2_exp = Truncate(
        exponent * truncated_log2_e, fpe_params_.fractional_multiplier);
    double real_result = pow(2, closest_representable_base2_exp);

    double acceptable_error =
        1.0 * ceil(3 * real_result + 1) / fpe_params_.fractional_multiplier;
    if (abs(real_result - computed_result) < acceptable_error) {
      return true;
    } else {
      return false;
    }
  }

 protected:
  std::unique_ptr<FixedPointElementFactory> fpe_factory_;
  std::unique_ptr<BasicRng> prng_;
  std::unique_ptr<SecureExponentiationPartyZero> party_zero_;
  std::unique_ptr<SecureExponentiationPartyOne> party_one_;

  FixedPointElementFactory::Params fpe_params_ = kSampleLargeFactoryParams;
  ExponentiationParams exp_params_ = kSampleLargeExpParams;
};

// Integration tests below
// Tests for individual functions are currently included in the
// secure_exponentiation.cc/h files in the
// privacy/blinders/cpp/poisson_regression repository

// Test the exponentiation of several exponents >= 1
// The types of exponents currently tested are:
// (1) Exponents that can be completely represented in the fixed-point
// representation.
// (2) Arbitrarily chosen exponents that won't fit within the number of
// fractional bits in the representation
TEST_F(VectorExponentiationTest, ExponentsGreaterThanOne) {
  std::vector<double> exponents = {
      // Simple exponents
      1.0,
      1.5,
      2.75,
      // Exponents that are not completely representable in the given number of
      // fractional bits.
      1.1234,
      1.345,
      2.178,
      2.3,
  };

  size_t length = exponents.size();

  // Run protocol for all exponents.
  // For each exponentiation, a fresh random sharing of the FixedPointElement
  // and fresh alpha, beta values will be used.

  ASSERT_OK_AND_ASSIGN(
      auto ab_pair, SampleMultToAddSharesVector(length, exp_params_.prime_q));

  std::vector<uint64_t> alpha_zero = std::move(ab_pair.first.first);
  std::vector<uint64_t> alpha_one = std::move(ab_pair.second.first);
  std::vector<uint64_t> beta_zero = std::move(ab_pair.first.second);
  std::vector<uint64_t> beta_one = std::move(ab_pair.second.second);

  // Secret share the exponents

  ASSERT_OK_AND_ASSIGN(FixedPointElement zero,
                       fpe_factory_->CreateFixedPointElementFromInt(0));

  std::vector<FixedPointElement> fpe_shares_zero(length, zero);
  std::vector<FixedPointElement> fpe_shares_one(length, zero);

  for (size_t idx = 0; idx < length; idx++) {
    ASSERT_OK_AND_ASSIGN(
        FixedPointElement fpe,
        fpe_factory_->CreateFixedPointElementFromDouble(exponents[idx]));

    // Sample randomness.
    ASSERT_OK_AND_ASSIGN(auto random_uint, prng_->Rand64());

    // Share the FixedPointElement
    ASSERT_OK_AND_ASSIGN(fpe_shares_one[idx],
                         fpe_factory_->ImportFixedPointElementFromUint64(
                             random_uint % fpe_params_.primary_ring_modulus));
    ASSERT_OK_AND_ASSIGN(fpe_shares_zero[idx], fpe - fpe_shares_one[idx]);
  }

  // Run first part of computation to get messages and state.
  ASSERT_OK_AND_ASSIGN(auto p0_return,
                   GenerateVectorMultToAddMessagePartyZero(
                       party_zero_, fpe_shares_zero, alpha_zero, beta_zero));
  ASSERT_OK_AND_ASSIGN(auto p1_return,
                   GenerateVectorMultToAddMessagePartyOne(
                       party_one_, fpe_shares_one, alpha_one, beta_one));

  // Feed the message/state into second half of computation.
  ASSERT_OK_AND_ASSIGN(std::vector<FixedPointElement> output_fpe_p0,
                   VectorExponentiationPartyZero(party_zero_, p1_return.first,
                                                 p0_return.second));
  ASSERT_OK_AND_ASSIGN(std::vector<FixedPointElement> output_fpe_p1,
                   VectorExponentiationPartyOne(party_one_, p0_return.first,
                                                p1_return.second));

  // Reconstruct output and verify the correctness.
  for (size_t idx = 0; idx < length; idx++) {
    ASSERT_OK_AND_ASSIGN(FixedPointElement computed_sum,
                         output_fpe_p0[idx] + output_fpe_p1[idx]);
    EXPECT_TRUE(IsComputedResultCloseToRealResult(exponents[idx],
                                                  computed_sum));
  }
}

TEST_F(VectorExponentiationTest, ExponentsLessThanOne) {
  std::vector<double> exponents = {
      // Simple negative exponents
      -3.0,
      -2.75,
      -1.5,
      -1.0,

      // Simple exponents between 0 and 1.
      0,
      0.125,
      0.25,
      0.75,

      // Exponents that are not completely representable in the given number of
      // fractional bits.
      -2.3,
      -2.178,
      -1.345,
      -1.1234,
      -0.4,
      -0.1234,
      0.7,
  };

  size_t length = exponents.size();

  // Run protocol for all exponents.
  // For each exponentiation, a fresh random sharing of the FixedPointElement
  // and fresh alpha, beta values will be used.

  ASSERT_OK_AND_ASSIGN(
      auto ab_pair, SampleMultToAddSharesVector(length, exp_params_.prime_q));

  std::vector<uint64_t> alpha_zero = std::move(ab_pair.first.first);
  std::vector<uint64_t> alpha_one = std::move(ab_pair.second.first);
  std::vector<uint64_t> beta_zero = std::move(ab_pair.first.second);
  std::vector<uint64_t> beta_one = std::move(ab_pair.second.second);

  // Secret share the exponents

  ASSERT_OK_AND_ASSIGN(FixedPointElement zero,
                       fpe_factory_->CreateFixedPointElementFromInt(0));

  std::vector<FixedPointElement> fpe_shares_zero(length, zero);
  std::vector<FixedPointElement> fpe_shares_one(length, zero);

  for (size_t idx = 0; idx < length; idx++) {
    ASSERT_OK_AND_ASSIGN(
        FixedPointElement fpe,
        fpe_factory_->CreateFixedPointElementFromDouble(exponents[idx]));

    // Sample randomness.
    ASSERT_OK_AND_ASSIGN(auto random_uint, prng_->Rand64());

    // Share the FixedPointElement
    ASSERT_OK_AND_ASSIGN(fpe_shares_one[idx],
                         fpe_factory_->ImportFixedPointElementFromUint64(
                             random_uint % fpe_params_.primary_ring_modulus));
    ASSERT_OK_AND_ASSIGN(fpe_shares_zero[idx], fpe - fpe_shares_one[idx]);
  }

  // Run first part of computation to get messages and state.
  ASSERT_OK_AND_ASSIGN(
      auto p0_return, GenerateVectorMultToAddMessagePartyZero(
                          party_zero_, fpe_shares_zero, alpha_zero, beta_zero));
  ASSERT_OK_AND_ASSIGN(auto p1_return,
                       GenerateVectorMultToAddMessagePartyOne(
                           party_one_, fpe_shares_one, alpha_one, beta_one));

  // Feed the message/state into second half of computation.
  ASSERT_OK_AND_ASSIGN(std::vector<FixedPointElement> output_fpe_p0,
                       VectorExponentiationPartyZero(
                           party_zero_, p1_return.first, p0_return.second));
  ASSERT_OK_AND_ASSIGN(std::vector<FixedPointElement> output_fpe_p1,
                       VectorExponentiationPartyOne(party_one_, p0_return.first,
                                                    p1_return.second));

  // Reconstruct output and verify the correctness.
  for (size_t idx = 0; idx < length; idx++) {
    ASSERT_OK_AND_ASSIGN(FixedPointElement computed_sum,
                         output_fpe_p0[idx] + output_fpe_p1[idx]);
    EXPECT_TRUE(
        IsComputedResultCloseToRealResult(exponents[idx], computed_sum));
  }
}

}  // namespace
}  // namespace private_join_and_compute
