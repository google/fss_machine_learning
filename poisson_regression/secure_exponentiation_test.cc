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

#include "poisson_regression/secure_exponentiation.h"

#include <vector>

#include "private_join_and_compute/crypto/context.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/prng/basic_rng.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "private_join_and_compute/util/status.inc"
#include "private_join_and_compute/util/status_testing.inc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace private_join_and_compute {
namespace {

using ::testing::HasSubstr;
using testing::StatusIs;

const uint64_t kSamplePrime = 2147483647;  // The mersene prime 2^{31} - 1
const uint64_t kLargePrime = 9223372036854775783;  // 63 bit prime
const uint64_t kSmallComposite = 20;
const uint64_t kLargeComposite = 4611686014132420609;  // (kSamplePrime)^2
const FixedPointElementFactory::Params kSampleFixedPointFactoryParams
    = {10, 32, 1024, 4194304, 4294967296};  // {10, 32, 2^10, 2^22, 2^32}
const FixedPointElementFactory::Params kSampleLargeFactoryParams
    = {10, 63, 1024,
       9007199254740992, 9223372036854775808UL};  // {10, 63, 2^10, 2^53, 2^63}

const ExponentiationParams kSampleExpParams = {3, kSamplePrime};
const ExponentiationParams kSampleLargeExpParams = {3, kLargePrime};
const ExponentiationParams kExpParamsNotPrime = {5, kSmallComposite};
const ExponentiationParams kExpParamsLargeComposite = {5, kLargeComposite};
const ExponentiationParams kExpParamsTooSmall = {20, kSamplePrime};  // 1024 *
// e^20 is larger than kSamplePrime

const double kLog2OfE = 1.4426950408889634073599;  // log_2(e)

const absl::string_view kPrngSeed = "0123456789abcdef0123456789abcdef";

class SecureExponentiationPartyCreateTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Create a sample 32-bit FixedPointElementFactory
    ASSERT_OK_AND_ASSIGN(FixedPointElementFactory temp_factory,
        FixedPointElementFactory::Create(
            kSampleFixedPointFactoryParams.num_fractional_bits,
            kSampleFixedPointFactoryParams.num_ring_bits));
    fpe_factory_ = absl::make_unique<FixedPointElementFactory>(
        std::move(temp_factory));
  }

 protected:
  std::unique_ptr<FixedPointElementFactory> fpe_factory_;
};

TEST_F(SecureExponentiationPartyCreateTest, ExpParamsBoundDoesNotFit) {
  // Party Zero
  EXPECT_THAT(
      SecureExponentiationPartyZero::Create(
          kSampleFixedPointFactoryParams, kExpParamsTooSmall),
      StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
               HasSubstr("should be large enough to fit the maximum "
                         "exponentiation result")));
  // Party One
  EXPECT_THAT(
      SecureExponentiationPartyOne::Create(
          kSampleFixedPointFactoryParams, kExpParamsTooSmall),
      StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
               HasSubstr("should be large enough to fit the maximum "
                         "exponentiation result")));
}

TEST_F(SecureExponentiationPartyCreateTest,
       TestExpParamsBoundLargerThanUint64) {
  ExponentiationParams exp_params = {22, kSamplePrime};
  // Party Zero
  EXPECT_THAT(
      SecureExponentiationPartyZero::Create(
          kSampleFixedPointFactoryParams, exp_params),
      StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
               HasSubstr("should be large enough to fit the maximum "
                         "exponentiation result")));
  // Party One
  EXPECT_THAT(
      SecureExponentiationPartyOne::Create(
          kSampleFixedPointFactoryParams, exp_params),
      StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
               HasSubstr("should be large enough to fit the maximum "
                         "exponentiation result")));
}

TEST_F(SecureExponentiationPartyCreateTest, ValidCreateBothParties) {
  ASSERT_OK(SecureExponentiationPartyZero::Create(
      kSampleFixedPointFactoryParams, kSampleExpParams));
  ASSERT_OK(SecureExponentiationPartyOne::Create(
      kSampleFixedPointFactoryParams, kSampleExpParams));
}

class SecureExponentiationPartyGenerateTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Create a sample 32-bit FixedPointElementFactory
    ASSERT_OK_AND_ASSIGN(FixedPointElementFactory temp_factory,
        FixedPointElementFactory::Create(
            kSampleFixedPointFactoryParams.num_fractional_bits,
            kSampleFixedPointFactoryParams.num_ring_bits));
    fpe_factory_ = absl::make_unique<FixedPointElementFactory>(
        std::move(temp_factory));

    // Sample alpha and beta values
    // For testing purposes only, private_join_and_compute::BigNum is used for modular
    // multiplication and exponentiation.
    Context context;
    BigNum big_num_prime = context.CreateBigNum(kSamplePrime);
    BigNum alpha_zero = context.GenerateRandLessThan(big_num_prime);
    BigNum alpha_one = context.GenerateRandLessThan(big_num_prime);
    BigNum beta_zero = context.GenerateRandLessThan(big_num_prime);

    BigNum alpha_mult = alpha_zero.ModMul(alpha_one, big_num_prime);
    BigNum beta_one = beta_zero.ModInverse(big_num_prime)
        .ModMul(context.One() - alpha_mult, big_num_prime);

    // Sanity check that the sampled correctly satisfy the requirement
    // alpha_zero * alpha_one + beta_zero * beta_one = 1.
    BigNum result = alpha_zero * alpha_one + beta_zero * beta_one;
    EXPECT_TRUE(result.Mod(big_num_prime).IsOne());

    ASSERT_OK_AND_ASSIGN(uint64_t a0_temp, alpha_zero.ToIntValue());
    ASSERT_OK_AND_ASSIGN(uint64_t a1_temp, alpha_one.ToIntValue());
    ASSERT_OK_AND_ASSIGN(uint64_t b0_temp, beta_zero.ToIntValue());
    ASSERT_OK_AND_ASSIGN(uint64_t b1_temp, beta_one.ToIntValue());

    sample_alpha_zero_ = std::move(a0_temp);
    sample_alpha_one_ = std::move(a1_temp);
    sample_beta_zero_ = std::move(b0_temp);
    sample_beta_one_ = std::move(b1_temp);
  }

 protected:
  std::unique_ptr<FixedPointElementFactory> fpe_factory_;
  uint64_t sample_alpha_zero_;
  uint64_t sample_alpha_one_;
  uint64_t sample_beta_zero_;
  uint64_t sample_beta_one_;
};

TEST_F(SecureExponentiationPartyGenerateTest, InvalidCreateNegativeAlphaBeta) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       fpe_factory_->CreateFixedPointElementFromDouble(-1.5));

  // Party Zero
  ASSERT_OK_AND_ASSIGN(auto party_zero,
                       SecureExponentiationPartyZero::Create(
                           kSampleFixedPointFactoryParams, kSampleExpParams));
  EXPECT_THAT(
      party_zero->GenerateMultToAddMessage(fpe, -1, 0),
      StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
               HasSubstr("alpha_zero and beta_zero must be non-negative "
                         "and less than prime_q.")));
  EXPECT_THAT(
      party_zero->GenerateMultToAddMessage(fpe, 0, -1),
      StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
               HasSubstr("alpha_zero and beta_zero must be non-negative "
                         "and less than prime_q.")));

  // Party One
  ASSERT_OK_AND_ASSIGN(auto party_one,
                       SecureExponentiationPartyOne::Create(
                           kSampleFixedPointFactoryParams, kSampleExpParams));
  EXPECT_THAT(
      party_one->GenerateMultToAddMessage(fpe, -1, 0),
      StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
               HasSubstr("alpha_one and beta_one must be non-negative "
                         "and less than prime_q.")));
  EXPECT_THAT(
      party_one->GenerateMultToAddMessage(fpe, 0, -1),
      StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
               HasSubstr("alpha_one and beta_one must be non-negative "
                         "and less than prime_q.")));
}

TEST_F(SecureExponentiationPartyGenerateTest,
       TestInvalidCreateTooLargeAlphaBeta) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       fpe_factory_->CreateFixedPointElementFromDouble(-1.5));

  // Party Zero
  ASSERT_OK_AND_ASSIGN(auto party_zero,
                       SecureExponentiationPartyZero::Create(
                           kSampleFixedPointFactoryParams, kSampleExpParams));
  EXPECT_THAT(
      party_zero->GenerateMultToAddMessage(fpe, kSamplePrime, 0),
      StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
               HasSubstr("alpha_zero and beta_zero must be non-negative "
                         "and less than prime_q.")));
  EXPECT_THAT(
      party_zero->GenerateMultToAddMessage(fpe, 0, kSamplePrime),
      StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
               HasSubstr("alpha_zero and beta_zero must be non-negative "
                         "and less than prime_q.")));

  // Party One
  ASSERT_OK_AND_ASSIGN(auto party_one,
                       SecureExponentiationPartyOne::Create(
                           kSampleFixedPointFactoryParams, kSampleExpParams));
  EXPECT_THAT(
      party_one->GenerateMultToAddMessage(fpe, kSamplePrime, 0),
      StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
               HasSubstr("alpha_one and beta_one must be non-negative "
                         "and less than prime_q.")));
  EXPECT_THAT(
      party_one->GenerateMultToAddMessage(fpe, 0, kSamplePrime),
      StatusIs(::private_join_and_compute::StatusCode::kInvalidArgument,
               HasSubstr("alpha_one and beta_one must be non-negative "
                         "and less than prime_q.")));
}

class SecureExponentiationTest : public ::testing::Test {
 public:
  struct AlphaBetas {
    uint64_t alpha_zero;
    uint64_t alpha_one;
    uint64_t beta_zero;
    uint64_t beta_one;
  };

  struct BatchedAlphaBetas {
    std::vector<uint64_t> alpha_zero;
    std::vector<uint64_t> alpha_one;
    std::vector<uint64_t> beta_zero;
    std::vector<uint64_t> beta_one;
  };

  void SetUp() override {
    // Create a sample large 63-bit FixedPointElementFactory
    ASSERT_OK_AND_ASSIGN(FixedPointElementFactory temp_factory,
        FixedPointElementFactory::Create(
            fpe_params_.num_fractional_bits,
            fpe_params_.num_ring_bits));
    fpe_factory_ = absl::make_unique<FixedPointElementFactory>(
        std::move(temp_factory));

    ASSERT_OK_AND_ASSIGN(auto temp_prng, BasicRng::Create(kPrngSeed));
    prng_ = std::move(temp_prng);

    // Create the protocol parties.
    ASSERT_OK_AND_ASSIGN(auto temp_zero,
                         SecureExponentiationPartyZero::Create(
                             fpe_params_, exp_params_));
    party_zero_ = std::move(temp_zero);

    ASSERT_OK_AND_ASSIGN(auto temp_one,
                         SecureExponentiationPartyOne::Create(
                             fpe_params_, exp_params_));
    party_one_ = std::move(temp_one);
  }

  StatusOr<AlphaBetas> SampleAlphaBeta() {
    ASSIGN_OR_RETURN(
        auto ab_pair,
        ::private_join_and_compute::internal::SampleMultToAddSharesWithPrng(
            1, exp_params_.prime_q));
    return AlphaBetas{ab_pair.first.first[0], ab_pair.second.first[0],
                     ab_pair.first.second[0], ab_pair.second.second[0]};
  }

  StatusOr<BatchedAlphaBetas> SampleBatchedAlphaBeta(size_t length) {
    ASSIGN_OR_RETURN(auto ab_pair,
                     ::private_join_and_compute::internal::SampleMultToAddSharesWithPrng(
                         length, exp_params_.prime_q));
    return BatchedAlphaBetas{ab_pair.first.first, ab_pair.second.first,
                             ab_pair.first.second, ab_pair.second.second};
  }

  double Truncate(const double val, uint64_t multiplier) {
    if (val >= 0) {
      return floor(val * multiplier) / multiplier;
    } else {
      return ceil(val * multiplier) / multiplier;
    }
  }

  // Run the secure exponentiation protocol and return the sum of the
  // output FixedPointElements.
  StatusOr<FixedPointElement> RunProtocolForGivenSharing(
      FixedPointElement& fpe_share_zero,
      FixedPointElement& fpe_share_one,
      AlphaBetas& ab) {
    // Run first part of computation to get messages and state.
    ASSIGN_OR_RETURN(auto p0_return,
                         party_zero_->GenerateMultToAddMessage(
                             fpe_share_zero, ab.alpha_zero, ab.beta_zero));
    ASSIGN_OR_RETURN(auto p1_return,
                         party_one_->GenerateMultToAddMessage(
                             fpe_share_one, ab.alpha_one, ab.beta_one));

    // Feed into second half of computation.
    ASSIGN_OR_RETURN(auto output_fpe_p0,
                         party_zero_->OutputResult(
                             p1_return.first, p0_return.second));
    ASSIGN_OR_RETURN(auto output_fpe_p1,
                         party_one_->OutputResult(
                             p0_return.first, p1_return.second));

    // Return combined computation result.
    return output_fpe_p0 + output_fpe_p1;
  }

  // Run the batched secure exponentiation protocol and return the sum of the
  // output FixedPointElements.
  StatusOr<std::vector<FixedPointElement>> RunBatchedProtocolForGivenSharing(
      std::vector<FixedPointElement>& fpe_shares_zero,
      std::vector<FixedPointElement>& fpe_shares_one, BatchedAlphaBetas& ab) {
    size_t length = fpe_shares_zero.size();

    // Run first part of computation to get messages and state.
    ASSIGN_OR_RETURN(auto p0_return,
                     party_zero_->GenerateBatchedMultToAddMessage(
                         fpe_shares_zero, ab.alpha_zero, ab.beta_zero));
    ASSIGN_OR_RETURN(auto p1_return,
                     party_one_->GenerateBatchedMultToAddMessage(
                         fpe_shares_one, ab.alpha_one, ab.beta_one));

    // Feed into second half of computation.
    ASSIGN_OR_RETURN(
        auto output_fpe_p0,
        party_zero_->BatchedOutputResult(p1_return.first, p0_return.second));
    ASSIGN_OR_RETURN(
        auto output_fpe_p1,
        party_one_->BatchedOutputResult(p0_return.first, p1_return.second));

    // Return reconstructed computation result.
    ASSIGN_OR_RETURN(FixedPointElement zero,
                     fpe_factory_->CreateFixedPointElementFromInt(0));
    std::vector<FixedPointElement> res(length, zero);
    for (size_t idx = 0; idx < length; idx++) {
      ASSIGN_OR_RETURN(res[idx], output_fpe_p0[idx] + output_fpe_p1[idx]);
    }
    return res;
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
  bool IsComputedResultCloseToRealResult(
      double exponent,
      FixedPointElement& computed_result_fpe) {
    double computed_result = computed_result_fpe.ExportToDouble();

    // Check with closest real result.
    double truncated_log2_e = Truncate(
        kLog2OfE, fpe_params_.fractional_multiplier);
    double closest_representable_base2_exp = Truncate(
        exponent * truncated_log2_e, fpe_params_.fractional_multiplier);
    double real_result = pow(2, closest_representable_base2_exp);

    double acceptable_error = 1.0 * ceil(3 * real_result + 1) /
        fpe_params_.fractional_multiplier;
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

// Test the exponentiation of several exponents >= 1
// The types of exponents currently tested are:
// (1) Exponents that can be completely representated in the fixed-point
// representation.
// (2) Arbitrarily chosen exponents that won't fit within the number of
// fractional bits in the representation
TEST_F(SecureExponentiationTest, SeveralExponentsGreaterThanOne) {
  std::vector<double> exponents = {
    // Simple exponents
    1.0, 1.5, 2.75,

    // Exponents that are not completely representable in the given number of
    // fractional bits.
    1.1234, 1.345, 2.178, 2.3,
  };

  // Run protocol for all exponents.
  // For each exponentiation, a fresh random sharing of the FixedPointElement
  // and fresh alpha, beta values will be used.
  for (size_t index = 0; index < exponents.size(); index++) {
    ASSERT_OK_AND_ASSIGN(FixedPointElement fpe, fpe_factory_
                       ->CreateFixedPointElementFromDouble(exponents[index]));

    // Sample randomness.
    ASSERT_OK_AND_ASSIGN(auto ab, SampleAlphaBeta());
    ASSERT_OK_AND_ASSIGN(auto random_uint, prng_->Rand64());

    // Share the FixedPointElement
    ASSERT_OK_AND_ASSIGN(
        auto fpe_share_one,
        fpe_factory_->ImportFixedPointElementFromUint64(
            random_uint % fpe_params_.primary_ring_modulus));
    ASSERT_OK_AND_ASSIGN(auto fpe_share_zero, fpe - fpe_share_one);

    ASSERT_OK_AND_ASSIGN(auto computed_sum,
                       RunProtocolForGivenSharing(
                           fpe_share_zero, fpe_share_one, ab));
    EXPECT_TRUE(IsComputedResultCloseToRealResult(
        exponents[index], computed_sum));
  }
}

// Test the batched exponentiation of several exponents >= 1
// The types of exponents currently tested are:
// (1) Exponents that can be completely representated in the fixed-point
// representation.
// (2) Arbitrarily chosen exponents that won't fit within the number of
// fractional bits in the representation
TEST_F(SecureExponentiationTest, BatchedSeveralExponentsGreaterThanOne) {
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
  ASSERT_OK_AND_ASSIGN(FixedPointElement zero,
                       fpe_factory_->CreateFixedPointElementFromInt(0));
  std::vector<FixedPointElement> fpes(length, zero);
  for (size_t index = 0; index < length; index++) {
    ASSERT_OK_AND_ASSIGN(
        fpes[index],
        fpe_factory_->CreateFixedPointElementFromDouble(exponents[index]));
  }

  // Sample randomness.
  ASSERT_OK_AND_ASSIGN(auto ab, SampleBatchedAlphaBeta(length));
  ASSERT_OK_AND_ASSIGN(
      auto random_uints,
      SampleVectorFromPrng(length, fpe_params_.primary_ring_modulus,
                           prng_.get()));

  // Share the FixedPointElements
  std::vector<FixedPointElement> fpe_shares_zero(length, zero);
  std::vector<FixedPointElement> fpe_shares_one(length, zero);
  for (size_t index = 0; index < length; index++) {
    ASSERT_OK_AND_ASSIGN(
        fpe_shares_one[index],
        fpe_factory_->ImportFixedPointElementFromUint64(random_uints[index]));
    ASSERT_OK_AND_ASSIGN(fpe_shares_zero[index],
                         fpes[index] - fpe_shares_one[index]);
  }
  ASSERT_OK_AND_ASSIGN(
      auto computed_sum,
      RunBatchedProtocolForGivenSharing(fpe_shares_zero, fpe_shares_one, ab));

  for (size_t index = 0; index < length; index++) {
    EXPECT_TRUE(IsComputedResultCloseToRealResult(exponents[index],
                                                  computed_sum[index]));
  }
}

// Test that the exponentiation protocol works correctly when to exponentiate
// w.f, the fractional part exponentiates 1.f and the integer part exponentiates
// (w-1). For this, a particular sharing of the FixedPointElement is chosen
// instead of a random sharing.
TEST_F(SecureExponentiationTest, FractionalPartExponentiatesGreaterThanOne) {
  double exponent = 2.0;

  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe, fpe_factory_
                       ->CreateFixedPointElementFromDouble(exponent));

  // Sample randomness only for the pre-processed tuple.
  ASSERT_OK_AND_ASSIGN(auto ab, SampleAlphaBeta());

  // Share the FixedPointElement in a way such that the two fractional shares
  // will add up to >= 1.
  // For this test case, the exponent 2.0 is shared as
  // 2.5 and w.5 where w = (integer_ring_modulus - 1)
  // The FixedPointElement shares are these shares multiplied by
  // fractional_multiplier.
  uint64_t share_one = fpe_params_.primary_ring_modulus
      - fpe_params_.fractional_multiplier / 2;
  ASSERT_OK_AND_ASSIGN(
      auto fpe_share_one,
      fpe_factory_->ImportFixedPointElementFromUint64(share_one));
  ASSERT_OK_AND_ASSIGN(auto fpe_share_zero, fpe - fpe_share_one);

  ASSERT_OK_AND_ASSIGN(auto computed_sum,
                       RunProtocolForGivenSharing(
                           fpe_share_zero, fpe_share_one, ab));
  EXPECT_TRUE(IsComputedResultCloseToRealResult(exponent, computed_sum));
}

// Test that the exponentiation protocol works correctly when to exponentiate
// w.f, the fractional part exponentiates 1.f and the integer part exponentiates
// (w-1). For this, a particular sharing of the FixedPointElement is chosen
// instead of a random sharing.
TEST_F(SecureExponentiationTest,
       BatchedFractionalPartExponentiatesGreaterThanOne) {
  double exponent = 2.0;

  ASSERT_OK_AND_ASSIGN(
      FixedPointElement fpe,
      fpe_factory_->CreateFixedPointElementFromDouble(exponent));

  // Sample randomness only for the pre-processed tuple.
  ASSERT_OK_AND_ASSIGN(auto ab, SampleBatchedAlphaBeta(1));

  // Share the FixedPointElement in a way such that the two fractional shares
  // will add up to >= 1.
  // For this test case, the exponent 2.0 is shared as
  // 2.5 and w.5 where w = (integer_ring_modulus - 1)
  // The FixedPointElement shares are these shares multiplied by
  // fractional_multiplier.
  uint64_t share_one =
      fpe_params_.primary_ring_modulus - fpe_params_.fractional_multiplier / 2;
  ASSERT_OK_AND_ASSIGN(
      auto fpe_share_one,
      fpe_factory_->ImportFixedPointElementFromUint64(share_one));
  ASSERT_OK_AND_ASSIGN(auto fpe_share_zero, fpe - fpe_share_one);

  std::vector<FixedPointElement> fpe_shares_zero = {fpe_share_zero};
  std::vector<FixedPointElement> fpe_shares_one = {fpe_share_one};

  ASSERT_OK_AND_ASSIGN(
      auto computed_sum,
      RunBatchedProtocolForGivenSharing(fpe_shares_zero, fpe_shares_one, ab));
  EXPECT_TRUE(IsComputedResultCloseToRealResult(exponent, computed_sum[0]));
}

// Test exponentiation of the smallest representable number > 1.
// This number is 1 + 1/fractional_multiplier which is
// fractional_multiplier + 1, as a FixedPointElement
TEST_F(SecureExponentiationTest, TestUintExponent) {
  uint64_t uint_exponent = fpe_params_.fractional_multiplier + 1;
  double real_exponent = 1 + 1.0/fpe_params_.fractional_multiplier;

  ASSERT_OK_AND_ASSIGN(auto fpe, fpe_factory_
                       ->ImportFixedPointElementFromUint64(uint_exponent));

  // Sample randomness.
  ASSERT_OK_AND_ASSIGN(auto ab, SampleAlphaBeta());
  ASSERT_OK_AND_ASSIGN(auto random_uint, prng_->Rand64());

  // Share the FixedPointElement
  ASSERT_OK_AND_ASSIGN(
      auto fpe_share_one,
      fpe_factory_->ImportFixedPointElementFromUint64(
          random_uint % fpe_params_.primary_ring_modulus));
  ASSERT_OK_AND_ASSIGN(auto fpe_share_zero, fpe - fpe_share_one);

  ASSERT_OK_AND_ASSIGN(auto computed_sum,
                       RunProtocolForGivenSharing(
                           fpe_share_zero, fpe_share_one, ab));
  EXPECT_TRUE(IsComputedResultCloseToRealResult(real_exponent, computed_sum));
}

// Test exponentiation of the smallest representable number > 1.
// This number is 1 + 1/fractional_multiplier which is
// fractional_multiplier + 1, as a FixedPointElement
TEST_F(SecureExponentiationTest, TestBatchedUintExponent) {
  uint64_t uint_exponent = fpe_params_.fractional_multiplier + 1;
  double real_exponent = 1 + 1.0 / fpe_params_.fractional_multiplier;

  ASSERT_OK_AND_ASSIGN(
      auto fpe, fpe_factory_->ImportFixedPointElementFromUint64(uint_exponent));

  // Sample randomness.
  ASSERT_OK_AND_ASSIGN(auto ab, SampleBatchedAlphaBeta(1));
  ASSERT_OK_AND_ASSIGN(auto random_uint, prng_->Rand64());

  // Share the FixedPointElement
  ASSERT_OK_AND_ASSIGN(auto fpe_share_one,
                       fpe_factory_->ImportFixedPointElementFromUint64(
                           random_uint % fpe_params_.primary_ring_modulus));
  ASSERT_OK_AND_ASSIGN(auto fpe_share_zero, fpe - fpe_share_one);

  std::vector<FixedPointElement> fpe_shares_zero = {fpe_share_zero};
  std::vector<FixedPointElement> fpe_shares_one = {fpe_share_one};

  ASSERT_OK_AND_ASSIGN(
      auto computed_sum,
      RunBatchedProtocolForGivenSharing(fpe_shares_zero, fpe_shares_one, ab));
  EXPECT_TRUE(
      IsComputedResultCloseToRealResult(real_exponent, computed_sum[0]));
}

// Both shares of the exponent are large, and lie in the upper half of the ring
// that represents negative fixed-point numbers.
TEST_F(SecureExponentiationTest, BothLargeShares) {
  ASSERT_OK_AND_ASSIGN(
      FixedPointElement fpe,
      fpe_factory_->ImportFixedPointElementFromUint64(2));
  ASSERT_OK_AND_ASSIGN(auto ab, SampleAlphaBeta());

  // Share fpe such that both shares are large.
  ASSERT_OK_AND_ASSIGN(
      auto fpe_share_one,
      fpe_factory_->ImportFixedPointElementFromUint64(
          fpe_params_.primary_ring_modulus / 2 + 1));
  ASSERT_OK_AND_ASSIGN(auto fpe_share_zero, fpe - fpe_share_one);
  ASSERT_OK_AND_ASSIGN(auto computed_sum,
                       RunProtocolForGivenSharing(
                           fpe_share_zero, fpe_share_one, ab));
  EXPECT_TRUE(IsComputedResultCloseToRealResult(
      2.0/fpe_params_.fractional_multiplier, computed_sum));
}

// Both shares of the exponent are large, and lie in the upper half of the ring
// that represents negative fixed-point numbers.
TEST_F(SecureExponentiationTest, BatchedBothLargeShares) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe,
                       fpe_factory_->ImportFixedPointElementFromUint64(2));
  ASSERT_OK_AND_ASSIGN(auto ab, SampleBatchedAlphaBeta(1));

  // Share fpe such that both shares are large.
  ASSERT_OK_AND_ASSIGN(auto fpe_share_one,
                       fpe_factory_->ImportFixedPointElementFromUint64(
                           fpe_params_.primary_ring_modulus / 2 + 1));
  ASSERT_OK_AND_ASSIGN(auto fpe_share_zero, fpe - fpe_share_one);

  std::vector<FixedPointElement> fpe_shares_zero = {fpe_share_zero};
  std::vector<FixedPointElement> fpe_shares_one = {fpe_share_one};

  ASSERT_OK_AND_ASSIGN(
      auto computed_sum,
      RunBatchedProtocolForGivenSharing(fpe_shares_zero, fpe_shares_one, ab));
  EXPECT_TRUE(IsComputedResultCloseToRealResult(
      2.0 / fpe_params_.fractional_multiplier, computed_sum[0]));
}

TEST_F(SecureExponentiationTest, ExponentsLessThanOne) {
  std::vector<double> exponents = {
    // Simple negative exponents
    -3.0, -2.75, -1.5, -1.0,

    // Simple exponents between 0 and 1.
    0, 0.125, 0.25, 0.75,

    // Exponents that are not completely representable in the given number of
    // fractional bits.
    -2.3, -2.178, -1.345, -1.1234, -0.4, -0.1234, 0.7,
  };

  // Run protocol for all exponents.
  // For each exponentiation, a fresh random sharing of the FixedPointElement
  // and fresh alpha, beta values will be used.
  for (size_t index = 0; index < exponents.size(); index++) {
    ASSERT_OK_AND_ASSIGN(FixedPointElement fpe, fpe_factory_
                       ->CreateFixedPointElementFromDouble(exponents[index]));

    // Sample randomness.
    ASSERT_OK_AND_ASSIGN(auto ab, SampleAlphaBeta());
    ASSERT_OK_AND_ASSIGN(auto random_uint, prng_->Rand64());

    // Share the FixedPointElement
    ASSERT_OK_AND_ASSIGN(
        auto fpe_share_one,
        fpe_factory_->ImportFixedPointElementFromUint64(
            random_uint % fpe_params_.primary_ring_modulus));
    ASSERT_OK_AND_ASSIGN(auto fpe_share_zero, fpe - fpe_share_one);

    ASSERT_OK_AND_ASSIGN(auto computed_sum,
                       RunProtocolForGivenSharing(
                           fpe_share_zero, fpe_share_one, ab));
    EXPECT_TRUE(IsComputedResultCloseToRealResult(
        exponents[index], computed_sum));
  }
}

TEST_F(SecureExponentiationTest, BatchedExponentsLessThanOne) {
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
  ASSERT_OK_AND_ASSIGN(FixedPointElement zero,
                       fpe_factory_->CreateFixedPointElementFromInt(0));
  std::vector<FixedPointElement> fpes(length, zero);
  for (size_t index = 0; index < length; index++) {
    ASSERT_OK_AND_ASSIGN(
        fpes[index],
        fpe_factory_->CreateFixedPointElementFromDouble(exponents[index]));
  }

  // Sample randomness.
  ASSERT_OK_AND_ASSIGN(auto ab, SampleBatchedAlphaBeta(length));
  ASSERT_OK_AND_ASSIGN(
      auto random_uints,
      SampleVectorFromPrng(length, fpe_params_.primary_ring_modulus,
                           prng_.get()));

  // Share the FixedPointElements
  std::vector<FixedPointElement> fpe_shares_zero(length, zero);
  std::vector<FixedPointElement> fpe_shares_one(length, zero);
  for (size_t index = 0; index < length; index++) {
    ASSERT_OK_AND_ASSIGN(
        fpe_shares_one[index],
        fpe_factory_->ImportFixedPointElementFromUint64(random_uints[index]));
    ASSERT_OK_AND_ASSIGN(fpe_shares_zero[index],
                         fpes[index] - fpe_shares_one[index]);
  }
  ASSERT_OK_AND_ASSIGN(
      auto computed_sum,
      RunBatchedProtocolForGivenSharing(fpe_shares_zero, fpe_shares_one, ab));

  for (size_t index = 0; index < length; index++) {
    EXPECT_TRUE(IsComputedResultCloseToRealResult(exponents[index],
                                                  computed_sum[index]));
  }
}

TEST_F(SecureExponentiationTest, LargeNegativeValidExponent) {
  // exponent is chosen such that
  // exponent*log_2(e) + ceil(log_2(e) * exp_bound + 1) >= 1 and ~= 1.
  // This is the close to the most negative exponent that will be handled
  // by the protocol.
  // Ideally, exp_bound will be chosen such that exponent >= -exp_bound, but
  // for the purpose of testing, an even smaller exponent is used.
  // Smaller exponents than this might still work if the sharing after
  // multiplying by kLog2OfE results in the fractional parts adding up to less
  // than 1.
  double exponent = -3.46;
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe, fpe_factory_
                       ->CreateFixedPointElementFromDouble(exponent));

  // Sample randomness.
  ASSERT_OK_AND_ASSIGN(auto ab, SampleAlphaBeta());
  ASSERT_OK_AND_ASSIGN(auto random_uint, prng_->Rand64());

  // Share the FixedPointElement
  ASSERT_OK_AND_ASSIGN(
      auto fpe_share_one,
      fpe_factory_->ImportFixedPointElementFromUint64(
          random_uint % fpe_params_.primary_ring_modulus));
  ASSERT_OK_AND_ASSIGN(auto fpe_share_zero, fpe - fpe_share_one);

  ASSERT_OK_AND_ASSIGN(auto computed_sum,
                       RunProtocolForGivenSharing(
                           fpe_share_zero, fpe_share_one, ab));

  EXPECT_TRUE(IsComputedResultCloseToRealResult(
      exponent, computed_sum));
}

TEST_F(SecureExponentiationTest, BatchedLargeNegativeValidExponent) {
  // exponent is chosen such that
  // exponent*log_2(e) + ceil(log_2(e) * exp_bound + 1) >= 1 and ~= 1.
  // This is close to the most negative exponent that will be handled
  // by the protocol.
  // Ideally, exp_bound will be chosen such that exponent >= -exp_bound, but
  // for the purpose of testing, an even smaller exponent is used.
  // Smaller exponents than this might still work if the sharing after
  // multiplying by kLog2OfE results in the fractional parts adding up to less
  // than 1.
  double exponent = -3.46;
  ASSERT_OK_AND_ASSIGN(
      FixedPointElement fpe,
      fpe_factory_->CreateFixedPointElementFromDouble(exponent));

  // Sample randomness.
  ASSERT_OK_AND_ASSIGN(auto ab, SampleBatchedAlphaBeta(1));
  ASSERT_OK_AND_ASSIGN(auto random_uint, prng_->Rand64());

  // Share the FixedPointElement
  ASSERT_OK_AND_ASSIGN(auto fpe_share_one,
                       fpe_factory_->ImportFixedPointElementFromUint64(
                           random_uint % fpe_params_.primary_ring_modulus));
  ASSERT_OK_AND_ASSIGN(auto fpe_share_zero, fpe - fpe_share_one);

  std::vector<FixedPointElement> fpe_shares_zero = {fpe_share_zero};
  std::vector<FixedPointElement> fpe_shares_one = {fpe_share_one};

  ASSERT_OK_AND_ASSIGN(
      auto computed_sum,
      RunBatchedProtocolForGivenSharing(fpe_shares_zero, fpe_shares_one, ab));

  EXPECT_TRUE(IsComputedResultCloseToRealResult(exponent, computed_sum[0]));
}

}  // namespace
}  // namespace private_join_and_compute
