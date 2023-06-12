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

#include "secret_sharing_mpc/gates/polynomial.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "private_join_and_compute/util/status_testing.inc"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace private_join_and_compute {
namespace {

using ::testing::HasSubstr;
using testing::StatusIs;
using ::testing::Test;

// l_f in secure poisson regression paper
const size_t kNumFractionalBits = 20;
// l in secure poisson regression paper
const size_t kNumRingBits = 63;
const uint64_t kRingModulus = (1ULL << 63);

// {20, 63, 2^10, 2^43, 2^63}
const FixedPointElementFactory::Params kSampleParams = {
        kNumFractionalBits, kNumRingBits, (1ULL << kNumFractionalBits),
        (1ULL << (kNumRingBits - kNumFractionalBits)), (1ULL << kNumRingBits)};

class PolynomialTest : public Test {
 protected:
  void SetUp() override {
      // Create a sample 63-bit factory.
      ASSERT_OK_AND_ASSIGN(
              FixedPointElementFactory temp63,
              FixedPointElementFactory::Create(kSampleParams.num_fractional_bits,
                                               kSampleParams.num_ring_bits));
      fp_factory_ =
              absl::make_unique<FixedPointElementFactory>(std::move(temp63));
  }
  StatusOr<std::unique_ptr<BasicRng>> MakePrng() {
      auto random_seed = BasicRng::GenerateSeed();
      if (!random_seed.ok()) {
          return InternalError("Random seed generation fails.");
      }
      return BasicRng::Create(random_seed.value());
  }
  std::unique_ptr<FixedPointElementFactory> fp_factory_;
};


// Tests that secure polynomial protocol works correctly.

// SamplePowersOfRandomVector() Tests

TEST_F(PolynomialTest, PolynomialPreprocessNonPositiveInputsFails) {
    size_t k = 0, n = 1;
    auto random_powers_shares =
            internal::PolynomialSamplePowersOfRandomVector(k, n, fp_factory_,
                    kRingModulus);

    EXPECT_THAT(random_powers_shares,
            StatusIs(StatusCode::kInvalidArgument,
                     HasSubstr("k and n must be positive integers")));
}

TEST_F(PolynomialTest, PolynomialPreprocessNonPositiveInputs2Fails) {
    size_t k = 1, n = 0;
    auto random_powers_shares =
            internal::PolynomialSamplePowersOfRandomVector(k, n, fp_factory_,
                                                 kRingModulus);

    EXPECT_THAT(random_powers_shares,
            StatusIs(StatusCode::kInvalidArgument,
                     HasSubstr("k and n must be positive integers")));
}

TEST_F(PolynomialTest, PolynomialPreprocessSucceeds) {
    size_t k = 3;
    size_t n = 10;
    ASSERT_OK_AND_ASSIGN(
            auto random_powers_shares,
            internal::PolynomialSamplePowersOfRandomVector(k, n,
                                                 fp_factory_,
                                                 kRingModulus));
    auto random_powers_0 = random_powers_shares.first;  // P_0 share
    auto random_powers_1 = random_powers_shares.second;  // P_1 share

    EXPECT_EQ(random_powers_0.size(), k);
    EXPECT_EQ(random_powers_1.size(), k);
    for (size_t idx = 0; idx < k; idx++) {
        EXPECT_EQ(random_powers_0[idx].size(), n);
        EXPECT_EQ(random_powers_1[idx].size(), n);
    }

    std::vector<uint64_t> vector_1, vector_2, vector_3;  // b, b^2, b^3
    vector_1.reserve(n);
    vector_2.reserve(n);
    vector_3.reserve(n);

    // Reconstruct the vectors 1, 2, 3.
    for (size_t idx = 0; idx < n; idx++) {
        vector_1[idx] =
        ModAdd(random_powers_0[0][idx], random_powers_1[0][idx], kRingModulus);
        vector_2[idx] =
        ModAdd(random_powers_0[1][idx], random_powers_1[1][idx], kRingModulus);
        vector_3[idx] =
        ModAdd(random_powers_0[2][idx], random_powers_1[2][idx], kRingModulus);
  }

    for (size_t idx = 0; idx < n; idx++) {
        ASSERT_OK_AND_ASSIGN(
                FixedPointElement vector_1_idx,
                fp_factory_->ImportFixedPointElementFromUint64(vector_1[idx]));
        ASSERT_OK_AND_ASSIGN(FixedPointElement vector_2_idx,
                             vector_1_idx.TruncMulFP(vector_1_idx));
        ASSERT_OK_AND_ASSIGN(FixedPointElement vector_3_idx,
                             vector_1_idx.TruncMulFP(vector_2_idx));
        EXPECT_EQ(vector_2[idx], vector_2_idx.ExportToUint64());
        EXPECT_EQ(vector_3[idx], vector_3_idx.ExportToUint64());
    }
}

TEST_F(PolynomialTest, PolynomialPreprocessOTsNonPositiveInputFails) {
size_t n = 0;
auto preprocessed_ots =
    internal::PolynomialPreprocessRandomOTs(n, fp_factory_);

EXPECT_THAT(preprocessed_ots,
    StatusIs(StatusCode::kInvalidArgument,
             HasSubstr("n must be a positive integer")));
}

TEST_F(PolynomialTest, PolynomialPreprocessRandomOTsSucceeds) {
  size_t n = 10;
  ASSERT_OK_AND_ASSIGN(
      auto random_ots,
      internal::PolynomialPreprocessRandomOTs(n, fp_factory_));
  auto random_ots_0 = random_ots.first;  // P_0
  auto random_ots_1 = random_ots.second;  // P_1

  EXPECT_EQ(random_ots_0.sender_msgs.size(), n);
  EXPECT_EQ(random_ots_0.receiver_msgs.size(), n);
  EXPECT_EQ(random_ots_1.sender_msgs.size(), n);
  EXPECT_EQ(random_ots_1.receiver_msgs.size(), n);
  for (size_t idx = 0; idx < n; idx++) {
    if (random_ots_0.receiver_msgs[idx].receiver_choice) {
      EXPECT_EQ(random_ots_0.receiver_msgs[idx].receiver_msg,
                random_ots_1.sender_msgs[idx].sender_msg1);
    } else {
      EXPECT_EQ(random_ots_0.receiver_msgs[idx].receiver_msg,
                random_ots_1.sender_msgs[idx].sender_msg0);
    }
    if (random_ots_1.receiver_msgs[idx].receiver_choice) {
      EXPECT_EQ(random_ots_1.receiver_msgs[idx].receiver_msg,
                random_ots_0.sender_msgs[idx].sender_msg1);
    } else {
      EXPECT_EQ(random_ots_1.receiver_msgs[idx].receiver_msg,
                random_ots_0.sender_msgs[idx].sender_msg0);
    }
  }
}

// PolynomialGenerateRoundOneMessage() Tests

TEST_F(PolynomialTest, PolynomialGenerateRoundOneMessageEmptyInputFails) {
    std::vector<uint64_t> empty_vector(0);
    std::vector<uint64_t> small_vector{1, 2, 3, 4, 5};
    PolynomialRandomOTPrecomputation precomputed_ots = {
        .sender_msgs = { {.sender_msg0 = 1, .sender_msg1 = 1 } },
        .receiver_msgs = { {.receiver_choice = false, .receiver_msg = 1 } }
    };
    auto round_one_state_and_message =
        PolynomialGenerateRoundOneMessage(empty_vector, {small_vector}, precomputed_ots,
                                         fp_factory_, kRingModulus);
    EXPECT_THAT(round_one_state_and_message,
            StatusIs(StatusCode::kInvalidArgument,
                     HasSubstr("PolynomialGenerateRoundOneMessage: input "
                               "must not be empty.")));
}

TEST_F(PolynomialTest, PolynomialGenerateRoundOneMessageEmptyRandomInputFails) {
std::vector<uint64_t> small_vector{1, 2, 3, 4, 5};
PolynomialRandomOTPrecomputation precomputed_ots = {
    .sender_msgs = { {.sender_msg0 = 1, .sender_msg1 = 1 } },
    .receiver_msgs = { {.receiver_choice = false, .receiver_msg = 1 } }
};
auto round_one_state_and_message =
    PolynomialGenerateRoundOneMessage(small_vector, {}, precomputed_ots,
                                      fp_factory_, kRingModulus);
EXPECT_THAT(round_one_state_and_message,
    StatusIs(StatusCode::kInvalidArgument,
             HasSubstr("PolynomialGenerateRoundOneMessage: input "
                       "must not be empty.")));
}

TEST_F(PolynomialTest, PolynomialGenerateRoundOneMessageIncorrectPowersDimensionsFails) {
  std::vector<uint64_t> small_vector_one{1, 2, 3, 4, 5};
  std::vector<uint64_t> small_vector_two{2, 3, 4, 5};
  PolynomialRandomOTPrecomputation precomputed_ots = {
      .sender_msgs = { {.sender_msg0 = 1, .sender_msg1 = 1 } },
      .receiver_msgs = { {.receiver_choice = false, .receiver_msg = 1 } }
  };
  auto round_one_state_and_message =
      PolynomialGenerateRoundOneMessage(small_vector_one, {small_vector_one, small_vector_two}, precomputed_ots,
                                        fp_factory_, kRingModulus);
  EXPECT_THAT(round_one_state_and_message,
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("shares must have the same length")));
}

TEST_F(PolynomialTest, PolynomialGenerateRoundOneMessageIncorrectOTDimensionsFails) {
  std::vector<uint64_t> small_vector_one{1, 2, 3};
  PolynomialRandomOTPrecomputation precomputed_ots = {
      .sender_msgs = { {.sender_msg0 = 1, .sender_msg1 = 1 },
                       {.sender_msg0 = 1, .sender_msg1 = 1 },
                       {.sender_msg0 = 1, .sender_msg1 = 1 }},
      .receiver_msgs = { {.receiver_choice = false, .receiver_msg = 1 },
                         {.receiver_choice = false, .receiver_msg = 1 }}
  };
  auto round_one_state_and_message =
      PolynomialGenerateRoundOneMessage(small_vector_one, {small_vector_one, small_vector_one}, precomputed_ots,
                                        fp_factory_, kRingModulus);
  EXPECT_THAT(round_one_state_and_message,
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("precomputed random ots invalid dimension")));
}

TEST_F(PolynomialTest, PolynomialGenerateRoundOneMessageSucceeds) {
  std::vector<uint64_t> small_vector_one{1, 2, 3};
  size_t n = small_vector_one.size();
  PolynomialRandomOTPrecomputation precomputed_ots = {
      .sender_msgs = { {.sender_msg0 = 1, .sender_msg1 = 1 },
                       {.sender_msg0 = 1, .sender_msg1 = 1 },
                       {.sender_msg0 = 1, .sender_msg1 = 1 } },
      .receiver_msgs = { {.receiver_choice = false, .receiver_msg = 1 },
                         {.receiver_choice = false, .receiver_msg = 1 },
                         {.receiver_choice = false, .receiver_msg = 1 } }
  };
  ASSERT_OK_AND_ASSIGN(auto round_one_state_and_message,
      PolynomialGenerateRoundOneMessage(small_vector_one,
          {small_vector_one, small_vector_one},
          precomputed_ots, fp_factory_, kRingModulus));
  auto state = round_one_state_and_message.first;
  auto message = round_one_state_and_message.second;
  EXPECT_EQ(state.ot_receiver_bit.size(), n);
  EXPECT_EQ(state.share_m_minus_b_fractional.size(), n);
  EXPECT_EQ(message.vector_m_minus_vector_b_shares_size(), n);
  EXPECT_EQ(message.receiver_bit_xor_rot_choice_bit_size(), n);
}

// PolynomialGenerateRoundTwoMessagePartyZero/One() Tests

TEST_F(PolynomialTest, PolynomialGenerateRoundTwoMessagePartyZeroEmptyStateFails) {
  PolynomialCoefficients coefs = { .coefficients = {} };
  PowersStateRoundOne state = {
    .ot_receiver_bit = {},
    .share_m_minus_b_fractional = {}
  };
  PolynomialRandomOTPrecomputation precomputed_ots = {
      .sender_msgs = { {.sender_msg0 = 1, .sender_msg1 = 1 } },
      .receiver_msgs = { {.receiver_choice = false, .receiver_msg = 1 } }
  };
  PowersMessageRoundOne other_party_message;
  auto round_two_share_and_message =
      PolynomialGenerateRoundTwoMessagePartyZero(coefs, state, {}, precomputed_ots,
                                        other_party_message, fp_factory_, kRingModulus);

  EXPECT_THAT(round_two_share_and_message,
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("Polynomial: state must not be empty")));
}

TEST_F(PolynomialTest, PolynomialGenerateRoundTwoMessagePartyOneEmptyStateFails) {
PolynomialCoefficients coefs = { .coefficients = {} };
PowersStateRoundOne state = {
    .ot_receiver_bit = {},
    .share_m_minus_b_fractional = {}
};
PolynomialRandomOTPrecomputation precomputed_ots = {
    .sender_msgs = { {.sender_msg0 = 1, .sender_msg1 = 1 } },
    .receiver_msgs = { {.receiver_choice = false, .receiver_msg = 1 } }
};
PowersMessageRoundOne other_party_message;
auto round_two_share_and_message =
    PolynomialGenerateRoundTwoMessagePartyOne(coefs, state, {}, precomputed_ots,
                                               other_party_message, fp_factory_, kRingModulus);

EXPECT_THAT(round_two_share_and_message,
    StatusIs(StatusCode::kInvalidArgument,
             HasSubstr("Polynomial: state must not be empty")));
}

// PolynomialOutput() Tests

// End-to-end test for secure polynomial protocol.
// Each party has secret-shared vector input [m] of length n,
// and k, the maximum power to which m should be raised.
// Public are the coefficients a_0 ... a_k
// The output are shares of n-element polynomials [a_0 + a_1 * m, ..., a_k * m^k].
// Recall m is in [0-1)
// In this test, m = {0, 1/e, 0.9} * 2^{10}
// and the length of m, n = 3, and k = 3
// This tests all +,0 entries of m
// and Tests for k > 2, (even and odd k, and large k=10)
TEST_F(PolynomialTest, PolSucceeds) {
    // fpe = 0.36787944117 (~1/e) and represented as 1/e * 2^{10}.
    // This is ~approx max for m in logistic regression application
    ASSERT_OK_AND_ASSIGN(
            FixedPointElement fpe_one,
            fp_factory_->CreateFixedPointElementFromDouble(0.36787944117));
    // Note that we will use m no greater than 1/e, but powers should work for
    // m in [0,1)
    ASSERT_OK_AND_ASSIGN(
            FixedPointElement fpe_two,
            fp_factory_->CreateFixedPointElementFromDouble(0.9));
    // zero = 0 and represented as 0 * 2^{10}.
    ASSERT_OK_AND_ASSIGN(FixedPointElement zero,
                         fp_factory_->CreateFixedPointElementFromInt(0));
    // Initialize m in the clear in ring Z_kRingModulus.
    // m = {0, 1/e} * 2^{10}
    std::vector<uint64_t> vector_m{zero.ExportToUint64(),
                                   fpe_one.ExportToUint64(),
                                   fpe_two.ExportToUint64()};

    size_t k = 10;
    size_t n = vector_m.size();

    PolynomialCoefficients coefficients = {
            .coefficients = {1,-1,1,-1,1,-1,1,-1,1,-1,1}
    };

    // Expected output m^10 in the clear: {1, 0.73107078855, 0.6914792611}.
    // Large power is tested as truncation introduces errors and logistic
    // regression, which uses the polynomial gate, requires computing high degrees
    ASSERT_OK_AND_ASSIGN(
            FixedPointElement fpe_zero_10,
            fp_factory_->CreateFixedPointElementFromInt(1));
    ASSERT_OK_AND_ASSIGN(
            FixedPointElement fpe_one_10,
            fp_factory_->CreateFixedPointElementFromDouble(0.73107078855));
    ASSERT_OK_AND_ASSIGN(
            FixedPointElement fpe_two_10,
            fp_factory_->CreateFixedPointElementFromDouble(0.6914792611));
    std::vector<uint64_t> vector_m_10 {
            fpe_zero_10.ExportToUint64(),
            fpe_one_10.ExportToUint64(),
            fpe_two_10.ExportToUint64()};

    // Batched polynomials computation on share of m.

    ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, MakePrng());

    // Generate [b], [b^2], ..., [b^k] for random vector b for P0 and P1.
    ASSERT_OK_AND_ASSIGN(
            auto powers_random_vector,
            internal::PolynomialSamplePowersOfRandomVector(
                    k, n, fp_factory_, kRingModulus));
    auto powers_vector_share_0 = powers_random_vector.first;
    auto powers_vector_share_1 = powers_random_vector.second;

    // Generate n Random Oblivious Transfers for P0 and P1.
    ASSERT_OK_AND_ASSIGN(
        auto random_ots,
        internal::PolynomialPreprocessRandomOTs(
            n, fp_factory_));
    auto random_ots_0 = random_ots.first;
    auto random_ots_1 = random_ots.second;

    // Generate random shares for vector m and distribute to P0 and P1.
    ASSERT_OK_AND_ASSIGN(auto share_m_0,
                         SampleVectorFromPrng(n, kRingModulus, prng.get()));
    ASSERT_OK_AND_ASSIGN(auto share_m_1,
                         BatchedModSub(vector_m, share_m_0, kRingModulus));

    // Each party generates its batched powers message and the first message
    // to convert random ots to 1-2 ots.
    ASSERT_OK_AND_ASSIGN(auto round_one_0, PolynomialGenerateRoundOneMessage(
            share_m_0, powers_vector_share_0, random_ots_0,
            fp_factory_, kRingModulus));
    auto state_round_one_0 = round_one_0.first;
    auto msg_round_one_0 = round_one_0.second;

    ASSERT_OK_AND_ASSIGN(auto round_one_1, PolynomialGenerateRoundOneMessage(
        share_m_1, powers_vector_share_1, random_ots_1,
        fp_factory_, kRingModulus));
    auto state_round_one_1 = round_one_1.first;
    auto msg_round_one_1 = round_one_1.second;
    // Compute inputs to the OTs (2 polynomial evaluations)
    // and construct message for the second round
    ASSERT_OK_AND_ASSIGN(auto round_two_0, PolynomialGenerateRoundTwoMessagePartyZero(
        coefficients, state_round_one_0, powers_vector_share_0, random_ots_0,
        msg_round_one_1, fp_factory_, kRingModulus));
    auto share_of_share_0 = round_two_0.first;
    auto msg_round_two_0 = round_two_0.second;
    ASSERT_OK_AND_ASSIGN(auto round_two_1, PolynomialGenerateRoundTwoMessagePartyOne(
        coefficients, state_round_one_1, powers_vector_share_1, random_ots_1,
        msg_round_one_0, fp_factory_, kRingModulus));
    auto share_of_share_1 = round_two_1.first;
    auto msg_round_two_1 = round_two_1.second;
    // Compute Share of Polynomial output by unmasking round 2 message and adding shares
    ASSERT_OK_AND_ASSIGN(
            auto out_share_0,
            PolynomialOutput(msg_round_two_1, random_ots_0,
                share_of_share_0, state_round_one_0, kRingModulus));
    ASSERT_OK_AND_ASSIGN(
        auto out_share_1,
        PolynomialOutput(msg_round_two_0, random_ots_1,
                         share_of_share_1, state_round_one_1, kRingModulus));
    // Reconstruct output and verify the correctness.
    std::vector<uint64_t> polynomial_sum(n, 0);
    ASSERT_OK_AND_ASSIGN(
            polynomial_sum,
            BatchedModAdd(out_share_0, out_share_1, kRingModulus));

    for (size_t idx = 0; idx < n; idx++) {

        // Test with doubles instead of uint64_t as 0 with minor error can produce
        // vastly different uint64_t (think very small negative number)
        EXPECT_NEAR(
                fp_factory_
        ->ImportFixedPointElementFromUint64(polynomial_sum[idx])
        ->ExportToDouble(),
                fp_factory_->ImportFixedPointElementFromUint64(vector_m_10[idx])
        ->ExportToDouble(),
        0.0002);
    }
}

}  // namespace
}  // namespace private_join_and_compute
