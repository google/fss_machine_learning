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

#include "secret_sharing_mpc/gates/powers.h"

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

class PowersTest : public Test {
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

// Tests that secure powers protocol works correctly.

// SamplePowersOfRandomVector() Tests

TEST_F(PowersTest, PowPreprocessNonPositiveInputsFails) {
  size_t k = 0, n = 1;
  auto random_powers_shares =
      internal::SamplePowersOfRandomVector(k, n, fp_factory_,
                                           kRingModulus);

  EXPECT_THAT(random_powers_shares,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("k and n must be positive integers")));
}

TEST_F(PowersTest, PowPreprocessNonPositiveInputs2Fails) {
  size_t k = 1, n = 0;
  auto random_powers_shares =
      internal::SamplePowersOfRandomVector(k, n, fp_factory_,
                                           kRingModulus);

  EXPECT_THAT(random_powers_shares,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("k and n must be positive integers")));
}

TEST_F(PowersTest, PowPreprocessSucceeds) {
  size_t k = 3;
  size_t n = 10;
  ASSERT_OK_AND_ASSIGN(
      auto random_powers_shares,
      internal::SamplePowersOfRandomVector(k, n,
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

// GenerateBatchedPowersMessage() Tests

TEST_F(PowersTest, PowMessageEmptyInputFails) {
  std::vector<uint64_t> empty_vector(0);
  std::vector<uint64_t> small_vector{1, 2, 3, 4, 5};
  auto pow_state_and_message =
      GenerateBatchedPowersMessage(empty_vector, {small_vector},
                                   kNumFractionalBits, kRingModulus);
  EXPECT_THAT(pow_state_and_message,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("GenerateBatchedPowersMessage: input "
                                 "must not be empty.")));
}

TEST_F(PowersTest, PowMessageEmptyRandomInputFails) {
  std::vector<uint64_t> small_vector{1, 2, 3, 4, 5};
  auto pow_state_and_message =
      GenerateBatchedPowersMessage(small_vector, {},
                                   kNumFractionalBits, kRingModulus);
  EXPECT_THAT(pow_state_and_message,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("GenerateBatchedPowersMessage: input "
                                 "must not be empty.")));
}

TEST_F(PowersTest, PowMessageEmptyMInputFails) {
  std::vector<uint64_t> empty_vector(0);
  std::vector<uint64_t> small_vector{1, 2, 3, 4, 5};
  auto pow_state_and_message =
      GenerateBatchedPowersMessage(empty_vector,
                                   {small_vector},
                                   kNumFractionalBits,
                                   kRingModulus);
  EXPECT_THAT(pow_state_and_message,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("GenerateBatchedPowersMessage: input "
                                 "must not be empty.")));
}

TEST_F(PowersTest, PowMessageNotEqualLengthSharesOneFails) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4, 5};
  std::vector<uint64_t> small_vector_2{1, 2, 3, 4, 5, 6};
  auto pow_state_and_message = GenerateBatchedPowersMessage(
      small_vector_1, {small_vector_2}, kNumFractionalBits, kRingModulus);
  EXPECT_THAT(pow_state_and_message,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("GenerateBatchedPowersMessage: shares "
                                 "must have the same length.")));
}

TEST_F(PowersTest, PowMessageNotEqualLengthSharesTwoFails) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4, 5};
  std::vector<uint64_t> small_vector_2{1, 2, 3, 4, 5, 6};
  auto pow_state_and_message = GenerateBatchedPowersMessage(
      small_vector_1, {small_vector_1,
                       small_vector_2},
                       kNumFractionalBits,
                       kRingModulus);
  EXPECT_THAT(pow_state_and_message,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("GenerateBatchedPowersMessage: shares "
                                 "must have the same length.")));
}

TEST_F(PowersTest, PowMessageMGreaterThanBSucceeds) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe0,
                       fp_factory_->CreateFixedPointElementFromDouble(10.5));
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe1,
                       fp_factory_->CreateFixedPointElementFromDouble(0.25));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, this->MakePrng());
  std::vector<uint64_t> share_m = {fpe0.ExportToUint64(),
                                   fpe1.ExportToUint64()};
  std::vector<uint64_t> share_b = {fpe1.ExportToUint64(),
                                   fpe1.ExportToUint64()};
  std::vector<std::vector<uint64_t>> random_powers_share {share_b};

  ASSERT_OK_AND_ASSIGN(
      auto pow_state_and_message,
      GenerateBatchedPowersMessage(share_m,
                                   random_powers_share,
                                   kNumFractionalBits,
                                   kRingModulus));
  auto state = pow_state_and_message.first;
  auto message = pow_state_and_message.second;
  EXPECT_EQ(0.25, fp_factory_->ImportFixedPointElementFromUint64(
      message.vector_m_minus_vector_b_shares(0))->ExportToDouble());
  EXPECT_EQ(0, fp_factory_->ImportFixedPointElementFromUint64(
      message.vector_m_minus_vector_b_shares(1))->ExportToDouble());
  EXPECT_EQ(0.25, fp_factory_->ImportFixedPointElementFromUint64(
      state.share_m_minus_b_fractional[0])->ExportToDouble());
  EXPECT_EQ(0, fp_factory_->ImportFixedPointElementFromUint64(
      state.share_m_minus_b_fractional[1])->ExportToDouble());
  EXPECT_EQ(10.25, fp_factory_->ImportFixedPointElementFromUint64(
      state.share_m_minus_b_full[0])->ExportToDouble());
  EXPECT_EQ(0, fp_factory_->ImportFixedPointElementFromUint64(
      state.share_m_minus_b_full[1])->ExportToDouble());
}

TEST_F(PowersTest, PowMessageMLessThanBSucceeds) {
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe0,
                       fp_factory_->CreateFixedPointElementFromDouble(1.5));
  ASSERT_OK_AND_ASSIGN(FixedPointElement fpe1,
                       fp_factory_->CreateFixedPointElementFromDouble(12.25));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, this->MakePrng());
  std::vector<uint64_t> share_m = {fpe0.ExportToUint64(),
                                   fpe1.ExportToUint64()};
  std::vector<uint64_t> share_b = {fpe1.ExportToUint64(),
                                   fpe1.ExportToUint64()};
  std::vector<std::vector<uint64_t>> random_powers_share {share_b};

  ASSERT_OK_AND_ASSIGN(
      auto pow_state_and_message,
      GenerateBatchedPowersMessage(share_m,
                                   random_powers_share,
                                   kNumFractionalBits,
                                   kRingModulus));
  auto state = pow_state_and_message.first;
  auto message = pow_state_and_message.second;
  EXPECT_EQ(0.25, fp_factory_->ImportFixedPointElementFromUint64(
      message.vector_m_minus_vector_b_shares(0))->ExportToDouble());
  EXPECT_EQ(0, fp_factory_->ImportFixedPointElementFromUint64(
      message.vector_m_minus_vector_b_shares(1))->ExportToDouble());
  EXPECT_EQ(0.25, fp_factory_->ImportFixedPointElementFromUint64(
      state.share_m_minus_b_fractional[0])->ExportToDouble());
  EXPECT_EQ(0, fp_factory_->ImportFixedPointElementFromUint64(
      state.share_m_minus_b_fractional[1])->ExportToDouble());
  EXPECT_EQ(-10.75, fp_factory_->ImportFixedPointElementFromUint64(
      state.share_m_minus_b_full[0])->ExportToDouble());
  EXPECT_EQ(0, fp_factory_->ImportFixedPointElementFromUint64(
      state.share_m_minus_b_full[1])->ExportToDouble());
}

TEST_F(PowersTest, PowMessageSmallInputSucceeds) {
  size_t n = 10;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, this->MakePrng());
  ASSERT_OK_AND_ASSIGN(auto share_m,
                       SampleVectorFromPrng(n, kRingModulus, prng.get()));
  ASSERT_OK_AND_ASSIGN(auto share_b,
                       SampleVectorFromPrng(n, kRingModulus, prng.get()));
  std::vector<std::vector<uint64_t>> random_powers_share {share_b};

  ASSERT_OK_AND_ASSIGN(
      auto pow_state_and_message,
      GenerateBatchedPowersMessage(share_m,
                                   random_powers_share,
                                   kNumFractionalBits,
                                   kRingModulus));
  auto state = pow_state_and_message.first;
  auto message = pow_state_and_message.second;
  ASSERT_OK_AND_ASSIGN(auto m_minus_b,
                       BatchedModSub(share_m, share_b, kRingModulus));
  EXPECT_EQ(m_minus_b.size(), message.vector_m_minus_vector_b_shares_size());
  EXPECT_EQ(m_minus_b.size(), state.share_m_minus_b_fractional.size());
  EXPECT_EQ(m_minus_b.size(), state.share_m_minus_b_full.size());
}

// PowersGenerateOTInputsPartyZero() PowersGenerateOTInputsPartyOne() Tests

TEST_F(PowersTest, PowEmptyMminusBStatePartyZeroFails) {
  std::vector<uint64_t> empty_vector(0);
  std::vector<uint64_t> small_vector{1, 2, 3, 4, 5};
  PowersStateMminusB state = {
    .share_m_minus_b_full = std::move(empty_vector),
    .share_m_minus_b_fractional = std::move(small_vector)
    };
  auto ot_inps_and_pow_share = PowersGenerateOTInputsPartyZero(
      state, {}, {}, fp_factory_, kRingModulus);

  EXPECT_THAT(ot_inps_and_pow_share,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("state must not be empty")));
}

TEST_F(PowersTest, PowMminusBNotEqualLengthPartyZeroFails) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4, 5};
  std::vector<uint64_t> small_vector_2{1, 2, 3, 4, 5, 6};
  PowersStateMminusB state = {
    .share_m_minus_b_full = std::move(small_vector_2),
    .share_m_minus_b_fractional = std::move(small_vector_1)
    };
  auto ot_inps_and_pow_share = PowersGenerateOTInputsPartyZero(
      state, {}, {}, fp_factory_, kRingModulus);

  EXPECT_THAT(ot_inps_and_pow_share,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("Powers: state has invalid dimensions")));
}

TEST_F(PowersTest, PowOTInputsInconsistentDimensionsPartyZeroFails) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4, 5};
  std::vector<uint64_t> small_vector_2{1, 2, 3, 4, 5, 6};
  PowersStateMminusB state = {
    .share_m_minus_b_full = small_vector_1,
    .share_m_minus_b_fractional = small_vector_1
  };
  auto ot_inps_and_pow_share = PowersGenerateOTInputsPartyZero(
      state, {small_vector_1, small_vector_2}, {}, fp_factory_, kRingModulus);

  EXPECT_THAT(ot_inps_and_pow_share,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr(
                           "Powers: incorrect number of preprocessed powers")));
}

TEST_F(PowersTest, PowOTInputsWrongMessageDimensionsPartyZeroFails) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4, 5};
  PowersStateMminusB state = {
    .share_m_minus_b_full = small_vector_1,
    .share_m_minus_b_fractional = small_vector_1
  };
  PowersMessageMminusB pow_message;
  pow_message.add_vector_m_minus_vector_b_shares(small_vector_1[0]);
  pow_message.add_vector_m_minus_vector_b_shares(small_vector_1[1]);

  auto ot_inps_and_pow_share = PowersGenerateOTInputsPartyZero(
      state, {small_vector_1, small_vector_1}, pow_message,
      fp_factory_, kRingModulus);

  EXPECT_THAT(ot_inps_and_pow_share,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr(
                           "m - b message size must equal n")));
}

TEST_F(PowersTest, PowEmptyMminusBStatePartyOneFails) {
  std::vector<uint64_t> empty_vector(0);
  std::vector<uint64_t> small_vector{1, 2, 3, 4, 5};
  PowersStateMminusB state = {
    .share_m_minus_b_full = std::move(empty_vector),
    .share_m_minus_b_fractional = std::move(small_vector)
    };
  auto ot_inps_and_pow_share = PowersGenerateOTInputsPartyOne(
      state, {}, {}, fp_factory_, kRingModulus);

  EXPECT_THAT(ot_inps_and_pow_share,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("state must not be empty")));
}

TEST_F(PowersTest, PowMminusBNotEqualLengthPartyOneFails) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4, 5};
  std::vector<uint64_t> small_vector_2{1, 2, 3, 4, 5, 6};
  PowersStateMminusB state = {
    .share_m_minus_b_full = std::move(small_vector_2),
    .share_m_minus_b_fractional = std::move(small_vector_1)
    };
  auto ot_inps_and_pow_share = PowersGenerateOTInputsPartyOne(
      state, {}, {}, fp_factory_, kRingModulus);

  EXPECT_THAT(ot_inps_and_pow_share,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("Powers: state has invalid dimensions")));
}

TEST_F(PowersTest, PowOTInputsInconsistentDimensionsPartyOneFails) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4, 5};
  std::vector<uint64_t> small_vector_2{1, 2, 3, 4, 5, 6};
  PowersStateMminusB state = {
    .share_m_minus_b_full = small_vector_1,
    .share_m_minus_b_fractional = small_vector_1
  };
  auto ot_inps_and_pow_share = PowersGenerateOTInputsPartyOne(
      state, {small_vector_1, small_vector_2}, {}, fp_factory_, kRingModulus);

  EXPECT_THAT(ot_inps_and_pow_share,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr(
                           "Powers: incorrect number of preprocessed powers")));
}

TEST_F(PowersTest, PowOTInputsWrongMessageDimensionsPartyOneFails) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4, 5};
  PowersStateMminusB state = {
    .share_m_minus_b_full = small_vector_1,
    .share_m_minus_b_fractional = small_vector_1
  };
  PowersMessageMminusB pow_message;
  pow_message.add_vector_m_minus_vector_b_shares(small_vector_1[0]);
  pow_message.add_vector_m_minus_vector_b_shares(small_vector_1[1]);

  auto ot_inps_and_pow_share = PowersGenerateOTInputsPartyOne(
      state, {small_vector_1, small_vector_1}, pow_message,
      fp_factory_, kRingModulus);

  EXPECT_THAT(ot_inps_and_pow_share,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr(
                           "m - b message size must equal n")));
}

TEST_F(PowersTest, PowOTInputsPartyZeroSucceeds) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4, 5};
  PowersStateMminusB state = {
    .share_m_minus_b_full = small_vector_1,
    .share_m_minus_b_fractional = small_vector_1
  };
  PowersMessageMminusB pow_message;
  pow_message.add_vector_m_minus_vector_b_shares(small_vector_1[0]);
  pow_message.add_vector_m_minus_vector_b_shares(small_vector_1[1]);
  pow_message.add_vector_m_minus_vector_b_shares(small_vector_1[2]);
  pow_message.add_vector_m_minus_vector_b_shares(small_vector_1[3]);
  pow_message.add_vector_m_minus_vector_b_shares(small_vector_1[4]);

  ASSERT_OK_AND_ASSIGN(auto ot_inps_and_pow_share,
                       PowersGenerateOTInputsPartyZero(
      state, {small_vector_1, small_vector_1}, pow_message,
      fp_factory_, kRingModulus));

  auto ot_inps = ot_inps_and_pow_share.first;
  auto pow_share = ot_inps_and_pow_share.second;
  EXPECT_EQ(ot_inps.receiver_bit.size(), small_vector_1.size());
  EXPECT_EQ(ot_inps.powers_input_0.size(), 2);
  EXPECT_EQ(ot_inps.powers_input_1.size(), 2);
  EXPECT_EQ(ot_inps.powers_input_0[0].size(), small_vector_1.size());
  EXPECT_EQ(ot_inps.powers_input_0[1].size(), small_vector_1.size());
  EXPECT_EQ(ot_inps.powers_input_1[0].size(), small_vector_1.size());
  EXPECT_EQ(ot_inps.powers_input_1[1].size(), small_vector_1.size());
  EXPECT_EQ(pow_share.powers_share.size(), 2);
  EXPECT_EQ(pow_share.powers_share[0].size(), small_vector_1.size());
  EXPECT_EQ(pow_share.powers_share[1].size(), small_vector_1.size());
}

TEST_F(PowersTest, PowOTInputsPartyOneSucceeds) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4, 5};
  PowersStateMminusB state = {
    .share_m_minus_b_full = small_vector_1,
    .share_m_minus_b_fractional = small_vector_1
  };
  PowersMessageMminusB pow_message;
  pow_message.add_vector_m_minus_vector_b_shares(small_vector_1[0]);
  pow_message.add_vector_m_minus_vector_b_shares(small_vector_1[1]);
  pow_message.add_vector_m_minus_vector_b_shares(small_vector_1[2]);
  pow_message.add_vector_m_minus_vector_b_shares(small_vector_1[3]);
  pow_message.add_vector_m_minus_vector_b_shares(small_vector_1[4]);

  ASSERT_OK_AND_ASSIGN(auto ot_inps_and_pow_share,
                       PowersGenerateOTInputsPartyOne(
      state, {small_vector_1, small_vector_1}, pow_message,
      fp_factory_, kRingModulus));

  auto ot_inps = ot_inps_and_pow_share.first;
  auto pow_share = ot_inps_and_pow_share.second;
  EXPECT_EQ(ot_inps.receiver_bit.size(), small_vector_1.size());
  EXPECT_EQ(ot_inps.powers_input_0.size(), 2);
  EXPECT_EQ(ot_inps.powers_input_1.size(), 2);
  EXPECT_EQ(ot_inps.powers_input_0[0].size(), small_vector_1.size());
  EXPECT_EQ(ot_inps.powers_input_0[1].size(), small_vector_1.size());
  EXPECT_EQ(ot_inps.powers_input_1[0].size(), small_vector_1.size());
  EXPECT_EQ(ot_inps.powers_input_1[1].size(), small_vector_1.size());
  EXPECT_EQ(pow_share.powers_share.size(), 2);
  EXPECT_EQ(pow_share.powers_share[0].size(), small_vector_1.size());
  EXPECT_EQ(pow_share.powers_share[1].size(), small_vector_1.size());
}

// PowersOutput Tests

TEST_F(PowersTest, PowersOutputInconsistentDimensionsFails) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4, 5};
  PowersShareOfPowersShare share_one = {
    .powers_share = {small_vector_1}
  };
  PowersShareOfPowersShare share_two = {
    .powers_share = {small_vector_1, small_vector_1}
  };
  auto output_share = PowersOutput(share_one, share_two, kRingModulus);

  EXPECT_THAT(output_share,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr(
                           "share0 and share1 dimensions mismatch")));
}

TEST_F(PowersTest, PowersOutputInconsistentDimensionsTwoFails) {
  std::vector<uint64_t> small_vector_1{1, 2, 3, 4, 5};
  std::vector<uint64_t> small_vector_2{1, 2, 3, 4, 5, 6};
  PowersShareOfPowersShare share_one = {
    .powers_share = {small_vector_1, small_vector_1}
  };
  PowersShareOfPowersShare share_two = {
    .powers_share = {small_vector_1, small_vector_2}
  };
  auto output_share = PowersOutput(share_one, share_two, kRingModulus);

  EXPECT_THAT(output_share,
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr(
                           "share0 and share1 dimensions inconsistent")));
}

TEST_F(PowersTest, PowersOutputSucceeds) {
  std::vector<uint64_t> small_vector{1, 2, 3};
  PowersShareOfPowersShare share_one = {
    .powers_share = {small_vector, small_vector}
  };
  PowersShareOfPowersShare share_two = {
    .powers_share = {small_vector, small_vector}
  };

  ASSERT_OK_AND_ASSIGN(auto output_share,
                       PowersOutput(share_one, share_two, kRingModulus));

  EXPECT_EQ(output_share.size(), share_one.powers_share.size());
  EXPECT_EQ(output_share[0].size(), small_vector.size());
  EXPECT_EQ(output_share[1].size(), small_vector.size());
}

// End-to-end test for secure powers protocol.
// Each party has secret-shared vector input [m] of length n,
// and k, the maximum power to which m should be raised.
// The output are shares of n-element vectors [m], [m^2], ..., [m^k].
// Recall m is in [0-1)
// In this test, m = {0, 1/e, 0.9} * 2^{10}
// and the length of m, n = 3, and k = 3
// This tests all +,0 entries of m
// and Tests for k > 2, (even and odd k, and large k=10)
TEST_F(PowersTest, PowSucceeds) {
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

  // Expected output m^2 in the clear:
  // {0, 0.13533528323, 0.81}.
  // Tests even power
  ASSERT_OK_AND_ASSIGN(
      FixedPointElement fpe_one_2,
      fp_factory_->CreateFixedPointElementFromDouble(0.13533528323));
    ASSERT_OK_AND_ASSIGN(
      FixedPointElement fpe_two_2,
      fp_factory_->CreateFixedPointElementFromDouble(0.81));
  std::vector<uint64_t> vector_m_2{
      zero.ExportToUint64(),
    fpe_one_2.ExportToUint64(), fpe_two_2.ExportToUint64()};
  // Expected output m^3 in the clear:
  // {0, 0.04978706836, 0.729}.
  // Tests odd power
  ASSERT_OK_AND_ASSIGN(
      FixedPointElement fpe_one_3,
      fp_factory_->CreateFixedPointElementFromDouble(0.04978706836));
  ASSERT_OK_AND_ASSIGN(
      FixedPointElement fpe_two_3,
      fp_factory_->CreateFixedPointElementFromDouble(0.729));
  std::vector<uint64_t> vector_m_3{zero.ExportToUint64(),
                                   fpe_one_3.ExportToUint64(),
                                   fpe_two_3.ExportToUint64()};
  // Expected output m^10 in the clear: {0, 0.00004539992, 0.3486784401}.
  // Large power is tested as truncation introduces errors and logistic
  // regression, which uses the powers gate, requires computing high powers
  ASSERT_OK_AND_ASSIGN(
      FixedPointElement fpe_one_10,
      fp_factory_->CreateFixedPointElementFromDouble(0.00004539992));
  ASSERT_OK_AND_ASSIGN(
      FixedPointElement fpe_two_10,
      fp_factory_->CreateFixedPointElementFromDouble(0.3486784401));
  std::vector<uint64_t> vector_m_10{
      zero.ExportToUint64(),
    fpe_one_10.ExportToUint64(), fpe_two_10.ExportToUint64()};

  // Batched powers computation on share of m.

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BasicRng> prng, MakePrng());

  // Generate [b], [b^2], ..., [b^k] for random vector b for P0 and P1.
  ASSERT_OK_AND_ASSIGN(
      auto powers_random_vector,
      internal::SamplePowersOfRandomVector(k, n, fp_factory_,
                                           kRingModulus));
  auto powers_vector_share_0 = powers_random_vector.first;
  auto powers_vector_share_1 = powers_random_vector.second;

  // Generate random shares for vector m and distribute to P0 and P1.
  ASSERT_OK_AND_ASSIGN(auto share_m_0,
                       SampleVectorFromPrng(n, kRingModulus, prng.get()));
  ASSERT_OK_AND_ASSIGN(auto share_m_1,
                       BatchedModSub(vector_m, share_m_0, kRingModulus));

  // Each party generates its batched powers message.
  ASSERT_OK_AND_ASSIGN(auto p0_return, GenerateBatchedPowersMessage(
                                           share_m_0, powers_vector_share_0,
                                           kNumFractionalBits, kRingModulus));
  auto state0 = p0_return.first;
  auto msg0 = p0_return.second;

  ASSERT_OK_AND_ASSIGN(auto p1_return, GenerateBatchedPowersMessage(
                                           share_m_1, powers_vector_share_1,
                                           kNumFractionalBits, kRingModulus));
  auto state1 = p1_return.first;
  auto msg1 = p1_return.second;

  // Compute inputs to the OTs and invoking party's share of share of powers

  ASSERT_OK_AND_ASSIGN(auto ot_inps_powers_share_0,
  PowersGenerateOTInputsPartyZero(
    state0, powers_vector_share_0, msg1, fp_factory_, kRingModulus));

  auto ot_inputs_0 = ot_inps_powers_share_0.first;
  auto share_of_share_0_p0 = ot_inps_powers_share_0.second;

  ASSERT_OK_AND_ASSIGN(auto ot_inps_powers_share_1,
  PowersGenerateOTInputsPartyOne(
    state1, powers_vector_share_1, msg0, fp_factory_, kRingModulus));

  auto ot_inputs_1 = ot_inps_powers_share_1.first;
  auto share_of_share_0_p1 = ot_inps_powers_share_1.second;

  // Generate OT Outputs via an insecure test function
  ASSERT_OK_AND_ASSIGN(auto ot_outs,
  PowersGenerateOTOutputForTesting(
    ot_inputs_0, ot_inputs_1, fp_factory_, kRingModulus));

  auto share_of_share_1_p0 = ot_outs.first;
  auto share_of_share_1_p1 = ot_outs.second;

  // Each party computes its share of output.
  ASSERT_OK_AND_ASSIGN(
      auto out_share_m_0,
      PowersOutput(share_of_share_0_p0, share_of_share_1_p0, kRingModulus));
  ASSERT_OK_AND_ASSIGN(
      auto out_share_m_1,
      PowersOutput(share_of_share_0_p1, share_of_share_1_p1, kRingModulus));

  // Reconstruct output and verify the correctness.
  std::vector<std::vector<uint64_t>> reconstructed_powers(
      k, std::vector<uint64_t>(n));
  for (size_t idx = 0; idx < k; idx++) {
    ASSERT_OK_AND_ASSIGN(
        reconstructed_powers[idx],
        BatchedModAdd(out_share_m_0[idx], out_share_m_1[idx], kRingModulus));
  }

  for (size_t idx = 0; idx < n; idx++) {
    // Test with doubles instead of uint64_t as 0 with minor error can produce
    // vastly different uint64_t (think very small negative number)
    EXPECT_NEAR(
        fp_factory_
            ->ImportFixedPointElementFromUint64(reconstructed_powers[0][idx])
            ->ExportToDouble(),
        fp_factory_->ImportFixedPointElementFromUint64(vector_m[idx])
            ->ExportToDouble(),
        0.0001);
    // Test 2nd power
    EXPECT_NEAR(
        fp_factory_
            ->ImportFixedPointElementFromUint64(reconstructed_powers[1][idx])
            ->ExportToDouble(),
        fp_factory_->ImportFixedPointElementFromUint64(vector_m_2[idx])
            ->ExportToDouble(),
        0.0001);
    // Test 3rd power
    EXPECT_NEAR(
        fp_factory_
            ->ImportFixedPointElementFromUint64(reconstructed_powers[2][idx])
            ->ExportToDouble(),
        fp_factory_->ImportFixedPointElementFromUint64(vector_m_3[idx])
            ->ExportToDouble(),
        0.0001);
    // Test 10th power
    EXPECT_NEAR(
        fp_factory_
            ->ImportFixedPointElementFromUint64(reconstructed_powers[9][idx])
            ->ExportToDouble(),
        fp_factory_->ImportFixedPointElementFromUint64(vector_m_10[idx])
            ->ExportToDouble(),
        0.0002);
  }
}

}  // namespace
}  // namespace private_join_and_compute
