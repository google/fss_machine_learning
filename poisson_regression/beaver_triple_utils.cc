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

#include "poisson_regression/beaver_triple_utils.h"

#include "poisson_regression/ring_arithmetic_utils.h"

namespace private_join_and_compute {
// Helper function to sample a random vector that contains elements of type
// uint64_t. The function takes 3 inputs: length of the desired output,
// a pseudorandom generator PRNG, and the modulus of the ring.
StatusOr<std::vector<uint64_t>> SampleVectorFromPrng(
    size_t length, uint64_t modulus, SecurePrng* prng) {
  if (length < 1) {
    return InvalidArgumentError(
        "SampleVectorFromPrng: length of vector must be a positive integer.");
  }
  std::vector<uint64_t> random_vector;
  random_vector.reserve(length);

  for (size_t i = 0; i < length; i++) {
    auto random_value = prng->Rand64();
    if (!random_value.ok()) {
      return InternalError("Fail to obtain random value.");
    }
    random_vector.push_back(random_value.value() % modulus);
  }

  return std::move(random_vector);
}

// Helper function to generate the message for Multiplication Gate.
StatusOr<std::pair<BatchedMultState, MultiplicationGateMessage>>
GenerateBatchedMultiplicationGateMessage(
    const std::vector<uint64_t>& share_x, const std::vector<uint64_t>& share_y,
    const BeaverTripleVector<uint64_t>& beaver_vector_share, uint64_t modulus) {
  if (share_x.empty() || share_y.empty()) {
    return InvalidArgumentError("GenerateBatchedMultiplicationGateMessage: "
                                "input must not be empty.");
  }
  if (share_x.size() != share_y.size()) {
    return InvalidArgumentError("GenerateBatchedMultiplicationGateMessage: "
                                "shares must have the same length.");
  }

  const auto& share_a = beaver_vector_share.GetA();
  const auto& share_b = beaver_vector_share.GetB();

  if (share_x.size() != share_a.size() || share_y.size() != share_b.size()) {
    return InvalidArgumentError("GenerateBatchedMultiplicationGateMessage: "
                                "input and beaver shares have different size.");
  }

  ASSIGN_OR_RETURN(auto x_minus_a, BatchedModSub(share_x, share_a, modulus));
  ASSIGN_OR_RETURN(auto y_minus_b, BatchedModSub(share_y, share_b, modulus));

  MultiplicationGateMessage mult_message;
  for (size_t idx = 0; idx < x_minus_a.size(); idx++) {
    mult_message.add_vector_x_minus_vector_a_shares(x_minus_a[idx]);
    mult_message.add_vector_y_minus_vector_b_shares(y_minus_b[idx]);
  }
  BatchedMultState state = {.share_x_minus_a = std::move(x_minus_a),
                 .share_y_minus_b = std::move(y_minus_b)};
  return std::make_pair(std::move(state), std::move(mult_message));
}

// Generate the output of the batched multiplication gate for Party Zero.
// Each party takes input: ([X]_i, [Y]_i, [Beaver]_i) for i = 0, 1.
// Each party outputs share of X[i]*Y[i], where * represents multiplication
// mod modulus.
StatusOr<std::vector<uint64_t>> GenerateBatchedMultiplicationOutputPartyZero(
    BatchedMultState state,
    const BeaverTripleVector<uint64_t>& beaver_vector_share,
    MultiplicationGateMessage other_party_message,
    uint64_t modulus) {
  if (state.share_x_minus_a.empty() || state.share_y_minus_b.empty()) {
    return InvalidArgumentError("GenerateBatchedMultiplicationOutput: "
                                "input must not be empty.");
  }

  size_t length = state.share_x_minus_a.size();
  if (state.share_y_minus_b.size() != length ||
      static_cast<size_t>(
      other_party_message.vector_x_minus_vector_a_shares_size()) != length ||
      static_cast<size_t>(
      other_party_message.vector_y_minus_vector_b_shares_size()) != length) {
    return InvalidArgumentError("GenerateBatchedMultiplicationOutput: "
                                "shares must have the same size.");
  }

  // Reconstruct (X-A) and (Y-B).
  std::vector<uint64_t> x_minus_a(length);
  std::vector<uint64_t> y_minus_b(length);
  for (size_t idx = 0; idx < length; idx++) {
    x_minus_a[idx] =
        ModAdd(other_party_message.vector_x_minus_vector_a_shares(idx),
               state.share_x_minus_a[idx], modulus);
    y_minus_b[idx] =
        ModAdd(other_party_message.vector_y_minus_vector_b_shares(idx),
               state.share_y_minus_b[idx], modulus);
  }

  // Compute share [B]_0*(X-A).
  ASSIGN_OR_RETURN(auto share_b_muliply_x_minus_a,
                   BatchedModMul(beaver_vector_share.GetB(),
                                 x_minus_a, modulus));
  // Compute share [A]_0*(Y-B).
  ASSIGN_OR_RETURN(auto share_a_multiply_y_minus_b,
                   BatchedModMul(beaver_vector_share.GetA(),
                                 y_minus_b, modulus));
  // Compute (X-A)*(Y-B). Only Party Zero needs to compute this.
  ASSIGN_OR_RETURN(auto x_minus_a_multiply_y_minus_b,
                   BatchedModMul(x_minus_a, y_minus_b, modulus));
  // Generate output: [XY]_0 = [B]_0*(X-A) + [A]_0*(Y-B) + [C]_0 + (X-A)*(Y-B).
  ASSIGN_OR_RETURN(auto share_xy,
                   BatchedModAdd(share_b_muliply_x_minus_a,
                                 share_a_multiply_y_minus_b, modulus));
  ASSIGN_OR_RETURN(share_xy,
                   BatchedModAdd(beaver_vector_share.GetC(),
                                 share_xy, modulus));
  ASSIGN_OR_RETURN(share_xy,
                   BatchedModAdd(share_xy,
                                 x_minus_a_multiply_y_minus_b, modulus));
  return share_xy;
}

// Generate the output of the batched multiplication gate for Party One.
StatusOr<std::vector<uint64_t>> GenerateBatchedMultiplicationOutputPartyOne(
    BatchedMultState state,
    const BeaverTripleVector<uint64_t>& beaver_vector_share,
    MultiplicationGateMessage other_party_message,
    uint64_t modulus) {
  if (state.share_x_minus_a.empty() || state.share_y_minus_b.empty()) {
    return InvalidArgumentError("GenerateBatchedMultiplicationOutput: "
                                "input must not be empty.");
  }

  size_t length = state.share_x_minus_a.size();
  if (state.share_y_minus_b.size() != length ||
      static_cast<size_t>(
      other_party_message.vector_x_minus_vector_a_shares_size()) != length ||
      static_cast<size_t>(
      other_party_message.vector_y_minus_vector_b_shares_size()) != length) {
    return InvalidArgumentError("GenerateBatchedMultiplicationOutput: "
                                "shares must have the same size.");
  }

  // Reconstruct (X-A) and (Y-B).
  std::vector<uint64_t> x_minus_a(length);
  std::vector<uint64_t> y_minus_b(length);
  for (size_t idx = 0; idx < length; idx++) {
    x_minus_a[idx] =
        ModAdd(other_party_message.vector_x_minus_vector_a_shares(idx),
               state.share_x_minus_a[idx], modulus);
    y_minus_b[idx] =
        ModAdd(other_party_message.vector_y_minus_vector_b_shares(idx),
               state.share_y_minus_b[idx], modulus);
  }

  // Compute share [B]_1*(X-A).
  ASSIGN_OR_RETURN(auto share_b_muliply_x_minus_a,
                   BatchedModMul(beaver_vector_share.GetB(),
                                 x_minus_a, modulus));
  // Compute share [A]_1*(Y-B).
  ASSIGN_OR_RETURN(auto share_a_multiply_y_minus_b,
                   BatchedModMul(beaver_vector_share.GetA(),
                                 y_minus_b, modulus));
  // Generate output: [XY]_1 = [B]_1*(X-A) + [A]_1*(Y-B) + [C]_1.
  ASSIGN_OR_RETURN(auto share_xy,
                   BatchedModAdd(share_b_muliply_x_minus_a,
                                 share_a_multiply_y_minus_b, modulus));
  ASSIGN_OR_RETURN(share_xy,
                   BatchedModAdd(beaver_vector_share.GetC(),
                                 share_xy, modulus));
  return share_xy;
}

// Helper function to generate the message for Matrix Multiplication Gate.
StatusOr<std::pair<MatrixMultState, MatrixMultiplicationGateMessage>>
GenerateMatrixMultiplicationGateMessage(
    const std::vector<uint64_t>& share_x, const std::vector<uint64_t>& share_y,
    const BeaverTripleMatrix<uint64_t>& beaver_matrix_share, uint64_t modulus) {
  if (share_x.empty() || share_y.empty()) {
    return InvalidArgumentError("GenerateMatrixMultiplicationGateMessage: "
                                "input must not be empty.");
  }
  const auto& share_a = beaver_matrix_share.GetA();
  const auto& share_b = beaver_matrix_share.GetB();

  if (share_x.size() != share_a.size() || share_y.size() != share_b.size()) {
    return InvalidArgumentError("GenerateMatrixMultiplicationGateMessage: "
                                "input and beaver shares have different size.");
  }
  ASSIGN_OR_RETURN(auto x_minus_a, BatchedModSub(share_x, share_a, modulus));
  ASSIGN_OR_RETURN(auto y_minus_b, BatchedModSub(share_y, share_b, modulus));

  MatrixMultiplicationGateMessage matrix_mult_message;
  for (size_t idx = 0; idx < x_minus_a.size(); idx++) {
    matrix_mult_message.add_matrix_x_minus_matrix_a_shares(x_minus_a[idx]);
  }
  for (size_t idx = 0; idx < y_minus_b.size(); idx++) {
    matrix_mult_message.add_matrix_y_minus_matrix_b_shares(y_minus_b[idx]);
  }
  MatrixMultState state = {.share_x_minus_a = std::move(x_minus_a),
                           .share_y_minus_b = std::move(y_minus_b)};
  return std::make_pair(std::move(state), std::move(matrix_mult_message));
}

// Functions to generate the output of the matrix multiplication gate for P0.
// Generate the output of the matrix multiplication gate.
// Each party takes input: ([X]_i, [Y]_i, [Beaver]_i) for i = 0, 1.
// Each party outputs share of X*Y, where * represents matrix multiplication
// mod modulus.
StatusOr<std::vector<uint64_t>> GenerateMatrixMultiplicationOutputPartyZero(
    MatrixMultState state,
    const BeaverTripleMatrix<uint64_t>& beaver_matrix_share,
    MatrixMultiplicationGateMessage other_party_message,
    uint64_t modulus) {
  if (state.share_x_minus_a.empty() || state.share_y_minus_b.empty()) {
    return InvalidArgumentError("GenerateMatrixMultiplicationOutput: "
                                "input must not be empty.");
  }
  if (state.share_x_minus_a.size() != beaver_matrix_share.GetA().size() ||
      state.share_y_minus_b.size() != beaver_matrix_share.GetB().size()) {
    return InvalidArgumentError("GenerateMatrixMultiplicationOutput: "
                                "input and beaver share dimension mismatch.");
  }
  // Reconstruct (X-A) and (Y-B).
  std::vector<uint64_t> x_minus_a(state.share_x_minus_a.size());
  std::vector<uint64_t> y_minus_b(state.share_y_minus_b.size());
  for (size_t idx = 0; idx < x_minus_a.size(); idx++) {
    x_minus_a[idx] =
        ModAdd(other_party_message.matrix_x_minus_matrix_a_shares(idx),
               state.share_x_minus_a[idx], modulus);
  }
  for (size_t idx = 0; idx < y_minus_b.size(); idx++) {
    y_minus_b[idx] =
        ModAdd(other_party_message.matrix_y_minus_matrix_b_shares(idx),
               state.share_y_minus_b[idx], modulus);
  }

  auto dimensions = beaver_matrix_share.GetDimensions();
  auto dim1 = std::get<0>(dimensions);
  auto dim2 = std::get<1>(dimensions);
  auto dim3 = std::get<2>(dimensions);
  // Compute share (X-A)*[B]_0.
  ASSIGN_OR_RETURN(auto x_minus_a_muliply_share_b,
                   ModMatrixMul(x_minus_a, beaver_matrix_share.GetB(),
                                dim1, dim2, dim3, modulus));
  // Compute share [A]_0*(Y-B).
  ASSIGN_OR_RETURN(auto share_a_multiply_y_minus_b,
                   ModMatrixMul(beaver_matrix_share.GetA(), y_minus_b,
                                dim1, dim2, dim3, modulus));
  // Compute (X-A)*(Y-B). Only Party Zero needs to compute this.
  ASSIGN_OR_RETURN(auto x_minus_a_muliply_y_minus_b,
                   ModMatrixMul(x_minus_a, y_minus_b,
                                dim1, dim2, dim3, modulus));
  // Generate output: [XY]_0 = (X-A)*[B]_0 + [A]_0*(Y-B) + [C]_0 + (X-A)*(Y-B).
  ASSIGN_OR_RETURN(auto share_xy,
                   BatchedModAdd(x_minus_a_muliply_share_b,
                                 share_a_multiply_y_minus_b, modulus));
  ASSIGN_OR_RETURN(share_xy,
                   BatchedModAdd(beaver_matrix_share.GetC(),
                                 share_xy, modulus));
  ASSIGN_OR_RETURN(share_xy,
                   BatchedModAdd(x_minus_a_muliply_y_minus_b,
                                 share_xy, modulus));
  return share_xy;
}

// Functions to generate the output of the matrix multiplication gate for P1.
StatusOr<std::vector<uint64_t>> GenerateMatrixMultiplicationOutputPartyOne(
    MatrixMultState state,
    const BeaverTripleMatrix<uint64_t>& beaver_matrix_share,
    MatrixMultiplicationGateMessage other_party_message, uint64_t modulus) {
  if (state.share_x_minus_a.empty() || state.share_y_minus_b.empty()) {
    return InvalidArgumentError("GenerateMatrixMultiplicationOutput: "
                                "input must not be empty.");
  }

  if (state.share_x_minus_a.size() != beaver_matrix_share.GetA().size() ||
      state.share_y_minus_b.size() != beaver_matrix_share.GetB().size()) {
    return InvalidArgumentError("GenerateMatrixMultiplicationOutput: "
                                "input and beaver share dimension mismatch.");
  }
  // Reconstruct (X-A) and (Y-B).
  std::vector<uint64_t> x_minus_a(state.share_x_minus_a.size());
  std::vector<uint64_t> y_minus_b(state.share_y_minus_b.size());
  for (size_t idx = 0; idx < x_minus_a.size(); idx++) {
    x_minus_a[idx] =
        ModAdd(other_party_message.matrix_x_minus_matrix_a_shares(idx),
               state.share_x_minus_a[idx], modulus);
  }
  for (size_t idx = 0; idx < y_minus_b.size(); idx++) {
    y_minus_b[idx] =
        ModAdd(other_party_message.matrix_y_minus_matrix_b_shares(idx),
               state.share_y_minus_b[idx], modulus);
  }

  auto dimensions = beaver_matrix_share.GetDimensions();
  auto dim1 = std::get<0>(dimensions);
  auto dim2 = std::get<1>(dimensions);
  auto dim3 = std::get<2>(dimensions);
  // Compute share (X-A)*[B]_1.
  ASSIGN_OR_RETURN(auto x_minus_a_muliply_share_b,
                   ModMatrixMul(x_minus_a, beaver_matrix_share.GetB(),
                                dim1, dim2, dim3, modulus));
  // Compute share [A]_1*(Y-B).
  ASSIGN_OR_RETURN(auto share_a_multiply_y_minus_b,
                   ModMatrixMul(beaver_matrix_share.GetA(), y_minus_b,
                                dim1, dim2, dim3, modulus));
  // Generate output: [XY]_1 = (X-A)*[B]_0 + [A]_0*(Y-B) + [C]_0.
  ASSIGN_OR_RETURN(auto share_xy,
                   BatchedModAdd(x_minus_a_muliply_share_b,
                                 share_a_multiply_y_minus_b, modulus));
  ASSIGN_OR_RETURN(share_xy,
                   BatchedModAdd(beaver_matrix_share.GetC(),
                                 share_xy, modulus));
  return share_xy;
}

// Helper functions to perform truncation of shares.
StatusOr<std::vector<uint64_t>> TruncateSharePartyZero(
    const std::vector<uint64_t>& shares,
    uint64_t divisor, uint64_t modulus) {
  if (shares.empty()) {
    return InvalidArgumentError("TruncateSharePartyZero: "
                                "input must not be empty.");
  }
  std::vector<uint64_t> truncated_shares(shares.size());
  for (size_t idx = 0; idx < shares.size(); idx++) {
    truncated_shares[idx] = shares[idx]/divisor;
  }
  return truncated_shares;
}

StatusOr<std::vector<uint64_t>> TruncateSharePartyOne(
    const std::vector<uint64_t>& shares,
    uint64_t divisor, uint64_t modulus) {
  if (shares.empty()) {
    return InvalidArgumentError("TruncateSharePartyOne: "
                                "input must not be empty.");
  }
  std::vector<uint64_t> truncated_shares(shares.size());
  for (size_t idx = 0; idx < shares.size(); idx++) {
    truncated_shares[idx] = ModSub(
        0, (modulus - shares[idx])/divisor, modulus);
  }
  return truncated_shares;
}

// These functions are only for testing purpose.
// The functions generate shares of BeaverTripleVector and BeaverTripleMatrix
// insecurely.
namespace internal {

// Helper function to generate random shares of a Beaver triple vector in the
// form of a pair ((A_1, B_1, C_1), (A_2, B_2, C_2)) from a PRNG such that:
// (A, B, C) = (A_1 + A_2 % modulus, B_1 + B_2 % modulus, C_1 + C_2 % modulus)
// and C[idx] = A[idx]*B[idx] % modulus. * represents regular multiplication.
StatusOr<std::pair<BeaverTripleVector<uint64_t>, BeaverTripleVector<uint64_t>>>
    SampleBeaverVectorShareWithPrng(size_t length, uint64_t modulus) {
  if (length < 1) {
    return InvalidArgumentError(
        "SampleVectorFromPrng: length of vector must be a positive integer.");
  }
  // Sample a PRNG to generate the shares.
  auto seed = BasicRng::GenerateSeed();
  if (!seed.ok()) {
    return InternalError("Random seed fails to be initialized.");
  }
  auto uptr_prng = BasicRng::Create(seed.value());
  if (!uptr_prng.ok()) {
    return InternalError("Prng fails to be initialized.");
  }
  BasicRng* prng = uptr_prng.value().get();

  // First, generate the first random beaver triple vector share.
  // beaver_triple_vector_1 <-- (A_1, B_1, C_1).
  ASSIGN_OR_RETURN(auto random_vector_a,
                   SampleVectorFromPrng(length, modulus, prng));
  ASSIGN_OR_RETURN(auto random_vector_b,
                   SampleVectorFromPrng(length, modulus, prng));
  ASSIGN_OR_RETURN(auto random_vector_c,
                   SampleVectorFromPrng(length, modulus, prng));
  ASSIGN_OR_RETURN(auto beaver_triple_vector_1,
                   BeaverTripleVector<uint64_t>::Create(
                       random_vector_a, random_vector_b, random_vector_c));

  // Now generate (A, B, C) such that C[i] = A[i]*B[i] where *
  // represents regular multiplication mod modulus.
  ASSIGN_OR_RETURN(auto vector_a,
                   SampleVectorFromPrng(length, modulus, prng));
  ASSIGN_OR_RETURN(auto vector_b,
                   SampleVectorFromPrng(length, modulus, prng));
  std::vector<uint64_t> vector_c;
  vector_c.reserve(length);
  for (size_t idx = 0; idx < length; idx++) {
    vector_c.push_back(ModMul(vector_a[idx], vector_b[idx], modulus));
  }

  // Compute (A_2, B_2, C_2) <-- (A-A_1, B-B_1, C-C_1).
  for (size_t idx = 0; idx < length; idx++) {
    vector_a[idx] = ModSub(vector_a[idx], random_vector_a[idx], modulus);
    vector_b[idx] = ModSub(vector_b[idx], random_vector_b[idx], modulus);
    vector_c[idx] = ModSub(vector_c[idx], random_vector_c[idx], modulus);
  }

  // Generate the second random beaver triple vector share.
  // beaver_triple_vector_2 <-- (A_2, B_2, C_2).
  ASSIGN_OR_RETURN(auto beaver_triple_vector_2,
                   BeaverTripleVector<uint64_t>::Create(
                       vector_a, vector_b, vector_c));
  return std::make_pair(beaver_triple_vector_1, beaver_triple_vector_2);
}

// Helper function to generate random shares of a Beaver triple matrix in the
// form of a pair ((A_1, B_1, C_1), (A_2, B_2, C_2)) from a PRNG such that:
// (A, B, C) = (A_1 + A_2 % modulus, B_1 + B_2 % modulus, C_1 + C_2 % modulus)
// and C = A*B % modulus. * represents matrix multiplication.
// (dim1, dim2, dim3) define the size of the matrices A, B, and C.
// #row x #col of A: dim1 x dim2.
// #row x #col of B: dim2 x dim3.
// #row x #col of C: dim1 x dim3.
StatusOr<std::pair<BeaverTripleMatrix<uint64_t>, BeaverTripleMatrix<uint64_t>>>
    SampleBeaverMatrixShareWithPrng(
          size_t dim1, size_t dim2, size_t dim3, uint64_t modulus) {
  if (dim1 < 1 || dim2 < 1 || dim3 < 1) {
    return InvalidArgumentError(
        "SampleVectorFromPrng: length of vector must be a positive integer."
        );
  }
  // Sample a PRNG to generate the shares.
  auto seed = BasicRng::GenerateSeed();
  if (!seed.ok()) {
    return InternalError("Random seed fails to be initialized.");
  }
  auto uptr_prng = BasicRng::Create(seed.value());
  if (!uptr_prng.ok()) {
    return InternalError("Prng fails to be initialized.");
  }
  BasicRng* prng = uptr_prng.value().get();

  // First, generate the first random beaver triple vector matrix.
  // beaver_triple_matrix_1 <-- (A_1, B_1, C_1).
  // The matrices are represented by 1D vectors.
  // The dimension of each matrix is:
  // A_1: dim1 x dim2.
  // B_1: dim2 x dim3.
  // C_1: dim1 x dim3.
  ASSIGN_OR_RETURN(auto random_vector_a,
                   SampleVectorFromPrng(dim1*dim2, modulus, prng));
  ASSIGN_OR_RETURN(auto random_vector_b,
                   SampleVectorFromPrng(dim2*dim3, modulus, prng));
  ASSIGN_OR_RETURN(auto random_vector_c,
                   SampleVectorFromPrng(dim1*dim3, modulus, prng));
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_1,
                   BeaverTripleMatrix<uint64_t>::Create(
                       random_vector_a, random_vector_b, random_vector_c,
                       dim1, dim2, dim3));

  // Generate a random BeaverTripleMatrix (A, B, C) such that
  // C = A*B, where * represents matrix multiplication mod modulus.
  ASSIGN_OR_RETURN(auto vector_a,
                   SampleVectorFromPrng(dim1*dim2, modulus, prng));
  ASSIGN_OR_RETURN(auto vector_b,
                   SampleVectorFromPrng(dim2*dim3, modulus, prng));

  // Compute C = A*B mod modulus.
  // The dimension of each matrix is:
  // A: dim1 x dim2.
  // B: dim2 x dim3.
  // C: dim1 x dim3.
  std::vector<uint64_t> vector_c;
  vector_c.resize(dim1*dim3);
  for (size_t rdx = 0; rdx < dim1; rdx++) {
    for (size_t cdx = 0; cdx < dim3; cdx++) {
      // Compute C[rdx, cdx] where rdx and cdx are the row number and column
      // number of C respectively.
      // C[rdx, cdx] = (sum A[rdx, kdx]*B[kdx, cdx]) mod modulus, where
      // kdx = 0 --> dim2 - 1.
      // A[rdx, kdx] is stored at vector_a[rdx*dim2 + kdx].
      // B[kdx, cdx] is stored at vector_b[kdx*dim3 + cdx].
      uint64_t sum = 0;
      for (size_t kdx = 0; kdx < dim2; kdx++) {
        sum = ModAdd(sum,
                     ModMul(vector_a[rdx*dim2 + kdx],
                            vector_b[kdx*dim3 + cdx], modulus), modulus);
      }

      // C[rdx, cdx] is stored at vector_c[rdx*dim3 + cdx].
      vector_c[rdx*dim3 + cdx] = sum;
    }
  }

  // Compute A_2 <-- A - A_1.
  for (size_t idx = 0; idx < dim1*dim2; idx++) {
    vector_a[idx] = ModSub(vector_a[idx], random_vector_a[idx], modulus);
  }
  // Compute B_2 <-- B - B_1.
  for (size_t idx = 0; idx < dim2*dim3; idx++) {
    vector_b[idx] = ModSub(vector_b[idx], random_vector_b[idx], modulus);
  }
  // Compute C_2 <-- C - C_1.
  for (size_t idx = 0; idx < dim1*dim3; idx++) {
    vector_c[idx] = ModSub(vector_c[idx], random_vector_c[idx], modulus);
  }

  // Generate the second random beaver triple matrix share.
  // beaver_triple_matrix_2 <-- (A_2, B_2, C_2).
  ASSIGN_OR_RETURN(auto beaver_triple_matrix_2,
                   BeaverTripleMatrix<uint64_t>::Create(
                       vector_a, vector_b, vector_c, dim1, dim2, dim3));

  return std::make_pair(beaver_triple_matrix_1, beaver_triple_matrix_2);
}

// Helper function to generate tuples of random values (a_0, a_1, b_0, b_1)
// using a PRNG such that a_0*a_1 + b_0*b_1 = 1 mod modulus.
// The number of tuples must be at least 1, else the function returns
// INVALID_ARGUMENT.
// The tuples are used to compute secure exponentiation.
StatusOr<std::pair<MultToAddShare, MultToAddShare>>
    SampleMultToAddSharesWithPrng(size_t length, uint64_t modulus) {
  if (length < 1) {
    return InvalidArgumentError(
        "SampleMultToAddSharesWithPrng: length must be a positive integer."
        );
  }

  // Sample a PRNG to generate the random values.
  auto seed = BasicRng::GenerateSeed();
  if (!seed.ok()) {
    return InternalError("Random seed fails to be initialized.");
  }
  auto uptr_prng = BasicRng::Create(seed.value());
  if (!uptr_prng.ok()) {
    return InternalError("Prng fails to be initialized.");
  }
  BasicRng* prng = uptr_prng.value().get();

  // Sample vectors of random values, each has size length.
  ASSIGN_OR_RETURN(auto vector_a_0,
                   SampleVectorFromPrng(length, modulus, prng));
  ASSIGN_OR_RETURN(auto vector_b_0,
                   SampleVectorFromPrng(length, modulus, prng));
  ASSIGN_OR_RETURN(auto vector_a_1,
                   SampleVectorFromPrng(length, modulus, prng));
  ASSIGN_OR_RETURN(auto vector_b_1,
                   SampleVectorFromPrng(length, modulus, prng));

  // Compute vector r where r[i] = a_0[i]*a_1[i] + b_0[i]*b_1[i] mod modulus.
  std::vector<uint64_t> vector_r(length);
  for (size_t idx = 0; idx < length; idx++) {
    vector_r[idx] = ModAdd(ModMul(vector_a_0[idx], vector_a_1[idx], modulus),
                       ModMul(vector_b_0[idx], vector_b_1[idx], modulus),
                       modulus);
    // When modulus is a large prime, r[i] has an inverse with high
    // probability. However, it's desirable to make sure that all r[i] has
    // inverse. When r[i] = 0, we sample a_0[i] again, and test if the new
    // r[i] is not 0. This repeats until r[i] != 0.
    // NOTE: if modulus is not prime, the loop may go forever.
    while (vector_r[idx] == 0) {
      auto random_value = prng->Rand64();
      if (!random_value.ok()) {
        return InternalError("Fail to obtain random value.");
      }
      vector_a_0[idx] = random_value.value();
      vector_r[idx] = ModAdd(
          ModMul(vector_a_0[idx], vector_a_1[idx], modulus),
          ModMul(vector_b_0[idx], vector_b_1[idx], modulus), modulus);
    }
  }

  // Compute batched mod inverse of vector_r.
  // vector_i_inv[i] = vector_r[i]^{-1} mod moduls.
  ASSIGN_OR_RETURN(auto vector_r_inv, BatchedModInv(vector_r, modulus));

  // For now, a_0[i]*a_1[i] + b_0[i]*b_1[i] = r[i] mod modulus.
  // We want to have a_0[i]*a_1[i] + b_0[i]*b_1[i] = 1 mod modulus instead.
  // To achieve that, we simply replace a_0[i] and b_0[i] with
  // a_0[i]*r[i]^{-1} mod modulus and b_0[i]*r[i]^{-1} mod modulus.
  for (size_t idx = 0; idx < length; idx++) {
    vector_a_0[idx] = ModMul(vector_a_0[idx], vector_r_inv[idx], modulus);
    vector_b_0[idx] = ModMul(vector_b_0[idx], vector_r_inv[idx], modulus);
  }

  return std::make_pair(std::make_pair(vector_a_0, vector_b_0),
                        std::make_pair(vector_a_1, vector_b_1));
}

// Helper function to generate share of zero. The output is a pair of random
// vectors A and B such that A[i] + B[i] = 0 mod modulus.
StatusOr<std::pair<std::vector<uint64_t>,
                   std::vector<uint64_t>>> SampleShareOfZero(
    size_t length, uint64_t modulus) {
  if (length < 1) {
    return InvalidArgumentError("SampleShareOfZero: length must positive.");
  }
  // Sample a PRNG to generate the random values.
  auto seed = BasicRng::GenerateSeed();
  if (!seed.ok()) {
    return InternalError("Random seed fails to be initialized.");
  }
  auto uptr_prng = BasicRng::Create(seed.value());
  if (!uptr_prng.ok()) {
    return InternalError("Prng fails to be initialized.");
  }
  BasicRng* prng = uptr_prng.value().get();

  // Sample vectors of random values, each has size length.
  ASSIGN_OR_RETURN(auto vector_a_0,
                   SampleVectorFromPrng(length, modulus, prng));
  std::vector<uint64_t> vector_a_1(vector_a_0.size());
  for (size_t idx = 0; idx < length; idx++) {
    vector_a_1[idx] = ModSub(0, vector_a_0[idx], modulus);
  }
  return std::make_pair(vector_a_0, vector_a_1);
}

}  // namespace internal
}  // namespace private_join_and_compute
