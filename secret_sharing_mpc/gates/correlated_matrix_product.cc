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

#include "secret_sharing_mpc/gates/correlated_matrix_product.h"

#include "poisson_regression/ring_arithmetic_utils.h"

namespace private_join_and_compute {

StatusOr<std::tuple<std::vector<uint64_t>, std::vector<uint64_t>,
                    std::vector<uint64_t>>>
SampleBeaverTripleMatrixA(size_t dim1, size_t dim2, uint64_t modulus) {
  if (dim1 < 1 || dim2 < 1) {
    return InvalidArgumentError(
        "SampleBeaverTripleMatrixA: length of vector must be a positive "
        "integer.");
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
  // A: dim1 x dim2.
  // The matrices are represented by 1D flattened vectors.
  // This is the matrix that will mask the multiplicand X
  // across several products with different Y's
  ASSIGN_OR_RETURN(std::vector<uint64_t> random_vector_a,
                   SampleVectorFromPrng(dim1 * dim2, modulus, prng));

  // Now generate a share for P_0: i.e. generate a random A_0
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_a_p0,
                   SampleVectorFromPrng(dim1 * dim2, modulus, prng));

  // Compute P_1's share A_1 <-- A - A_0.
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_a_p1,
                   BatchedModSub(random_vector_a, share_a_p0, modulus));

  return std::make_tuple(share_a_p0, share_a_p1, random_vector_a);
}

StatusOr<std::pair<
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>>,   // P_0's [B], [C]
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>>  // P_1's [B], [C]
SampleBeaverTripleMatrixBandC(
    std::tuple<std::vector<uint64_t>, std::vector<uint64_t>,
               std::vector<uint64_t>>& vector_a,
    size_t dim1, size_t dim2, size_t dim3, uint64_t modulus) {
  if (dim1 < 1 || dim2 < 1 || dim3 < 1) {
    return InvalidArgumentError(
        "SampleBeaverTripleMatrixBandC: length of vector must be a positive "
        "integer.");
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

  // Sample random matrix B
  // B: dim2 x dim3
  ASSIGN_OR_RETURN(auto random_vector_b,
                   SampleVectorFromPrng(dim2 * dim3, modulus, prng));

  // Compute C = A*B mod modulus.
  // C: dim1 x dim3.
  ASSIGN_OR_RETURN(std::vector<uint64_t> vector_c,
                   ModMatrixMul(std::get<2>(vector_a), random_vector_b, dim1,
                                dim2, dim3, modulus));

  // Generate shares randomly for P_0: share_b_p0, share_c_p0
  ASSIGN_OR_RETURN(auto share_b_p0,
                   SampleVectorFromPrng(dim2 * dim3, modulus, prng));
  ASSIGN_OR_RETURN(auto share_c_p0,
                   SampleVectorFromPrng(dim1 * dim3, modulus, prng));

  // Generate Beaver triple shares of B and C from P_0's shares (B_0, C_0).
  auto beaver_triple_b_c_0 = std::make_pair(share_b_p0, share_c_p0);

  // Compute shares for P_1
  // Compute B_1 <-- B - B_0.
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_b_p1,
                   BatchedModSub(random_vector_b, share_b_p0, modulus));
  // Compute C_1 <-- C - C_0.
  ASSIGN_OR_RETURN(std::vector<uint64_t> share_c_p1,
                   BatchedModSub(vector_c, share_c_p0, modulus));

  // Generate Beaver triple shares of B and C from P_1's shares (B_1, C_1).
  auto beaver_triple_b_c_1 = std::make_pair(share_b_p1, share_c_p1);

  return std::make_pair(beaver_triple_b_c_0, beaver_triple_b_c_1);
}

StatusOr<std::pair<std::vector<uint64_t>, MatrixXminusAProductMessage>>
GenerateMatrixXminusAProductMessage(
    const std::vector<uint64_t>& share_x,
    const std::vector<uint64_t>& beaver_matrix_share_a, uint64_t modulus) {
  if (share_x.empty()) {
    return InvalidArgumentError(
        "GenerateMatrixXminusAProductMessage: "
        "input must not be empty.");
  }

  if (share_x.size() != beaver_matrix_share_a.size()) {
    return InvalidArgumentError(
        "GenerateMatrixXminusAProductMessage: "
        "input and beaver shares have different size.");
  }
  ASSIGN_OR_RETURN(auto x_minus_a,
                   BatchedModSub(share_x, beaver_matrix_share_a, modulus));

  MatrixXminusAProductMessage matrix_prod_message;
  for (size_t idx = 0; idx < x_minus_a.size(); idx++) {
    matrix_prod_message.add_matrix_x_minus_matrix_a_shares(x_minus_a[idx]);
  }

  return std::make_pair(std::move(x_minus_a), std::move(matrix_prod_message));
}

StatusOr<std::pair<MatrixMultState, MatrixYminusBProductMessage>>
GenerateMatrixYminusBProductMessage(
    const std::vector<uint64_t>& share_y,
    const std::vector<uint64_t>& share_x_minus_a,
    const std::vector<uint64_t>& beaver_matrix_share_b, uint64_t modulus) {
  if (share_y.empty() || share_x_minus_a.empty()) {
    return InvalidArgumentError(
        "GenerateMatrixYminusBProductMessage: "
        "input must not be empty.");
  }

  if (share_y.size() != beaver_matrix_share_b.size()) {
    return InvalidArgumentError(
        "GenerateMatrixYminusBProductMessage: "
        "input and beaver shares have different size.");
  }
  ASSIGN_OR_RETURN(auto share_y_minus_b,
                   BatchedModSub(share_y, beaver_matrix_share_b, modulus));

  MatrixYminusBProductMessage matrix_prod_message;
  for (size_t idx = 0; idx < share_y_minus_b.size(); idx++) {
    matrix_prod_message.add_matrix_y_minus_matrix_b_shares(
        share_y_minus_b[idx]);
  }
  MatrixMultState state = {.share_x_minus_a = std::move(share_x_minus_a),
                           .share_y_minus_b = std::move(share_y_minus_b)};
  return std::make_pair(std::move(state), std::move(matrix_prod_message));
}

StatusOr<std::vector<uint64_t>> CorrelatedMatrixProductPartyZero(
    MatrixMultState state, const std::vector<uint64_t>& beaver_matrix_share_a,
    const std::vector<uint64_t>& beaver_matrix_share_b,
    const std::vector<uint64_t>& beaver_matrix_share_c,
    MatrixXminusAProductMessage other_party_x_minus_a_message,
    MatrixYminusBProductMessage other_party_y_minus_b_message, size_t dim1,
    size_t dim2, size_t dim3, uint8_t num_fractional_bits, uint64_t modulus) {
  // Party 0 computes its share of the output from its input, Beaver triple
  // matrix, and other party's messages.
  // The product includes an additional term 2^num_fractional_bits, which needs
  // to be divided. Thus, the result needs to be truncated which introduces
  // an error of at most 2^-lf (lf = num_fractional_bits).
  // The truncation happens only once at the end for each element of the matrix.

  if (state.share_x_minus_a.empty() || state.share_y_minus_b.empty()) {
    return InvalidArgumentError(
        "CorrelatedMatrixProductPartyZero: "
        "input must not be empty.");
  }
  if (state.share_x_minus_a.size() != beaver_matrix_share_a.size() ||
      state.share_y_minus_b.size() != beaver_matrix_share_b.size() ||
      beaver_matrix_share_a.size() != (dim1 * dim2) ||
      beaver_matrix_share_b.size() != (dim2 * dim3) ||
      beaver_matrix_share_c.size() != (dim1 * dim3)) {
    return InvalidArgumentError(
        "CorrelatedMatrixProductPartyZero: "
        "input and beaver share dimension mismatch.");
  }
  // Reconstruct (X-A) and (Y-B).
  size_t length_x_minus_a = state.share_x_minus_a.size();
  size_t length_y_minus_b = state.share_y_minus_b.size();
  std::vector<uint64_t> x_minus_a(length_x_minus_a);
  std::vector<uint64_t> y_minus_b(length_y_minus_b);
  for (size_t idx = 0; idx < length_x_minus_a; idx++) {
    x_minus_a[idx] = ModAdd(
        other_party_x_minus_a_message.matrix_x_minus_matrix_a_shares(idx),
        state.share_x_minus_a[idx], modulus);
  }
  for (size_t idx = 0; idx < length_y_minus_b; idx++) {
    y_minus_b[idx] = ModAdd(
        other_party_y_minus_b_message.matrix_y_minus_matrix_b_shares(idx),
        state.share_y_minus_b[idx], modulus);
  }

  // Compute share (X-A)*[B]_0.
  ASSIGN_OR_RETURN(auto x_minus_a_muliply_share_b,
                   ModMatrixMul(x_minus_a, beaver_matrix_share_b, dim1, dim2,
                                dim3, modulus));
  // Compute share [A]_0*(Y-B).
  ASSIGN_OR_RETURN(auto share_a_multiply_y_minus_b,
                   ModMatrixMul(beaver_matrix_share_a, y_minus_b, dim1, dim2,
                                dim3, modulus));
  // Compute (X-A)*(Y-B). Only Party Zero needs to compute this.
  ASSIGN_OR_RETURN(
      auto x_minus_a_muliply_y_minus_b,
      ModMatrixMul(x_minus_a, y_minus_b, dim1, dim2, dim3, modulus));
  // Generate output: [XY]_0 = (X-A)*[B]_0 + [A]_0*(Y-B) + [C]_0 + (X-A)*(Y-B).
  ASSIGN_OR_RETURN(auto share_xy,
                   BatchedModAdd(x_minus_a_muliply_share_b,
                                 share_a_multiply_y_minus_b, modulus));
  ASSIGN_OR_RETURN(share_xy,
                   BatchedModAdd(beaver_matrix_share_c, share_xy, modulus));
  ASSIGN_OR_RETURN(
      share_xy, BatchedModAdd(x_minus_a_muliply_y_minus_b, share_xy, modulus));
  ASSIGN_OR_RETURN(
      share_xy,
      TruncateSharePartyZero(share_xy, (1ULL << num_fractional_bits), modulus));

  return share_xy;
}

StatusOr<std::vector<uint64_t>> CorrelatedMatrixProductPartyOne(
    MatrixMultState state, const std::vector<uint64_t>& beaver_matrix_share_a,
    const std::vector<uint64_t>& beaver_matrix_share_b,
    const std::vector<uint64_t>& beaver_matrix_share_c,
    MatrixXminusAProductMessage other_party_x_minus_a_message,
    MatrixYminusBProductMessage other_party_y_minus_b_message, size_t dim1,
    size_t dim2, size_t dim3, uint8_t num_fractional_bits, uint64_t modulus) {
  // Party 1 computes its share of the output from its input, Beaver triple
  // matrix, and other party's messages.
  // The product includes an additional term 2^num_fractional_bits, which needs
  // to be divided. Thus, the result needs to be truncated which introduces
  // an error of at most 2^-lf (lf = num_fractional_bits).
  // The truncation happens only once at the end for each element of the matrix.
  if (state.share_x_minus_a.empty() || state.share_y_minus_b.empty()) {
    return InvalidArgumentError(
        "CorrelatedMatrixProductPartyOne: "
        "input must not be empty.");
  }

  if (state.share_x_minus_a.size() != beaver_matrix_share_a.size() ||
      state.share_y_minus_b.size() != beaver_matrix_share_b.size() ||
      beaver_matrix_share_a.size() != (dim1 * dim2) ||
      beaver_matrix_share_b.size() != (dim2 * dim3) ||
      beaver_matrix_share_c.size() != (dim1 * dim3)) {
    return InvalidArgumentError(
        "CorrelatedMatrixProductPartyOne: "
        "input and beaver share dimension mismatch.");
  }
  // Reconstruct (X-A) and (Y-B).
  size_t length_x_minus_a = state.share_x_minus_a.size();
  size_t length_y_minus_b = state.share_y_minus_b.size();
  std::vector<uint64_t> x_minus_a(length_x_minus_a);
  std::vector<uint64_t> y_minus_b(length_y_minus_b);
  for (size_t idx = 0; idx < length_x_minus_a; idx++) {
    x_minus_a[idx] = ModAdd(
        other_party_x_minus_a_message.matrix_x_minus_matrix_a_shares(idx),
        state.share_x_minus_a[idx], modulus);
  }
  for (size_t idx = 0; idx < length_y_minus_b; idx++) {
    y_minus_b[idx] = ModAdd(
        other_party_y_minus_b_message.matrix_y_minus_matrix_b_shares(idx),
        state.share_y_minus_b[idx], modulus);
  }

  // Compute share (X-A)*[B]_1.
  ASSIGN_OR_RETURN(auto x_minus_a_muliply_share_b,
                   ModMatrixMul(x_minus_a, beaver_matrix_share_b, dim1, dim2,
                                dim3, modulus));
  // Compute share [A]_1*(Y-B).
  ASSIGN_OR_RETURN(auto share_a_multiply_y_minus_b,
                   ModMatrixMul(beaver_matrix_share_a, y_minus_b, dim1, dim2,
                                dim3, modulus));
  // Generate output: [XY]_1 = (X-A)*[B]_0 + [A]_0*(Y-B) + [C]_0.
  ASSIGN_OR_RETURN(auto share_xy,
                   BatchedModAdd(x_minus_a_muliply_share_b,
                                 share_a_multiply_y_minus_b, modulus));
  ASSIGN_OR_RETURN(share_xy,
                   BatchedModAdd(beaver_matrix_share_c, share_xy, modulus));
  ASSIGN_OR_RETURN(
      share_xy,
      TruncateSharePartyOne(share_xy, (1ULL << num_fractional_bits), modulus));
  return share_xy;
}

}  // namespace private_join_and_compute
