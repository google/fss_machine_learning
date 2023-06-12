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

#ifndef OPEN_SOURCE_SECURE_COMPARISON_RPC_IMPL_H_
#define OPEN_SOURCE_SECURE_COMPARISON_RPC_IMPL_H_

#include "include/grpcpp/grpcpp.h"
#include "include/grpcpp/server_context.h"
#include "include/grpcpp/support/status.h"
#include "applications/secure_comparison/secure_comparison_rpc.grpc.pb.h"
#include "applications/secure_comparison/secure_comparison_rpc.pb.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "absl/strings/string_view.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "applications/secure_comparison/secure_comparison.h"
#include <random>
#include <chrono>

namespace private_join_and_compute {
namespace secure_comparison {

// Implements the secure_comparison RPC-handling Server.
class SecureComparisonRpcImpl : public SecureComparisonRpc::Service {
 public:
	 // Expects inputs to be cleartext (not shares).
  SecureComparisonRpcImpl(  
		std::vector<uint64_t> inputs,
		size_t block_length,
		size_t num_splits,
		ComparisonEqualityGates comparison_equality_gates,
		ComparisonPreprocessedValues comparison_preprocessed_values
		);

  // Executes a round of the protocol.
  ::grpc::Status Handle(::grpc::ServerContext* context,
                        const SecureComparisonClientMessage* request,
                        SecureComparisonServerMessage* response) override;
	
	bool shut_down() {
	    return finished_;
	  }

 private:
	 // Internal version of Handle, that returns a non-grpc Status.
	 Status HandleInternal(::grpc::ServerContext* context,
	 	const SecureComparisonClientMessage* request,
		SecureComparisonServerMessage* response);
	 
	std::vector<uint64_t> inputs_;
	const size_t block_length_;
	const size_t num_splits_;
	ComparisonEqualityGates gates_;
	ComparisonPreprocessedValues preprocessed_values_;
  size_t num_mult_rounds_;
	 
	// Intermediates
  SecureComparisonClientMessage previous_client_message_;
	size_t current_mult_round_ = 0;
	size_t num_gates_in_current_mult_round_;
	

	// Intermediates
	ComparisonStateRoundOne state_round_one_;
	ComparisonShortComparisonEquality combination_input_;
	BatchedMultState mult_state_;
	 
	bool finished_ = false;
	
 size_t total_client_message_size_ = 0;
 size_t total_server_message_size_ = 0;
	
};

}  // namespace secure_comparison
}  // namespace private_join_and_compute

#endif  // OPEN_SOURCE_SECURE_COMPARISON_RPC_IMPL_H_