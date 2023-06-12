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

#ifndef OPEN_SOURCE_CONST_ROUND_SECURE_COMPARISON_RPC_IMPL_H_
#define OPEN_SOURCE_CONST_ROUND_SECURE_COMPARISON_RPC_IMPL_H_

#include "include/grpcpp/grpcpp.h"
#include "include/grpcpp/server_context.h"
#include "include/grpcpp/support/status.h"
#include "applications/const_round_secure_comparison/const_round_secure_comparison_rpc.grpc.pb.h"
#include "applications/const_round_secure_comparison/const_round_secure_comparison_rpc.pb.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "absl/strings/string_view.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "applications/secure_comparison/secure_comparison.h"
#include "applications/const_round_secure_comparison/const_round_secure_comparison.h"
#include <random>
#include <chrono>

namespace private_join_and_compute {
namespace applications {

// Implements the secure_comparison RPC-handling Server.
class ConstRoundSecureComparisonRpcImpl : public ConstRoundSecureComparisonRpc::Service {
 public:
	 // Expects inputs to be cleartext (not shares).
  ConstRoundSecureComparisonRpcImpl(  
		std::vector<uint64_t> first_bit_share,
	std::vector<uint64_t> comparison_input,
		SecureComparisonPrecomputedValue precomp,
		std::unique_ptr<ConstRoundSecureComparison> secure_comp
		);

  // Executes a round of the protocol.
  ::grpc::Status Handle(::grpc::ServerContext* context,
                        const ConstRoundSecureComparisonClientMessage* request,
                        ConstRoundSecureComparisonServerMessage* response) override;
	
	bool shut_down() {
	    return finished_;
	  }

 private:
	 // Internal version of Handle, that returns a non-grpc Status.
	 Status HandleInternal(::grpc::ServerContext* context,
	 	const ConstRoundSecureComparisonClientMessage* request,
		ConstRoundSecureComparisonServerMessage* response);
	 
		std::vector<uint64_t> first_bit_share_;
	std::vector<uint64_t> comparison_input_;
	SecureComparisonPrecomputedValue precomp_;
	std::unique_ptr<ConstRoundSecureComparison> secure_comp_;
	
	RoundOneSecureComparisonState round_1_state_;
	RoundTwoSecureComparisonState round_2_state_;
	RoundThreeSecureComparisonState round_3_state_;
	 
	volatile bool finished_ = false;
	
 size_t total_client_message_size_ = 0;
 size_t total_server_message_size_ = 0;
	
};

}  // namespace applications
}  // namespace private_join_and_compute

#endif  // OPEN_SOURCE_CONST_ROUND_SECURE_COMPARISON_RPC_IMPL_H_