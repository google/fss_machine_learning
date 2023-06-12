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

#ifndef OPEN_SOURCE_SECURE_SIGMOID_RPC_IMPL_H_
#define OPEN_SOURCE_SECURE_SIGMOID_RPC_IMPL_H_

#include "include/grpcpp/grpcpp.h"
#include "include/grpcpp/server_context.h"
#include "include/grpcpp/support/status.h"
#include "applications/secure_sigmoid/secure_sigmoid_rpc.grpc.pb.h"
#include "applications/secure_sigmoid/secure_sigmoid_rpc.pb.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "absl/strings/string_view.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "applications/secure_sigmoid/secure_sigmoid.h"
#include <random>
#include <chrono>

namespace private_join_and_compute {
namespace applications {

// Implements the Gradient Descent RPC-handling Server.
class SecureSigmoidRpcImpl : public SecureSigmoidRpc::Service {
 public:
  SecureSigmoidRpcImpl(  
		std::vector<uint64_t> sigmoid_input_shares,
		SigmoidPrecomputedValue
			sigmoid_precomputed_value,
		applications::SecureSigmoidParameters sigmoid_params);

  // Executes a round of the protocol.
  ::grpc::Status Handle(::grpc::ServerContext* context,
                        const SecureSigmoidClientMessage* request,
                        SecureSigmoidServerMessage* response) override;
	
	bool shut_down() {
	    return finished_;
	  }

 private:
	 // Internal version of Handle, that returns a non-grpc Status.
	 Status HandleInternal(::grpc::ServerContext* context,
	 	const SecureSigmoidClientMessage* request,
		SecureSigmoidServerMessage* response);
	 
	 std::vector<uint64_t> sigmoid_input_shares_;
 	 SigmoidPrecomputedValue sigmoid_precomputed_value_;
	 const SecureSigmoidParameters sigmoid_params_;
	 
	 std::unique_ptr<SecureSigmoid> secure_sigmoid_;
	 
	 bool finished_ = false;
	 
	 SecureSigmoidClientMessage previous_client_message_;

	 // Intermediates
	 RoundOneSigmoidState state_round_1_;
	 RoundTwoSigmoidState state_round_2_;
	 RoundThreeSigmoidState state_round_3_;
	 RoundFourSigmoidState state_round_4_;
	 
	 size_t total_client_message_size_ = 0;
	 size_t total_server_message_size_ = 0;
};

}  // namespace applications
}  // namespace private_join_and_compute

#endif  // OPEN_SOURCE_SECURE_SIGMOID_RPC_IMPL_H_
