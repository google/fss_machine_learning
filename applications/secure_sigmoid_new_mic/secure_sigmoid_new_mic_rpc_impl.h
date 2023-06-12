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

#ifndef OPEN_SOURCE_SECURE_SIGMOID_NEW_MIC_RPC_IMPL_H_
#define OPEN_SOURCE_SECURE_SIGMOID_NEW_MIC_RPC_IMPL_H_

#include "include/grpcpp/grpcpp.h"
#include "include/grpcpp/server_context.h"
#include "include/grpcpp/support/status.h"
#include "applications/secure_sigmoid_new_mic/secure_sigmoid_new_mic_rpc.grpc.pb.h"
#include "applications/secure_sigmoid_new_mic/secure_sigmoid_new_mic_rpc.pb.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "absl/strings/string_view.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "applications/secure_sigmoid_new_mic/secure_sigmoid_new_mic.h"
#include <random>
#include <chrono>

namespace private_join_and_compute {
namespace applications {

// Implements the Gradient Descent RPC-handling Server.
class SecureSigmoidNewMicRpcImpl : public SecureSigmoidNewMicRpc::Service {
 public:
  SecureSigmoidNewMicRpcImpl(  
		std::vector<uint64_t> sigmoid_input_shares,
		SigmoidPrecomputedValueNewMic
			sigmoid_precomputed_value,
		applications::SecureSigmoidNewMicParameters sigmoid_params);

  // Executes a round of the protocol.
  ::grpc::Status Handle(::grpc::ServerContext* context,
                        const SecureSigmoidNewMicClientMessage* request,
                        SecureSigmoidNewMicServerMessage* response) override;
	
	bool shut_down() {
	    return finished_;
	  }

 private:
	 // Internal version of Handle, that returns a non-grpc Status.
	 Status HandleInternal(::grpc::ServerContext* context,
	 	const SecureSigmoidNewMicClientMessage* request,
		SecureSigmoidNewMicServerMessage* response);
	 
	 std::vector<uint64_t> sigmoid_input_shares_;
 	 SigmoidPrecomputedValueNewMic sigmoid_precomputed_value_;
	 const SecureSigmoidNewMicParameters sigmoid_params_;
	 
	 std::unique_ptr<SecureSigmoidNewMic> secure_sigmoid_;
	 
	 bool finished_ = false;

	 // Intermediates
	 RoundOneSigmoidNewMicState state_round_1_;
	 RoundTwoSigmoidNewMicState state_round_2_;
	 RoundThreeSigmoidNewMicState state_round_3_;
	 RoundThreePointFiveSigmoidNewMicState state_round_3_point_5_;
	 RoundFourSigmoidNewMicState state_round_4_;
	 RoundFiveSigmoidNewMicState state_round_5_;
	 
	 size_t total_client_message_size_ = 0;
	 size_t total_server_message_size_ = 0;
};

}  // namespace applications
}  // namespace private_join_and_compute

#endif  // OPEN_SOURCE_SECURE_SIGMOID_NEW_MIC_RPC_IMPL_H_