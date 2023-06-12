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

#ifndef OPEN_SOURCE_GRADIENT_DESCENT_RPC_IMPL_H_
#define OPEN_SOURCE_GRADIENT_DESCENT_RPC_IMPL_H_

#include "include/grpcpp/grpcpp.h"
#include "include/grpcpp/server_context.h"
#include "include/grpcpp/support/status.h"
#include "applications/logistic_regression/gradient_descent_rpc.grpc.pb.h"
#include "applications/logistic_regression/gradient_descent_rpc.pb.h"
#include "applications/logistic_regression/gradient_descent.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "absl/strings/string_view.h"
#include "poisson_regression/beaver_triple_utils.h"
#include "applications/logistic_regression/gradient_descent.h"
#include "applications/secure_sigmoid/secure_sigmoid.h"
#include "applications/secure_sigmoid_new_mic/secure_sigmoid_new_mic.h"
#include <random>
#include <chrono>

namespace private_join_and_compute {
namespace logistic_regression {

// Implements the Gradient Descent RPC-handling Server.
class GradientDescentRpcImpl : public GradientDescentRpc::Service {
 public:
	// For 4-round sigmoid protocol. LogRegShareProvider must be of the
	// matching type.
  GradientDescentRpcImpl(  
		std::vector<uint64_t> share_x,
		std::vector<uint64_t> share_y,
  	std::vector<uint64_t> share_theta,
  	std::unique_ptr<LogRegShareProvider> share_provider,
  	FixedPointElementFactory::Params fpe_params,
  	GradientDescentParams param, 
		applications::SecureSigmoidParameters sigmoid_params);
	
	// For 6-round Sigmoid with New MIC gate. LogRegShareProvider must be of the
	// matching type.
	GradientDescentRpcImpl(  
			std::vector<uint64_t> share_x,
			std::vector<uint64_t> share_y,
	  	std::vector<uint64_t> share_theta,
	  	std::unique_ptr<LogRegShareProvider> share_provider,
	  	FixedPointElementFactory::Params fpe_params,
	  	GradientDescentParams param, 
			applications::SecureSigmoidNewMicParameters sigmoid_params_new_mic);

  // Executes a round of the protocol.
  ::grpc::Status Handle(::grpc::ServerContext* context,
                        const GradientDescentClientMessage* request,
                        GradientDescentServerMessage* response) override;
	
	bool shut_down() {
            if (current_iteration_ < 0) std::cerr << "current_iteration is smaller than 0" << std::endl;
            assert (current_iteration_ >= 0);
	    return num_iterations_ == static_cast<size_t>(current_iteration_);
	  }

 private:
	 // Internal version of Handle, that returns a non-grpc Status.
	 Status HandleInternal(::grpc::ServerContext* context,
	 	const GradientDescentClientMessage* request,
		GradientDescentServerMessage* response);
	 
	 const FixedPointElementFactory::Params fpe_params_;
	 const GradientDescentParams gd_params_;

	 const size_t num_iterations_;
	 
	 std::unique_ptr<GradientDescentPartyOne> gradient_descent_party_one_;
	 
	 StateMaskedX state_masked_x_;
	 StateMaskedXTranspose state_masked_x_transpose_;
	 MaskedXMessage server_masked_x_message_;
   MaskedXTransposeMessage server_masked_x_transpose_message_;
	 MaskedXMessage client_masked_x_message_;
   MaskedXTransposeMessage client_masked_x_transpose_message_;
	 
	 // Intermediates
	 // Cached message from the client for the previous round.
	 StateRound1 state_round_1_;
	 SigmoidInput sigmoid_input_;
	 SigmoidOutput sigmoid_output_;
	 
	 // State, objects and messages for the 4-round Sigmoid implementation
	 applications::SecureSigmoidParameters sigmoid_params_;
	 std::unique_ptr<applications::SecureSigmoid> secure_sigmoid_;
	 applications::SigmoidPrecomputedValue sigmoid_precomputed_value_;
	 applications::RoundOneSigmoidState sigmoid_round_1_state_;
	 applications::RoundTwoSigmoidState sigmoid_round_2_state_;
	 applications::RoundThreeSigmoidState sigmoid_round_3_state_;
	 applications::RoundFourSigmoidState sigmoid_round_4_state_;
	 
	 // State, objects and messages for the 6-round Sigmoid implementation with
	 // new MIC gate.
	 applications::SecureSigmoidNewMicParameters sigmoid_params_new_mic_;
	 std::unique_ptr<applications::SecureSigmoidNewMic> secure_sigmoid_new_mic_;
	 applications::SigmoidPrecomputedValueNewMic*
		  sigmoid_precomputed_value_new_mic_;
	 applications::RoundOneSigmoidNewMicState sigmoid_new_mic_round_1_state_;
	 applications::RoundTwoSigmoidNewMicState sigmoid_new_mic_round_2_state_;
	 applications::RoundThreeSigmoidNewMicState sigmoid_new_mic_round_3_state_;
	 applications::RoundThreePointFiveSigmoidNewMicState
		  sigmoid_new_mic_round_3_point_5_state_;
	 applications::RoundFourSigmoidNewMicState sigmoid_new_mic_round_4_state_;
	 applications::RoundFiveSigmoidNewMicState sigmoid_new_mic_round_5_state_;
	 
	 StateRound3 state_round_3_;
	
	 int current_iteration_ = 0;
	 
	 size_t total_client_message_size_ = 0;
	 size_t total_server_message_size_ = 0;
};

}  // namespace logistic_regression
}  // namespace private_join_and_compute

#endif  // OPEN_SOURCE_GRADIENT_DESCENT_RPC_IMPL_H_
