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

#include "gradient_descent_rpc_impl.h"

#include "private_join_and_compute/util/status.inc"

namespace private_join_and_compute {
namespace logistic_regression {


namespace {
	using namespace ::private_join_and_compute::applications;
	
	// Translates Status to grpc::Status
	::grpc::Status ConvertStatus(const Status& status) {
	  if (status.ok()) {
	    return ::grpc::Status::OK;
	  }
	  if (IsInvalidArgument(status)) {
	    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
	                          std::string(status.message()));
	  }
	  if (IsInternal(status)) {
	    return ::grpc::Status(::grpc::StatusCode::INTERNAL,
	                          std::string(status.message()));
	  }
	  return ::grpc::Status(::grpc::StatusCode::UNKNOWN,
	                        std::string(status.message()));
	}
}

GradientDescentRpcImpl::GradientDescentRpcImpl(
  std::vector<uint64_t> share_x,
	std::vector<uint64_t> share_y,
  std::vector<uint64_t> share_theta,
  std::unique_ptr<LogRegShareProvider> share_provider,
  FixedPointElementFactory::Params fpe_params,
  GradientDescentParams param, SecureSigmoidParameters sigmoid_params):
	  fpe_params_(std::move(fpe_params)),
	  gd_params_(std::move(param)),
		num_iterations_(gd_params_.num_iterations),
		sigmoid_params_(std::move(sigmoid_params))
		 {
	gradient_descent_party_one_ = std::make_unique<GradientDescentPartyOne>(std::move(GradientDescentPartyOne::Init(
			       std::move(share_x),
			       std::move(share_y),
			       std::move(share_theta),
			       std::move(share_provider),
			       fpe_params_,
			       gd_params_).value()));
	secure_sigmoid_ = SecureSigmoid::Create(gd_params_.num_examples,
		sigmoid_params_).value();
		std::tie(state_masked_x_, server_masked_x_message_) = gradient_descent_party_one_->GenerateCorrelatedProductMessageForX().value();
	  std::tie(state_masked_x_transpose_, server_masked_x_transpose_message_) = gradient_descent_party_one_->GenerateCorrelatedProductMessageForXTranspose().value();
}


GradientDescentRpcImpl::GradientDescentRpcImpl(
  std::vector<uint64_t> share_x,
	std::vector<uint64_t> share_y,
  std::vector<uint64_t> share_theta,
  std::unique_ptr<LogRegShareProvider> share_provider,
  FixedPointElementFactory::Params fpe_params,
  GradientDescentParams param, SecureSigmoidNewMicParameters
		sigmoid_params_new_mic):
	  fpe_params_(std::move(fpe_params)),
	  gd_params_(std::move(param)),
		num_iterations_(gd_params_.num_iterations),
		sigmoid_params_new_mic_(std::move(sigmoid_params_new_mic))
		 {
	gradient_descent_party_one_ = std::make_unique<GradientDescentPartyOne>(std::move(GradientDescentPartyOne::Init(
			       std::move(share_x),
			       std::move(share_y),
			       std::move(share_theta),
			       std::move(share_provider),
			       fpe_params_,
			       gd_params_).value()));
	secure_sigmoid_new_mic_ = SecureSigmoidNewMic::Create(gd_params_.num_examples,
		sigmoid_params_new_mic_).value();
		std::tie(state_masked_x_, server_masked_x_message_) = gradient_descent_party_one_->GenerateCorrelatedProductMessageForX().value();
	  std::tie(state_masked_x_transpose_, server_masked_x_transpose_message_) = gradient_descent_party_one_->GenerateCorrelatedProductMessageForXTranspose().value();
}

::grpc::Status GradientDescentRpcImpl::Handle(::grpc::ServerContext* context,
                      const GradientDescentClientMessage* request,
                      GradientDescentServerMessage* response) {
	return ConvertStatus(HandleInternal(context, request, response));
}

Status GradientDescentRpcImpl::HandleInternal(::grpc::ServerContext* context,
	const GradientDescentClientMessage* request,
	GradientDescentServerMessage* response)	{
	
  // Preliminary Rounds (should only be called once).
		if (request->has_client_masked_x_message()){
			std::cout<< "Server: got client masked_x_message" << std::endl;
	    client_masked_x_message_ = request->client_masked_x_message();
			*response->mutable_server_masked_x_message() = server_masked_x_message_;
			return OkStatus();
		} 
		if (request->has_client_masked_x_transpose_message()){
			std::cout<< "Server: got client masked_x_transpose_message" << std::endl;
	    client_masked_x_transpose_message_ = request->client_masked_x_transpose_message();
			*response->mutable_server_masked_x_transpose_message() = server_masked_x_transpose_message_;
			return OkStatus();
		}
		// Loop Rounds
		// Switch over different client message types
	  if(request->has_start_message()) {
			std::cout<< "Executing server iteration " << current_iteration_ + 1 <<  std::endl;
	  	// execute GD round 1.
		ASSIGN_OR_RETURN(
			std::tie(state_round_1_, *response->mutable_server_gradient_descent_round_1_message()), 			gradient_descent_party_one_->GenerateGradientDescentRoundOneMessage(state_masked_x_));
	  } 
		///////////////////////////////////////////////////////////////////////////
		// 4-round sigmoid rounds
		///////////////////////////////////////////////////////////////////////////
		else if(request->has_client_gradient_descent_round_1_message() && secure_sigmoid_!= nullptr) {
			
			ASSIGN_OR_RETURN(sigmoid_input_,     
			gradient_descent_party_one_->GenerateSigmoidInput(
	                           state_round_1_, client_masked_x_message_,                         		request->client_gradient_descent_round_1_message()));
		// execute GD round 2 (Sigmoid Round 1).
		ASSIGN_OR_RETURN(sigmoid_precomputed_value_,
			gradient_descent_party_one_->share_provider_->GetSigmoidPrecomputedValue()		);
		ASSIGN_OR_RETURN(
			std::tie(sigmoid_round_1_state_, *response->mutable_server_sigmoid_round_1_message()), 			secure_sigmoid_->GenerateSigmoidRoundOneMessage(
	          1, sigmoid_precomputed_value_, sigmoid_input_.sigmoid_input));
	  } else if(request->has_client_sigmoid_round_1_message()) {
		// execute GD round 3 (Sigmoid Round 2).
			ASSIGN_OR_RETURN(
				std::tie(sigmoid_round_2_state_, *response->mutable_server_sigmoid_round_2_message()), 			secure_sigmoid_->GenerateSigmoidRoundTwoMessage(
		          1, sigmoid_precomputed_value_, sigmoid_round_1_state_,
			request->client_sigmoid_round_1_message()));
	  } else if(request->has_client_sigmoid_round_2_message()) {
		// execute GD round 4 (Sigmoid Round 3).
			ASSIGN_OR_RETURN(
				std::tie(sigmoid_round_3_state_, *response->mutable_server_sigmoid_round_3_message()), 			secure_sigmoid_->GenerateSigmoidRoundThreeMessage(
		          1, sigmoid_precomputed_value_, sigmoid_round_2_state_, 
			request->client_sigmoid_round_2_message()));
	} else if(request->has_client_sigmoid_round_3_message()) {
	// execute GD round 5 (Sigmoid Round 4).
		ASSIGN_OR_RETURN(
			std::tie(sigmoid_round_4_state_, *response->mutable_server_sigmoid_round_4_message()), 			secure_sigmoid_->GenerateSigmoidRoundFourMessage(
	          1, sigmoid_precomputed_value_, sigmoid_round_3_state_, 
		request->client_sigmoid_round_3_message()));
	}	else if(request->has_client_sigmoid_round_4_message()) {
		// Compute sigmoid output and the gradient descent final message
		ASSIGN_OR_RETURN(std::vector<uint64_t> raw_sigmoid_output_shares,
		 secure_sigmoid_->GenerateSigmoidResult(1,
	                                            sigmoid_precomputed_value_,
	                                            sigmoid_round_4_state_,
									request->client_sigmoid_round_4_message()));
		sigmoid_output_ = {
	      .sigmoid_output = std::move(raw_sigmoid_output_shares)
	  };
	  ASSIGN_OR_RETURN(
	      std::tie(state_round_3_, *response->mutable_server_gradient_descent_final_message()),
	      gradient_descent_party_one_->GenerateGradientDescentRoundThreeMessage(
	          sigmoid_output_, state_masked_x_transpose_));
		
	} 
		///////////////////////////////////////////////////////////////////////////
		// 6-round Sigmoid with New MIC Gate
		///////////////////////////////////////////////////////////////////////////
	else if(request->has_client_gradient_descent_round_1_message() && secure_sigmoid_new_mic_!= nullptr) {
		
		ASSIGN_OR_RETURN(sigmoid_input_,     
		gradient_descent_party_one_->GenerateSigmoidInput(
                           state_round_1_, client_masked_x_message_,                         		request->client_gradient_descent_round_1_message()));
	// execute Sigmoid Round 1.
  sigmoid_precomputed_value_new_mic_ = 
		&(gradient_descent_party_one_->
			share_provider_->GetSigmoidPrecomputedValueNewMic());
	ASSIGN_OR_RETURN(
		std::tie(sigmoid_new_mic_round_1_state_, *response->mutable_server_sigmoid_new_mic_round_1_message()), 			secure_sigmoid_new_mic_->GenerateSigmoidRoundOneMessage(
          1, *sigmoid_precomputed_value_new_mic_, sigmoid_input_.sigmoid_input));
  } else if(request->has_client_sigmoid_new_mic_round_1_message()) {
	// execute Sigmoid Round 2.
		ASSIGN_OR_RETURN(
			std::tie(sigmoid_new_mic_round_2_state_, *response->mutable_server_sigmoid_new_mic_round_2_message()), 			secure_sigmoid_new_mic_->GenerateSigmoidRoundTwoMessage(
	          1, *sigmoid_precomputed_value_new_mic_, sigmoid_new_mic_round_1_state_, request->client_sigmoid_new_mic_round_1_message()));
  } else if(request->has_client_sigmoid_new_mic_round_2_message()) {
	// execute Sigmoid Round 3.
		ASSIGN_OR_RETURN(
			std::tie(sigmoid_new_mic_round_3_state_, *response->mutable_server_sigmoid_new_mic_round_3_message()), 			secure_sigmoid_new_mic_->GenerateSigmoidRoundThreeMessage(
	          1, *sigmoid_precomputed_value_new_mic_, sigmoid_new_mic_round_2_state_, request->client_sigmoid_new_mic_round_2_message()));
} else if(request->has_client_sigmoid_new_mic_round_3_message()) {
	// execute Sigmoid Round 3.
		ASSIGN_OR_RETURN(
			std::tie(sigmoid_new_mic_round_3_point_5_state_, *response->mutable_server_sigmoid_new_mic_round_3_point_5_message()), 			secure_sigmoid_new_mic_->GenerateSigmoidRoundThreePointFiveMessage(
	          1, *sigmoid_precomputed_value_new_mic_, sigmoid_new_mic_round_3_state_, request->client_sigmoid_new_mic_round_3_message()));
} else if(request->has_client_sigmoid_new_mic_round_3_point_5_message()) {
// execute Sigmoid Round 4.
	ASSIGN_OR_RETURN(
		std::tie(sigmoid_new_mic_round_4_state_, *response->mutable_server_sigmoid_new_mic_round_4_message()), 			secure_sigmoid_new_mic_->GenerateSigmoidRoundFourMessage(
          1, *sigmoid_precomputed_value_new_mic_, sigmoid_new_mic_round_3_point_5_state_, request->client_sigmoid_new_mic_round_3_point_5_message()));
} else if(request->has_client_sigmoid_new_mic_round_4_message()) {
		ASSIGN_OR_RETURN(
			std::tie(sigmoid_new_mic_round_5_state_, *response->mutable_server_sigmoid_new_mic_round_5_message()), 			secure_sigmoid_new_mic_->GenerateSigmoidRoundFiveMessage(
	          1, *sigmoid_precomputed_value_new_mic_, sigmoid_new_mic_round_4_state_, request->client_sigmoid_new_mic_round_4_message()));
	} else if (request->has_client_sigmoid_new_mic_round_5_message()) {
	ASSIGN_OR_RETURN(std::vector<uint64_t> raw_sigmoid_output_shares,
	 secure_sigmoid_new_mic_->GenerateSigmoidResult(1,
                                            *sigmoid_precomputed_value_new_mic_,
                                            sigmoid_new_mic_round_5_state_,
								request->client_sigmoid_new_mic_round_5_message()));
	sigmoid_output_ = {
      .sigmoid_output = std::move(raw_sigmoid_output_shares)
		};
  ASSIGN_OR_RETURN(
      std::tie(state_round_3_, *response->mutable_server_gradient_descent_final_message()),
      gradient_descent_party_one_->GenerateGradientDescentRoundThreeMessage(
          sigmoid_output_, state_masked_x_transpose_));
} else if(request->has_client_gradient_descent_final_message()) {
	// Final message for Gradient Descent.

		// Compute gradient update.
	    Status status = gradient_descent_party_one_->ComputeGradientUpdate(
      state_round_3_, client_masked_x_transpose_message_, 			request->client_gradient_descent_final_message()
      );
	    if (!status.ok()) {
			return status;
	    }
			*response->mutable_end_message() = EndMessage();

		// Clear all state and increment iteration number.
	 	 state_round_1_ = StateRound1();
	 	 sigmoid_precomputed_value_ = SigmoidPrecomputedValue();
	 	 sigmoid_input_ = SigmoidInput();
	 	 sigmoid_round_1_state_ = RoundOneSigmoidState();
	 	 sigmoid_round_2_state_ = RoundTwoSigmoidState();
	 	 sigmoid_round_3_state_ = RoundThreeSigmoidState();
	 	 sigmoid_round_4_state_ = RoundFourSigmoidState();
	 	 state_round_3_ = StateRound3();
		current_iteration_++;
	  } else {
	  	return InvalidArgumentError(absl::StrCat("Gradient Descent server received an"
			" unrecognized message, with case ", request->client_message_oneof_case()));
	  }

	total_client_message_size_ += request->ByteSizeLong();
	total_server_message_size_ += response->ByteSizeLong();
		
  if(shut_down()) {
		std::cout << "Server completed." <<std::endl
			<< "Total client message size = "<<  total_client_message_size_
				 << " bytes" <<std::endl
 			<< "Total server message size = " << total_server_message_size_
 				 << " bytes" <<std::endl
 			<< "Grand total message size = " 
							<< total_server_message_size_ + total_client_message_size_
 				 << " bytes" <<std::endl;
	}
	   return OkStatus();
}

}  // namespace logistic_regression
}  // namespace private_join_and_compute
