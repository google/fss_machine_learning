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

#include "secure_sigmoid_rpc_impl.h"

#include "private_join_and_compute/util/status.inc"

namespace private_join_and_compute {
namespace applications {


namespace {
	
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

SecureSigmoidRpcImpl::SecureSigmoidRpcImpl(
		std::vector<uint64_t> sigmoid_input_shares,
		SigmoidPrecomputedValue
			sigmoid_precomputed_value,
		applications::SecureSigmoidParameters sigmoid_params):
		sigmoid_input_shares_(std::move(sigmoid_input_shares)),
	  sigmoid_precomputed_value_(std::move(sigmoid_precomputed_value)),
		sigmoid_params_(std::move(sigmoid_params)){
	secure_sigmoid_ = SecureSigmoid::Create(sigmoid_input_shares_.size(),
		sigmoid_params_).value();
}

::grpc::Status SecureSigmoidRpcImpl::Handle(::grpc::ServerContext* context,
                      const SecureSigmoidClientMessage* request,
                      SecureSigmoidServerMessage* response) {
	return ConvertStatus(HandleInternal(context, request, response));
}

Status SecureSigmoidRpcImpl::HandleInternal(::grpc::ServerContext* context,
	const SecureSigmoidClientMessage* request,
	SecureSigmoidServerMessage* response)	{

		// Switch over different client message types
	  if(request->has_start_message()) {
			std::cout<< "SecureSigmoidRpcImpl: Executing round 1"<<  std::endl;
	  	// execute Sigmoid round 1.
		ASSIGN_OR_RETURN(
			std::tie(state_round_1_, *response->mutable_server_round_1_message()), 			secure_sigmoid_->GenerateSigmoidRoundOneMessage(1,
			 	sigmoid_precomputed_value_, sigmoid_input_shares_));
	  } else if(request->has_client_round_1_message()) {
		std::cout<< "SecureSigmoidRpcImpl: Executing round 2"<<  std::endl;
		// execute Sigmoid Round 2.
	  	
			ASSIGN_OR_RETURN(
				std::tie(state_round_2_, *response->mutable_server_round_2_message()), 			secure_sigmoid_->GenerateSigmoidRoundTwoMessage(1,
				 	sigmoid_precomputed_value_, state_round_1_,
					request->client_round_1_message()));
	  } else if(request->has_client_round_2_message()) {
			std::cout<< "SecureSigmoidRpcImpl: Executing round 3"<<  std::endl;
		// execute Sigmoid Round 3.
			ASSIGN_OR_RETURN(
				std::tie(state_round_3_, *response->mutable_server_round_3_message()), 			secure_sigmoid_->GenerateSigmoidRoundThreeMessage(1,
				 	sigmoid_precomputed_value_, state_round_2_,
					request->client_round_2_message()));
	  } else if(request->has_client_round_3_message()) {
			std::cout<< "SecureSigmoidRpcImpl: Executing round 4"<<  std::endl;
		// execute Sigmoid Round 4.
			ASSIGN_OR_RETURN(
				std::tie(state_round_4_, *response->mutable_server_round_4_message()), 			secure_sigmoid_->GenerateSigmoidRoundFourMessage(1,
				 	sigmoid_precomputed_value_, state_round_3_,
					request->client_round_3_message()));
				}
					else if(request->has_client_round_4_message()) { 
				std::cout<< "SecureSigmoidRpcImpl: Computing result"<<  std::endl;
			ASSIGN_OR_RETURN(auto sigmoid_result, 
				secure_sigmoid_->GenerateSigmoidResult(
			            1,
			            sigmoid_precomputed_value_,
			            state_round_4_,
			            request->client_round_4_message()));
			*response->mutable_end_message() = EndMessage::default_instance();
			finished_ = true;
		}
	 else {
	  	return InvalidArgumentError(absl::StrCat("Sigmoid server received an"
			" unrecognized message, with case ", 
			request->client_message_oneof_case()));
	  }

  previous_client_message_ = *request;
	
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

}  // namespace applications
}  // namespace private_join_and_compute