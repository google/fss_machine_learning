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

#include "secure_comparison_rpc_impl.h"

#include "private_join_and_compute/util/status.inc"

namespace private_join_and_compute {
namespace secure_comparison {


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

SecureComparisonRpcImpl::SecureComparisonRpcImpl(  
		std::vector<uint64_t> inputs,
		size_t block_length,
		size_t num_splits,
		ComparisonEqualityGates comparison_equality_gates,
		ComparisonPreprocessedValues comparison_preprocessed_values
		): 
		inputs_(std::move(inputs)), 
		block_length_(std::move(block_length)), 
		num_splits_(std::move(num_splits)),
		gates_(std::move(comparison_equality_gates)),
		preprocessed_values_(std::move(comparison_preprocessed_values)){
		num_mult_rounds_ = log2(num_splits_);
		num_gates_in_current_mult_round_ = num_splits_/2;
}

::grpc::Status SecureComparisonRpcImpl::Handle(::grpc::ServerContext* context,
                      const SecureComparisonClientMessage* request,
                      SecureComparisonServerMessage* response) {
	return ConvertStatus(HandleInternal(context, request, response));
}

Status SecureComparisonRpcImpl::HandleInternal(::grpc::ServerContext* context,
	const SecureComparisonClientMessage* request,
	SecureComparisonServerMessage* response)	{

	// Switch over different client message types
	if(request->has_client_round_1_message()) {
		std::cout << "Comparison Server received Round 1 Message." << std::endl;
	  ASSIGN_OR_RETURN(
			std::tie(state_round_one_, *response->mutable_server_round_1_message()),
     ComparisonGenerateRoundOneMessage(
         inputs_,
         preprocessed_values_,
         num_splits_, block_length_));
    ASSIGN_OR_RETURN(
       combination_input_,
       ComparisonComputeShortComparisonEqualityPartyOne(
           gates_,
           state_round_one_,
           request->client_round_1_message(),
           preprocessed_values_,
           num_splits_, block_length_));
	} else if(request->has_client_multiplication_gate_message()) {
		std::cout << "Comparison Server received Multiplication Gate Message for"
			" round " << current_mult_round_ + 1 << std::endl;
		if (current_mult_round_ == num_mult_rounds_) {
			return InvalidArgumentError(absl::StrCat("Comparison server received a"
			"  client_multiplication_gate_message even though it has already "
			" completed", num_mult_rounds_ + 1, " multiplication rounds."));
		}
		
    ASSIGN_OR_RETURN(
        std::tie(
					mult_state_, 
					*response->mutable_server_multiplication_gate_message()),
        ComparisonGenerateNextRoundMessage(
            combination_input_,
            preprocessed_values_));
		
	  ASSIGN_OR_RETURN(
	      combination_input_,
	      ComparisonProcessNextRoundMessagePartyOne(
	          combination_input_,
	          mult_state_,
	          preprocessed_values_,
	          request->client_multiplication_gate_message(),
	          num_gates_in_current_mult_round_));
		
		current_mult_round_++;
		num_gates_in_current_mult_round_ /= 2;
		
		if (current_mult_round_ == num_mult_rounds_) {
			// The result is contained in combination_input_.
			std::cout << "Comparison Server finished " << std::endl;
			finished_ = true;
		}		
	} else {
		return InvalidArgumentError(absl::StrCat("Comparison server received an"
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

}  // namespace secure_comparison
}  // namespace private_join_and_compute