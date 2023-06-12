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

#include "const_round_secure_comparison_rpc_impl.h"

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

ConstRoundSecureComparisonRpcImpl::ConstRoundSecureComparisonRpcImpl(  
	std::vector<uint64_t> first_bit_share,
std::vector<uint64_t> comparison_input,
	SecureComparisonPrecomputedValue precomp,
	std::unique_ptr<ConstRoundSecureComparison> secure_comp)
		: 
		first_bit_share_(std::move(first_bit_share)),
		comparison_input_(std::move(comparison_input)),
		precomp_(std::move(precomp)), 
		secure_comp_(std::move(secure_comp)) {}

::grpc::Status ConstRoundSecureComparisonRpcImpl::Handle(::grpc::ServerContext* context,
                      const ConstRoundSecureComparisonClientMessage* request,
                      ConstRoundSecureComparisonServerMessage* response) {
	return ConvertStatus(HandleInternal(context, request, response));
}

Status ConstRoundSecureComparisonRpcImpl::HandleInternal(::grpc::ServerContext* context,
	const ConstRoundSecureComparisonClientMessage* request,
	ConstRoundSecureComparisonServerMessage* response)	{

	// Switch over different client message types
	if(request->has_client_start_message()) {
		ASSIGN_OR_RETURN(std::tie(round_1_state_,
			 									*response->mutable_server_round_1_message()),
		                 secure_comp_->GenerateComparisonRoundOneMessage(1,
										 precomp_, comparison_input_));
	} else if(request->has_client_round_1_message()) {
		std::cout << "Comparison Server received Round 1 Message." << std::endl;
		ASSIGN_OR_RETURN(std::tie(round_2_state_,
			 									*response->mutable_server_round_2_message()),
		                 secure_comp_->GenerateComparisonRoundTwoMessage(1,
										 precomp_, round_1_state_, 
										 request->client_round_1_message()));
	} else if(request->has_client_round_2_message()) {
		ASSIGN_OR_RETURN(std::tie(round_3_state_,
			 									*response->mutable_server_round_3_message()),
		                 secure_comp_->GenerateComparisonRoundThreeMessage(1,
										 precomp_, round_2_state_, 
										 request->client_round_2_message()));
	} else if(request->has_client_round_3_message()) {
		ASSIGN_OR_RETURN(auto partial_result,
		                 secure_comp_->GenerateComparisonResult(1,
										 precomp_, round_3_state_, 
										 request->client_round_3_message()));
		ASSIGN_OR_RETURN(std::vector<uint64_t> output,
			private_join_and_compute::secure_comparison::SecretSharedComparisonFinishReduction(
				                                     first_bit_share_,
				                                     partial_result,
				                                     comparison_input_.size()));
		*response->mutable_server_end_message() = EndMessage();
		finished_ = true;
	} else {
		return InvalidArgumentError(absl::StrCat("ConstRoundSecureComparison server received an"
		" unrecognized message, with case ", 
		request->client_message_oneof_case()));
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

}  // namespace applications
}  // namespace private_join_and_compute