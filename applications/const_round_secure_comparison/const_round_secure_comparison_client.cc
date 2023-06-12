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

#include <iostream>
#include <memory>
#include <string>
#include <random>
#include <chrono>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "include/grpc/grpc_security_constants.h"
#include "include/grpcpp/grpcpp.h"
#include "include/grpcpp/security/server_credentials.h"
#include "applications/const_round_secure_comparison/const_round_secure_comparison_rpc.grpc.pb.h"
#include "applications/const_round_secure_comparison/const_round_secure_comparison_rpc.pb.h"
#include "applications/const_round_secure_comparison/const_round_secure_comparison.pb.h"
#include "applications/const_round_secure_comparison/const_round_secure_comparison.h"
#include "applications/secure_comparison/secure_comparison.h"
#include "include/grpcpp/server_builder.h"
#include "include/grpcpp/server_context.h"
#include "include/grpcpp/support/status.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "absl/strings/string_view.h"

ABSL_FLAG(std::string, port, "0.0.0.0:10501",
          "Port on which to contact server");

ABSL_FLAG(size_t, num_inputs, 1000, "num inputs");
ABSL_FLAG(size_t, string_length, 62, "string length");
ABSL_FLAG(size_t, num_pieces, 4, "num pieces");
ABSL_FLAG(size_t, piece_length, 16, "piece_length");

namespace private_join_and_compute {
namespace applications {

namespace {

int ExecuteProtocol() {
	
	const size_t num_inputs = absl::GetFlag(FLAGS_num_inputs);
	const size_t string_length = absl::GetFlag(FLAGS_string_length);
	const size_t num_pieces = absl::GetFlag(FLAGS_num_pieces);
	const size_t piece_length = absl::GetFlag(FLAGS_piece_length);
	const uint64_t modulus = 1ULL << string_length;


	const SecureComparisonParameters secure_comparison_params{
	        string_length,
	        num_pieces,
	        piece_length
	};
	
	auto prng = BasicRng::Create("").value();
	
  // Setup
	// Purposely incorrect: each party generates precomputation independently.
  std::unique_ptr<ConstRoundSecureComparison> seccomp =
                       ConstRoundSecureComparison::Create(num_inputs, 
											 secure_comparison_params).value();

  SecureComparisonPrecomputedValue precomp = seccomp->PerformComparisonPrecomputation().value().first;

  // Comparing x > y
  std::vector<uint64_t> comparison_inputs_x (num_inputs, 0); // = {0, 1, 2};

  std::vector<uint64_t> comparison_inputs_y (num_inputs, 0);//= {3, 1, 1};

  std::vector<uint64_t> share_of_comparison_inputs_x =
                      SampleVectorFromPrng(num_inputs, modulus, prng.get()).value();

  std::vector<uint64_t> share_of_comparison_inputs_y =
                      SampleVectorFromPrng(num_inputs, modulus, prng.get()).value();


  // Reduce secret-shared comparison to non-secret shared
	std::vector<uint64_t> first_bit_share;
	std::vector<uint64_t> comparison_input;


 	std::tie(first_bit_share, comparison_input) =
                      private_join_and_compute::secure_comparison::SecretSharedComparisonPrepareInputsPartyOne(
                          share_of_comparison_inputs_x,
                          share_of_comparison_inputs_y,
                          num_inputs,
                          modulus,
                          string_length).value();

	// Consider grpc::SslServerCredentials if not running locally.
	std::cout << "Client: Creating server stub..." << std::endl;
	 	grpc::ChannelArguments ch_args;
	  ch_args.SetMaxReceiveMessageSize(-1); // consider limiting max message size

  std::unique_ptr<ConstRoundSecureComparisonRpc::Stub> stub =
      ConstRoundSecureComparisonRpc::NewStub(::grpc::CreateCustomChannel(
          absl::GetFlag(FLAGS_port), grpc::InsecureChannelCredentials(), 			ch_args));

	std::cout << "Client: Starting Comparison "	<< std::endl;
  double pzero_time = 0;
  double pone_time_incl_comm = 0;
  double end_to_end_time = 0;


	auto start = std::chrono::high_resolution_clock::now();
	auto client_start = start;
	auto client_end = start;
	auto server_start = start;
	auto server_end = start;
	
	::grpc::Status grpc_status;
	::grpc::CompletionQueue cq;

	RoundOneSecureComparisonState round_1_state;
	RoundTwoSecureComparisonState round_2_state;
	RoundThreeSecureComparisonState round_3_state;
		
	ConstRoundSecureComparisonClientMessage client_message;
	ConstRoundSecureComparisonServerMessage server_message;
		
	// Initiate server work.
	std::cout << "Client: Starting Server's work "	<< std::endl;
	
	uint64_t cq_index=1;
	void* got_tag;
	bool ok = false;
	::grpc::ClientContext client_context0;
	client_message = ConstRoundSecureComparisonClientMessage();
	*client_message.mutable_client_start_message() = StartMessage();
	std::unique_ptr<grpc::ClientAsyncResponseReader<ConstRoundSecureComparisonServerMessage> > rpc(stub->AsyncHandle(&client_context0, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
		
	// Run SecComp Round 1
		std::cout << "Client: Starting SecComp Server message 1 "	<< std::endl;
	
	client_message = ConstRoundSecureComparisonClientMessage();
  std::tie(round_1_state,
	 *client_message.mutable_client_round_1_message()) = 
   seccomp->GenerateComparisonRoundOneMessage(0,
										 precomp, comparison_input).value();
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
	        client_end - client_start).count())/ 1e6;

	server_start = std::chrono::high_resolution_clock::now();
	GPR_ASSERT(cq.Next(&got_tag, &ok));
	GPR_ASSERT(got_tag == (void*) cq_index);
	GPR_ASSERT(ok);
	if (!grpc_status.ok()) {
	std::cerr << "Client: Failed on message round 1 with status " <<
		grpc_status.error_code() << " error_message: " <<
	  grpc_status.error_message() << std::endl;
	return 1;
	}
	server_end = std::chrono::high_resolution_clock::now();
	pone_time_incl_comm +=
	(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
		server_start).count())/ 1e6;

	// Run SecComp Round 2
	
		std::cout << "Client: Starting SecComp Server message 2 "	<< std::endl;
	
	client_start = std::chrono::high_resolution_clock::now();
	::grpc::ClientContext client_context1;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<ConstRoundSecureComparisonServerMessage>>(stub->AsyncHandle(&client_context1, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
	client_message = ConstRoundSecureComparisonClientMessage();
	std::tie(round_2_state,
	 		*client_message.mutable_client_round_2_message()) =
		seccomp->GenerateComparisonRoundTwoMessage(0, precomp,
        round_1_state, server_message.server_round_1_message()).value();
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            client_end - client_start).count())/ 1e6;

	server_start = std::chrono::high_resolution_clock::now();
	GPR_ASSERT(cq.Next(&got_tag, &ok));
	GPR_ASSERT(got_tag == (void*) cq_index);
	GPR_ASSERT(ok);
	if (!grpc_status.ok()) {
		std::cerr << "Client: Failed on message round 2 with status " <<
			grpc_status.error_code() << " error_message: " <<
		  grpc_status.error_message() << std::endl;
		return 1;
	}
	server_end = std::chrono::high_resolution_clock::now();
	pone_time_incl_comm +=
		(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;
	
	// Run SecComp Round 3
		std::cout << "Client: Starting SecComp Server message 3 "	<< std::endl;
	
	client_start = std::chrono::high_resolution_clock::now();
	

	::grpc::ClientContext client_context2;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<ConstRoundSecureComparisonServerMessage>>(stub->AsyncHandle(&client_context2, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
	client_message = ConstRoundSecureComparisonClientMessage();
	std::tie(round_3_state,
	 		*client_message.mutable_client_round_3_message()) =
		seccomp->GenerateComparisonRoundThreeMessage(0, precomp,
        round_2_state, server_message.server_round_2_message()).value();
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            client_end - client_start).count())/ 1e6;

	server_start = std::chrono::high_resolution_clock::now();
	GPR_ASSERT(cq.Next(&got_tag, &ok));
	GPR_ASSERT(got_tag == (void*) cq_index);
	GPR_ASSERT(ok);
	if (!grpc_status.ok()) {
		std::cerr << "Client: Failed on message round 3 with status " <<
			grpc_status.error_code() << " error_message: " <<
		  grpc_status.error_message() << std::endl;
		return 1;
	}
	server_end = std::chrono::high_resolution_clock::now();
	pone_time_incl_comm +=
		(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;
	
	
	// Compute Result.
	std::cout << "Client: Starting SecComp Server result "	<< std::endl;
	
	client_start = std::chrono::high_resolution_clock::now();
	::grpc::ClientContext client_context5;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<ConstRoundSecureComparisonServerMessage>>(stub->AsyncHandle(&client_context5, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
	
	auto partial_result =
	                 seccomp->GenerateComparisonResult(0,
									 precomp, round_3_state, 
									 server_message.server_round_3_message()).value();
	std::vector<uint64_t> output =
		private_join_and_compute::secure_comparison::SecretSharedComparisonFinishReduction(
			                                     first_bit_share,
			                                     partial_result,
			                                     comparison_input.size()).value();

		client_end = std::chrono::high_resolution_clock::now();
		pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
	            client_end - client_start).count())/ 1e6;

		server_start = std::chrono::high_resolution_clock::now();
		GPR_ASSERT(cq.Next(&got_tag, &ok));
		GPR_ASSERT(got_tag == (void*) cq_index);
		GPR_ASSERT(ok);
	
		if (!grpc_status.ok()) {
			std::cerr << "Client: Failed on end message " <<
				grpc_status.error_code() << " error_message: " <<
			  grpc_status.error_message() << std::endl;
			return 1;
		}
		server_end = std::chrono::high_resolution_clock::now();
		pone_time_incl_comm +=
			(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
				server_start).count())/ 1e6;

  auto end = std::chrono::high_resolution_clock::now();

  // Add in preprocessing phase. For the online phase, since the initial round for client and server can be done at the same time
  end_to_end_time = (std::chrono::duration_cast<std::chrono::microseconds>(
          end-start).count())
      / 1e6;


	// Print results
	std::cout << "Completed run" << std::endl << "num_inputs="
		<< num_inputs << std::endl
	  << "Client time total (s) =" << pzero_time <<std::endl
	  << "Server time (incl. comm) total (s) = " << pone_time_incl_comm <<std::endl
	      << "End to End time (excluding preprocessing) total (s) = " << end_to_end_time <<std::endl;

  return 0;
}

}  // namespace
}  // namespace applications
}  // namespace private_join_and_compute

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  return private_join_and_compute::applications::ExecuteProtocol();
}
