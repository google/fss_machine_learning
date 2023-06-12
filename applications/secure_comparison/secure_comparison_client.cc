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

#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "glog/logging.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "include/grpc/grpc_security_constants.h"
#include "include/grpcpp/grpcpp.h"
#include "include/grpcpp/security/server_credentials.h"
#include "applications/secure_comparison/secure_comparison_rpc.grpc.pb.h"
#include "applications/secure_comparison/secure_comparison_rpc.pb.h"
#include "applications/secure_comparison/secure_comparison.pb.h"
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

// FPE Params 
ABSL_FLAG(int, num_fractional_bits, 15, "num fractional bits");
ABSL_FLAG(int, num_ring_bits, 63, "num ring bits");

// Comparison Params
ABSL_FLAG(size_t, num_inputs, 10, "num inputs");
ABSL_FLAG(size_t, block_length, 16, "block length");
ABSL_FLAG(size_t, num_splits, 4, "num splits or blocks");

namespace private_join_and_compute {
namespace secure_comparison {

namespace {

int ExecuteProtocol() {
	
	size_t num_inputs = absl::GetFlag(FLAGS_num_inputs);
	size_t block_length =	absl::GetFlag(FLAGS_block_length);
	size_t num_splits =	absl::GetFlag(FLAGS_num_splits);
	size_t num_mult_rounds = log2(num_splits);
	size_t num_mult_gates_in_current_round = num_splits/2;
	
  // Setup
	// Purposely incorrect: each party generates precomputation independently.
  auto precomp = 
      internal::ComparisonPrecomputation(
          num_inputs, 
					block_length, 
					num_splits).value();

  auto short_gates = std::move(std::get<0>(precomp));
  auto precomputed_values = std::move(std::get<2>(precomp));
	// Input is hardcoded to all 1 for the time being
	std::vector<uint64_t> inputs(num_inputs, 1);
	
	SecureComparisonClientMessage client_message;
	SecureComparisonServerMessage server_message;

	// Consider grpc::SslServerCredentials if not running locally.
	std::cout << "Client: Creating server stub..." << std::endl;
	 	grpc::ChannelArguments ch_args;
	  ch_args.SetMaxReceiveMessageSize(-1); // consider limiting max message size

  std::unique_ptr<SecureComparisonRpc::Stub> stub =
      SecureComparisonRpc::NewStub(::grpc::CreateCustomChannel(
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

	ComparisonStateRoundOne state_round_one;
	ComparisonShortComparisonEquality combination_input;
	BatchedMultState mult_state;
		
	// Run Comparison Round 1
	client_message = SecureComparisonClientMessage();
  std::tie(state_round_one,
	 *client_message.mutable_client_round_1_message()) = 
     ComparisonGenerateRoundOneMessage(
         inputs,
         precomputed_values,
         num_splits, block_length).value();
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
	        client_end - client_start).count())/ 1e6;

	server_start = std::chrono::high_resolution_clock::now();
	server_message = SecureComparisonServerMessage();
	::grpc::ClientContext client_context1;
	grpc_status =
	stub->Handle(&client_context1, client_message, &server_message);
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
	
	
	client_start = std::chrono::high_resolution_clock::now();
	// Generate initial input for mult rounds.
  combination_input =
      ComparisonComputeShortComparisonEqualityPartyZero(
          short_gates,
          state_round_one,
          server_message.server_round_1_message(),
          precomputed_values,
          num_splits, block_length).value();
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            client_end - client_start).count())/ 1e6;
	
	// Run Comparison Multiplication Rounds
	for (size_t current_mult_round = 0; 
			current_mult_round < num_mult_rounds;
			current_mult_round++) {
	client_start = std::chrono::high_resolution_clock::now();
	
	client_message = SecureComparisonClientMessage();
  std::tie(mult_state, 
		*client_message.mutable_client_multiplication_gate_message()) =
      ComparisonGenerateNextRoundMessage(
          combination_input,
          precomputed_values).value();
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            client_end - client_start).count())/ 1e6;

	server_start = std::chrono::high_resolution_clock::now();
	server_message = SecureComparisonServerMessage();
	::grpc::ClientContext client_context_mult;
	grpc_status =
		stub->Handle(&client_context_mult, client_message, &server_message);
	if (!grpc_status.ok()) {
		std::cerr << "Client: Failed on mult. message with status " <<
			grpc_status.error_code() << " error_message: " <<
		  grpc_status.error_message() << std::endl;
		return 1;
	}
	server_end = std::chrono::high_resolution_clock::now();
	pone_time_incl_comm +=
		(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;
	
	client_start = std::chrono::high_resolution_clock::now();
  combination_input =
      ComparisonProcessNextRoundMessagePartyZero(
          combination_input,
          mult_state,
          precomputed_values,
          server_message.server_multiplication_gate_message(),
          num_mult_gates_in_current_round).value();
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            client_end - client_start).count())/ 1e6;
	
	
	num_mult_gates_in_current_round /= 2;
	}

	// Result is in combination_input
	auto end = std::chrono::high_resolution_clock::now();
  end_to_end_time = (std::chrono::duration_cast<std::chrono::microseconds>(
          end - start).count())
      / 1e6;


	// Print results
	std::cout << "Completed run" << std::endl << "num_fractional_bits="
		<< absl::GetFlag(FLAGS_num_fractional_bits) << std::endl
	  << "num_ring_bits=" << absl::GetFlag(FLAGS_num_ring_bits) << std::endl
	  << "num_inputs=" << absl::GetFlag(FLAGS_num_inputs) << std::endl
		<< "block_length=" << absl::GetFlag(FLAGS_block_length) << std::endl
		<< "num_splits=" << absl::GetFlag(FLAGS_num_splits) << std::endl
	  << "Client time total (s) =" << pzero_time <<std::endl
	  << "Server time (incl. comm) total (s) = " << pone_time_incl_comm <<std::endl
	      << "End to End time (excluding preprocessing) total (s) = " << end_to_end_time <<std::endl;

  return 0;
}

}  // namespace
}  // namespace secure_comparison
}  // namespace private_join_and_compute

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  return private_join_and_compute::secure_comparison::ExecuteProtocol();
}
