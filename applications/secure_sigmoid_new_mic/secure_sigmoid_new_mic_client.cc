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
#include "applications/secure_sigmoid_new_mic/secure_sigmoid_new_mic_rpc.grpc.pb.h"
#include "applications/secure_sigmoid_new_mic/secure_sigmoid_new_mic_rpc.pb.h"
#include "applications/secure_sigmoid_new_mic/secure_sigmoid_new_mic.pb.h"
#include "applications/secure_sigmoid_new_mic/secure_sigmoid_new_mic.h"
#include "include/grpcpp/server_builder.h"
#include "include/grpcpp/server_context.h"
#include "include/grpcpp/support/status.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "absl/strings/string_view.h"

ABSL_FLAG(std::string, port, "10.128.0.10:10501",
          "Port on which to contact server");

// FPE Params 
ABSL_FLAG(int, num_fractional_bits, 20, "num fractional bits");
ABSL_FLAG(int, num_ring_bits, 63, "num ring bits");

// Exponent Params
ABSL_FLAG(int, exponent_bound, 13, "exponent bound");

// Gradient Descent Params
ABSL_FLAG(size_t, num_examples, 10, "num examples");

// New MIC Params
ABSL_FLAG(size_t, block_length, 16, "new MIC block length");
ABSL_FLAG(size_t, num_splits, 4, "new MIC num splits");

namespace private_join_and_compute {
namespace applications {

namespace {

int ExecuteProtocol() {
	
  // Sigmoid setup
	const uint64_t modulus = (1ULL << absl::GetFlag(FLAGS_num_ring_bits));

  const size_t kLogGroupSize = 63;
  const uint64_t kIntervalCount = 10;
  const uint64_t kTaylorPolynomialDegree = 10;

  const std::vector<double> sigmoid_spline_lower_bounds{
      0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

  const std::vector<double> sigmoid_spline_upper_bounds = {
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

  const std::vector<double> sigmoid_spline_slope = {
      0.24979187478940013, 0.24854809833537939, 0.24608519499181072,
      0.24245143300792976, 0.23771671089402596, 0.23196975023940808,
      0.2253146594237077, 0.2178670895944635, 0.20975021497391394,
      0.2010907600500101};

  const std::vector<double> sigmoid_spline_yIntercept = {0.5,
                                                         0.5001243776454021,
                                                         0.5006169583141158,
                                                         0.5017070869092801,
                                                         0.5036009757548416,
                                                         0.5064744560821506,
                                                         0.5104675105715708,
                                                         0.5156808094520418,
                                                         0.5221743091484814,
                                                         0.5299678185799949};

  const applications::SecureSplineParameters sigmoid_spline_params{
      kLogGroupSize,
      kIntervalCount,
      static_cast<uint8_t>(absl::GetFlag(FLAGS_num_fractional_bits)),
      sigmoid_spline_lower_bounds,
      sigmoid_spline_upper_bounds,
      sigmoid_spline_slope,
      sigmoid_spline_yIntercept
  };

  const uint64_t kLargePrime = 9223372036854775783;  // 63 bit prime
	
  const ExponentiationParams kSampleLargeExpParams = {
      static_cast<uint8_t>(absl::GetFlag(FLAGS_exponent_bound)),
      kLargePrime};

  const applications::SecureSigmoidNewMicParameters sigmoid_params {
      kLogGroupSize,
      sigmoid_spline_params,
      static_cast<uint8_t>(absl::GetFlag(FLAGS_num_fractional_bits)),
      kTaylorPolynomialDegree,
      kSampleLargeExpParams,
			absl::GetFlag(FLAGS_block_length),
			absl::GetFlag(FLAGS_num_splits)
  };

  std::unique_ptr<applications::SecureSigmoidNewMic> secure_sigmoid = 
		applications::SecureSigmoidNewMic::Create(absl::GetFlag(FLAGS_num_examples),
			sigmoid_params).value();
	
	// Purposely incorrect: sample shares of x, y and theta at random.
  std::vector<uint64_t> sigmoid_input_share =
		internal::SampleShareOfZero(
			absl::GetFlag(FLAGS_num_examples),
	    modulus).value().first;

	// TODO actually make the client communicate the trusted setup.
	// Purposely incorrect: each party generates shares independently.
  // Initialize preprocessed shares in trusted setup.
  SigmoidPrecomputedValueNewMic sigmoid_precomputed_value =
		secure_sigmoid->PerformSigmoidPrecomputation().value().first;
	
	SecureSigmoidNewMicClientMessage client_message;
	SecureSigmoidNewMicServerMessage server_message;

	// Consider grpc::SslServerCredentials if not running locally.
	std::cout << "Client: Creating server stub..." << std::endl;
	 	grpc::ChannelArguments ch_args;
	  ch_args.SetMaxReceiveMessageSize(-1); // consider limiting max message size

  std::unique_ptr<SecureSigmoidNewMicRpc::Stub> stub =
      SecureSigmoidNewMicRpc::NewStub(::grpc::CreateCustomChannel(
          absl::GetFlag(FLAGS_port), grpc::InsecureChannelCredentials(), 			ch_args));

	std::cout << "Client: Starting Sigmoid "	<< std::endl;
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

	RoundOneSigmoidNewMicState sigmoid_round_1_state;
	RoundTwoSigmoidNewMicState sigmoid_round_2_state;
	RoundThreeSigmoidNewMicState sigmoid_round_3_state;
	RoundThreePointFiveSigmoidNewMicState sigmoid_round_3_point_5_state;
	RoundFourSigmoidNewMicState sigmoid_round_4_state;
	RoundFiveSigmoidNewMicState sigmoid_round_5_state;
		
		
	// Initiate server work.
	std::cout << "Client: Starting Sigmoid Server Start "	<< std::endl;
	
	uint64_t cq_index=1;
	void* got_tag;
	bool ok = false;
	::grpc::ClientContext client_context0;
	client_message = SecureSigmoidNewMicClientMessage();
	*client_message.mutable_start_message() = StartMessage();
	std::unique_ptr<grpc::ClientAsyncResponseReader<SecureSigmoidNewMicServerMessage> > rpc(stub->AsyncHandle(&client_context0, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
		
	// Run Sigmoid Round 1
		std::cout << "Client: Starting Sigmoid Server message 1 "	<< std::endl;
	
	client_message = SecureSigmoidNewMicClientMessage();
  std::tie(sigmoid_round_1_state,
	 *client_message.mutable_client_round_1_message()) = 
   secure_sigmoid->GenerateSigmoidRoundOneMessage(0,
                                                  sigmoid_precomputed_value,
																									sigmoid_input_share)
																										.value();
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

	// Run Sigmoid Round 2
	
		std::cout << "Client: Starting Sigmoid Server message 2 "	<< std::endl;
	
	client_start = std::chrono::high_resolution_clock::now();
	::grpc::ClientContext client_context1;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<SecureSigmoidNewMicServerMessage>>(stub->AsyncHandle(&client_context1, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
	client_message = SecureSigmoidNewMicClientMessage();
	std::tie(sigmoid_round_2_state,
	 		*client_message.mutable_client_round_2_message()) =
		secure_sigmoid->GenerateSigmoidRoundTwoMessage(0, sigmoid_precomputed_value,
        sigmoid_round_1_state, server_message.server_round_1_message()).value();
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
	
	// Run Sigmoid Round 3
		std::cout << "Client: Starting Sigmoid Server message 3 "	<< std::endl;
	
	client_start = std::chrono::high_resolution_clock::now();
	

	::grpc::ClientContext client_context2;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<SecureSigmoidNewMicServerMessage>>(stub->AsyncHandle(&client_context2, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
	client_message = SecureSigmoidNewMicClientMessage();
	std::tie(sigmoid_round_3_state,
	 		*client_message.mutable_client_round_3_message()) =
		secure_sigmoid->GenerateSigmoidRoundThreeMessage(0, 
				sigmoid_precomputed_value,
        sigmoid_round_2_state, server_message.server_round_2_message()).value();
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
	
	
	// Run Sigmoid Round 3.5
		std::cout << "Client: Starting Sigmoid Server message 3.5 "	<< std::endl;
	
	client_start = std::chrono::high_resolution_clock::now();
	
	::grpc::ClientContext client_context3;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<SecureSigmoidNewMicServerMessage>>(stub->AsyncHandle(&client_context3, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
	client_message = SecureSigmoidNewMicClientMessage();
	std::tie(sigmoid_round_3_point_5_state,
	 		*client_message.mutable_client_round_3_point_5_message()) =
		secure_sigmoid->GenerateSigmoidRoundThreePointFiveMessage(0, 
				sigmoid_precomputed_value,
        sigmoid_round_3_state, server_message.server_round_3_message()).value();
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            client_end - client_start).count())/ 1e6;

	server_start = std::chrono::high_resolution_clock::now();
	GPR_ASSERT(cq.Next(&got_tag, &ok));
	GPR_ASSERT(got_tag == (void*) cq_index);
	GPR_ASSERT(ok);
	
	if (!grpc_status.ok()) {
		std::cerr << "Client: Failed on message round 3.5 with status " <<
			grpc_status.error_code() << " error_message: " <<
		  grpc_status.error_message() << std::endl;
		return 1;
	}
	server_end = std::chrono::high_resolution_clock::now();
	pone_time_incl_comm +=
		(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;
	
	// Run Sigmoid Round 4
		std::cout << "Client: Starting Sigmoid Server message 4 "	<< std::endl;
	
	client_start = std::chrono::high_resolution_clock::now();
	

	::grpc::ClientContext client_context3_point_5;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<SecureSigmoidNewMicServerMessage>>(stub->AsyncHandle(&client_context3_point_5, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
	client_message = SecureSigmoidNewMicClientMessage();
	std::tie(sigmoid_round_4_state,
	 		*client_message.mutable_client_round_4_message()) =
		secure_sigmoid->GenerateSigmoidRoundFourMessage(0, 
				sigmoid_precomputed_value,
        sigmoid_round_3_point_5_state,
				server_message.server_round_3_point_5_message()).value();
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            client_end - client_start).count())/ 1e6;

	server_start = std::chrono::high_resolution_clock::now();
	GPR_ASSERT(cq.Next(&got_tag, &ok));
	GPR_ASSERT(got_tag == (void*) cq_index);
	GPR_ASSERT(ok);

	if (!grpc_status.ok()) {
		std::cerr << "Client: Failed on message round 4 with status " <<
			grpc_status.error_code() << " error_message: " <<
		  grpc_status.error_message() << std::endl;
		return 1;
	}
	server_end = std::chrono::high_resolution_clock::now();
	pone_time_incl_comm +=
		(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;
	
	// Run Sigmoid Round 5
		std::cout << "Client: Starting Sigmoid Server message 5 "	<< std::endl;
	
	client_start = std::chrono::high_resolution_clock::now();
	
	::grpc::ClientContext client_context4;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<SecureSigmoidNewMicServerMessage>>(stub->AsyncHandle(&client_context4, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
	client_message = SecureSigmoidNewMicClientMessage();
	std::tie(sigmoid_round_5_state,
	 		*client_message.mutable_client_round_5_message()) =
		secure_sigmoid->GenerateSigmoidRoundFiveMessage(0, 
				sigmoid_precomputed_value,
        sigmoid_round_4_state,
				server_message.server_round_4_message()).value();
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            client_end - client_start).count())/ 1e6;

	server_start = std::chrono::high_resolution_clock::now();
	GPR_ASSERT(cq.Next(&got_tag, &ok));
	GPR_ASSERT(got_tag == (void*) cq_index);
	GPR_ASSERT(ok);
	

	
	if (!grpc_status.ok()) {
		std::cerr << "Client: Failed on message round 5 with status " <<
			grpc_status.error_code() << " error_message: " <<
		  grpc_status.error_message() << std::endl;
		return 1;
	}
	server_end = std::chrono::high_resolution_clock::now();
	pone_time_incl_comm +=
		(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;
	
	// Compute Result.
	std::cout << "Client: Starting Sigmoid Server result "	<< std::endl;
	
	client_start = std::chrono::high_resolution_clock::now();
	::grpc::ClientContext client_context5;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<SecureSigmoidNewMicServerMessage>>(stub->AsyncHandle(&client_context5, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
	
	std::vector<uint64_t> raw_sigmoid_output_shares = 
		secure_sigmoid->GenerateSigmoidResult(0,
	                                            sigmoid_precomputed_value,
	                                            sigmoid_round_5_state,
																							server_message.
																								server_round_5_message())
																					.value();

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
	std::cout << "Completed run" << std::endl << "num_fractional_bits="
		<< absl::GetFlag(FLAGS_num_fractional_bits) << std::endl
	  << "num_ring_bits=" << absl::GetFlag(FLAGS_num_ring_bits) << std::endl
	  << "num_examples=" << absl::GetFlag(FLAGS_num_examples) << std::endl
	  << "num_splits=" << absl::GetFlag(FLAGS_num_splits) << std::endl
	  << "block_length=" << absl::GetFlag(FLAGS_block_length) << std::endl
	  << "Client time total (s) =" << pzero_time <<std::endl
	  << "Server time (incl. comm) total (s) = " << pone_time_incl_comm <<std::endl
	      << "End to End time (excluding preprocessing) total (s) = " << end_to_end_time <<std::endl;

  return 0;
}

}  // namespace
}  // namespace applications
}  // namespace private_join_and_compute

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  return private_join_and_compute::applications::ExecuteProtocol();
}
