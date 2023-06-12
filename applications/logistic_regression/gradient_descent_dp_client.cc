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
#include "applications/logistic_regression/gradient_descent_dp_rpc.grpc.pb.h"
#include "applications/logistic_regression/gradient_descent_dp_rpc.pb.h"
#include "applications/logistic_regression/gradient_descent_dp_messages.pb.h"
#include "include/grpcpp/server_builder.h"
#include "include/grpcpp/server_context.h"
#include "include/grpcpp/support/status.h"
#include "applications/logistic_regression/gradient_descent_dp.h"
#include "applications/secure_sigmoid_new_mic/secure_sigmoid_new_mic.h"
#include "secret_sharing_mpc/gates/correlated_matrix_product.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "absl/strings/string_view.h"
#include "poisson_regression/beaver_triple_utils.h"

ABSL_FLAG(std::string, port, "10.128.0.10:10501",
          "Port on which to contact server");

ABSL_FLAG(size_t, num_iterations, 6,
          "The number of iterations to execute.");

// FPE Params 
ABSL_FLAG(int, num_fractional_bits, 20, "num fractional bits");
ABSL_FLAG(int, num_ring_bits, 63, "num ring bits");

// Exponent Params
ABSL_FLAG(int, exponent_bound, 13, "exponent bound");

// Gradient Descent Params
ABSL_FLAG(size_t, num_examples, 10, "num examples");
ABSL_FLAG(size_t, num_features, 10, "num features");
ABSL_FLAG(double, alpha, 0.0001, "alpha to use for training");
ABSL_FLAG(double, lambda, 0, "lambda regularization parameter to use for training");

// New MIC Params
ABSL_FLAG(bool, use_new_mic, false, "Whether to use the 6-round new MIC");
ABSL_FLAG(size_t, block_length, 16, "new MIC block length");
ABSL_FLAG(size_t, num_splits, 4, "new MIC num splits");

namespace private_join_and_compute {
namespace logistic_regression_dp {

namespace {

int ExecuteProtocol() {

	bool use_new_mic = absl::GetFlag(FLAGS_use_new_mic);

	// Generate parameters for gradient descent.
  const GradientDescentParams gd_params = {
      .num_examples = absl::GetFlag(FLAGS_num_examples),
      .num_features = absl::GetFlag(FLAGS_num_features),
      .num_iterations = absl::GetFlag(FLAGS_num_iterations),
      .num_ring_bits = static_cast<uint8_t>(absl::GetFlag(FLAGS_num_ring_bits)),
      .num_fractional_bits =
				 static_cast<uint8_t>(absl::GetFlag(FLAGS_num_fractional_bits)),
      .alpha = absl::GetFlag(FLAGS_alpha),
      .lambda = absl::GetFlag(FLAGS_lambda),
      .modulus = (1ULL << absl::GetFlag(FLAGS_num_ring_bits))};

  const FixedPointElementFactory::Params fpe_params = {
      .num_fractional_bits =
				 static_cast<uint8_t>(absl::GetFlag(FLAGS_num_fractional_bits)),
      .num_ring_bits = static_cast<uint8_t>(absl::GetFlag(FLAGS_num_ring_bits)),
      .fractional_multiplier = 1ULL << absl::GetFlag(FLAGS_num_fractional_bits),
      .integer_ring_modulus = 1ULL << (absl::GetFlag(FLAGS_num_ring_bits) - absl::GetFlag(FLAGS_num_fractional_bits)),
      .primary_ring_modulus = 1ULL << absl::GetFlag(FLAGS_num_ring_bits)};
	
  // Sigmoid setup

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

  const applications::SecureSigmoidParameters sigmoid_params {
      kLogGroupSize,
      sigmoid_spline_params,
      static_cast<uint8_t>(absl::GetFlag(FLAGS_num_fractional_bits)),
      kTaylorPolynomialDegree,
      kSampleLargeExpParams
  };

  std::unique_ptr<applications::SecureSigmoid> secure_sigmoid = 
		applications::SecureSigmoid::Create(gd_params.num_examples,
			sigmoid_params).value();
	
	const applications::SecureSigmoidNewMicParameters sigmoid_params_new_mic {
	    kLogGroupSize,
	    sigmoid_spline_params,
	    static_cast<uint8_t>(absl::GetFlag(FLAGS_num_fractional_bits)),
	    kTaylorPolynomialDegree,
	    kSampleLargeExpParams,
			absl::GetFlag(FLAGS_block_length),
			absl::GetFlag(FLAGS_num_splits)
	};
	std::unique_ptr<applications::SecureSigmoidNewMic> secure_sigmoid_new_mic = 
		applications::SecureSigmoidNewMic::Create(gd_params.num_examples,
			sigmoid_params_new_mic).value();
	
	// Purposely incorrect: sample shares of x and y.
  std::vector<uint64_t> share_x =
		internal::SampleShareOfZero(
			gd_params.num_examples * gd_params.num_features,
	    gd_params.modulus).value().first;		
  std::vector<uint64_t> share_y = internal::SampleShareOfZero(
			gd_params.num_examples,
	    gd_params.modulus).value().first;
	// Set theta to 0.
 	std::vector<double> theta(gd_params.num_features, 0.0);
	
	// TODO actually make the client communicate the trusted setup.
	// Purposely incorrect: each party generates shares independently.
  // Initialize preprocessed shares in trusted setup.
  std::vector<applications::SigmoidPrecomputedValue> sigmoid_p0;
	std::vector<applications::SigmoidPrecomputedValueNewMic> sigmoid_p0_new_mic;
  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
      beaver_triple_matrix_b_c_p0;

  // Generates mask for X and for X^T. These masks are part of the correlated
  // matrix product and do not change across iterations
  auto vector_a_masks_x_transpose =
      SampleBeaverTripleMatrixA(gd_params.num_features, gd_params.num_examples,
                                gd_params.modulus).value();
																
	// Initialize DP noise also in trusted setup (set dp noise to 0 for now)
	std::vector<std::vector<uint64_t>> share_noise_p0 (gd_params.num_iterations,
	std::vector<uint64_t>(gd_params.num_features, 0));

  for (size_t idx = 0; idx < gd_params.num_iterations; idx++) {

    // sigmoid(u): u (#examples x 1)
		// Performs precomputations for both MIC options, only the right one will be
		// used.
    std::pair<applications::SigmoidPrecomputedValue, 		applications::SigmoidPrecomputedValue> preCompRes;
    preCompRes = secure_sigmoid->PerformSigmoidPrecomputation().value();
    sigmoid_p0.push_back(std::move(preCompRes.first));
		sigmoid_p0_new_mic.push_back(std::move(
			secure_sigmoid_new_mic->PerformSigmoidPrecomputation().value().first));


    // X.transpose() * d: X.transpose() (#features * #examples),
    // d (#examples x 1).
    auto matrix_shares_3 =
                         SampleBeaverTripleMatrixBandC(
                             vector_a_masks_x_transpose, gd_params.num_features,
                             gd_params.num_examples, 1, 
														 gd_params.modulus).value();
    beaver_triple_matrix_b_c_p0.push_back(matrix_shares_3.first);
  }
  // Initialize LogRegShareProvider with preprocessed shares.
  std::unique_ptr<LogRegDPShareProvider> share_provider;
	
	if(use_new_mic) {
		share_provider = 
  std::make_unique<LogRegDPShareProvider>(
			LogRegDPShareProvider::CreateNewMic(
      std::move(sigmoid_p0_new_mic), std::get<0>(vector_a_masks_x_transpose),
      beaver_triple_matrix_b_c_p0,
			std::move(share_noise_p0), gd_params.num_examples,
      gd_params.num_features, gd_params.num_iterations).value());
	}
	else {
		share_provider = 
  std::make_unique<LogRegDPShareProvider>(LogRegDPShareProvider::Create(
      sigmoid_p0, std::get<0>(vector_a_masks_x_transpose),
      beaver_triple_matrix_b_c_p0,
			std::move(share_noise_p0), gd_params.num_examples,
      gd_params.num_features, gd_params.num_iterations).value());
		}

  // Client acts as Party 0.
  std::unique_ptr<GradientDescentPartyZero> gradient_descent_party_zero = 
		absl::make_unique<GradientDescentPartyZero>(
			GradientDescentPartyZero::Init(
				share_x, share_y, theta,
			  std::move(share_provider),
				fpe_params, gd_params).value());
	
 	 StateMaskedXTranspose state_masked_x_transpose;
   MaskedXTransposeMessage server_masked_x_transpose_message;
   MaskedXTransposeMessage client_masked_x_transpose_message;

 	 // Intermediates for the gradient descent loop
 	 // Cached message from the client for the previous round.
	 const int num_iterations = absl::GetFlag(FLAGS_num_iterations);
	
	GradientDescentDpClientMessage client_message;
	GradientDescentDpServerMessage server_message;

	// Consider grpc::SslServerCredentials if not running locally.
	std::cout << "Client: Creating server stub..." << std::endl;
	 	grpc::ChannelArguments ch_args;
	  ch_args.SetMaxReceiveMessageSize(-1); // consider limiting max message size

  std::unique_ptr<GradientDescentDpRpc::Stub> stub =
      GradientDescentDpRpc::NewStub(::grpc::CreateCustomChannel(
          absl::GetFlag(FLAGS_port), grpc::InsecureChannelCredentials(), 			ch_args));

	std::cout << "Client: Executing " << num_iterations << " iterations..."
		<< std::endl;
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
	uint64_t cq_index=1;
	void* got_tag;
	bool ok = false;
	
	// One-time rounds.
  std::tie(state_masked_x_transpose, client_masked_x_transpose_message) = 
                       gradient_descent_party_zero->
												GenerateCorrelatedProductMessageForXTranspose().value();

	::grpc::ClientContext client_context_masked_x_transpose;
	client_message = GradientDescentDpClientMessage();
	server_message = GradientDescentDpServerMessage();
	*(client_message.mutable_client_masked_x_transpose_message()) = client_masked_x_transpose_message;
	grpc_status = stub->Handle(&client_context_masked_x_transpose,
	 client_message, &server_message);
	if (!grpc_status.ok()) {
				std::cerr << "Client: Failed on masked_x_transpose_message  with status " <<
					grpc_status.error_code() << " error_message: " <<
				  grpc_status.error_message() << std::endl;
				return 1;
			}
	server_masked_x_transpose_message  = server_message.server_masked_x_transpose_message();
	
	
	for (int i = 0; i < num_iterations; i++) {
		std::cout << "Executing iteration " << i+1 << std::endl;
		::grpc::ClientContext client_context0;
		client_message = GradientDescentDpClientMessage();
		*client_message.mutable_start_message() = StartMessage();
		std::unique_ptr<grpc::ClientAsyncResponseReader<GradientDescentDpServerMessage> > rpc(stub->AsyncHandle(&client_context0, client_message, &cq));
		rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
		
		SigmoidInput sigmoid_input;
  	 StateXTransposeD state_x_transpose_d;
		 StateReconstructGradient state_reconstruct_gradient;

  client_start = std::chrono::high_resolution_clock::now();
	// Compute Sigmoid Input
 	sigmoid_input = gradient_descent_party_zero->GenerateSigmoidInput().value();
		SigmoidOutput sigmoid_output;
	 //////////////////////////////////////////////////////////////////////////
	 //	 4-round Sigmoid
	 //////////////////////////////////////////////////////////////////////////	 
	 if (!use_new_mic) {
		 applications::SigmoidPrecomputedValue sigmoid_precomputed_value;
  	 applications::RoundOneSigmoidState sigmoid_round_1_state;
  	 applications::RoundTwoSigmoidState sigmoid_round_2_state;
  	 applications::RoundThreeSigmoidState sigmoid_round_3_state;
  	 applications::RoundFourSigmoidState sigmoid_round_4_state;

		 	sigmoid_precomputed_value = 
						gradient_descent_party_zero->share_provider_->
								GetSigmoidPrecomputedValue().value(); 

			// Run Sigmoid Round 1
			client_message = GradientDescentDpClientMessage();
	    std::tie(sigmoid_round_1_state,
			 *client_message.mutable_client_sigmoid_round_1_message()) = 
	     secure_sigmoid->GenerateSigmoidRoundOneMessage(0,
	        sigmoid_precomputed_value,
					sigmoid_input.sigmoid_input)
																												.value();
			client_end = std::chrono::high_resolution_clock::now();
			pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
			        client_end - client_start).count())/ 1e6;

			server_start = std::chrono::high_resolution_clock::now();
			GPR_ASSERT(cq.Next(&got_tag, &ok));
			GPR_ASSERT(got_tag == (void*) cq_index);
			GPR_ASSERT(ok);
			if (!grpc_status.ok()) {
			std::cerr << "Client: Failed on message round 2 iteration " << i <<
				" with status " <<
				grpc_status.error_code() << " error_message: " <<
			  grpc_status.error_message() << std::endl;
			return 1;
			}
			server_end = std::chrono::high_resolution_clock::now();
			pone_time_incl_comm +=
			(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
				server_start).count())/ 1e6;

		// Run Round 3 (Sigmoid Round 2)
		client_start = std::chrono::high_resolution_clock::now();

		::grpc::ClientContext client_context2;
		cq_index++;
		ok=false;
		 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<GradientDescentDpServerMessage>>(stub->AsyncHandle(&client_context2, client_message, &cq));
		rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
		
		client_message = GradientDescentDpClientMessage();
		std::tie(sigmoid_round_2_state,
		 		*client_message.mutable_client_sigmoid_round_2_message()) =
			secure_sigmoid->GenerateSigmoidRoundTwoMessage(0,
				 sigmoid_precomputed_value,
	       sigmoid_round_1_state,
				 server_message.server_sigmoid_round_1_message()).value();
		client_end = std::chrono::high_resolution_clock::now();
		pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
	            client_end - client_start).count())/ 1e6;

		server_start = std::chrono::high_resolution_clock::now();
		GPR_ASSERT(cq.Next(&got_tag, &ok));
		GPR_ASSERT(got_tag == (void*) cq_index);
		GPR_ASSERT(ok);
		if (!grpc_status.ok()) {
			std::cerr << "Client: Failed on message round 3 iteration " << i <<
				" with status " <<
				grpc_status.error_code() << " error_message: " <<
			  grpc_status.error_message() << std::endl;
			return 1;
		}
		server_end = std::chrono::high_resolution_clock::now();
		pone_time_incl_comm +=
			(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
				server_start).count())/ 1e6;

		// Run Round 4 (Sigmoid Round 3)
		client_start = std::chrono::high_resolution_clock::now();

		::grpc::ClientContext client_context3;
		cq_index++;
		ok=false;
		 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<GradientDescentDpServerMessage>>(stub->AsyncHandle(&client_context3, client_message, &cq));
		rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
		
		client_message = GradientDescentDpClientMessage();
		std::tie(sigmoid_round_3_state,
		 		*client_message.mutable_client_sigmoid_round_3_message()) =
			secure_sigmoid->GenerateSigmoidRoundThreeMessage(0, 
					sigmoid_precomputed_value,
	        sigmoid_round_2_state,
					server_message.server_sigmoid_round_2_message()).value();
		client_end = std::chrono::high_resolution_clock::now();
		pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
	            client_end - client_start).count())/ 1e6;

		server_start = std::chrono::high_resolution_clock::now();
		GPR_ASSERT(cq.Next(&got_tag, &ok));
		GPR_ASSERT(got_tag == (void*) cq_index);
		GPR_ASSERT(ok);
		if (!grpc_status.ok()) {
			std::cerr << "Client: Failed on message round 4 iteration " << i <<
				" with status " <<
				grpc_status.error_code() << " error_message: " <<
			  grpc_status.error_message() << std::endl;
			return 1;
		}
		server_end = std::chrono::high_resolution_clock::now();
		pone_time_incl_comm +=
			(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
				server_start).count())/ 1e6;

		// Run Round 5 (Sigmoid Round 4)
		client_start = std::chrono::high_resolution_clock::now();
		::grpc::ClientContext client_context4;
		cq_index++;
		ok=false;
		 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<GradientDescentDpServerMessage>>(stub->AsyncHandle(&client_context4, client_message, &cq));
		rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
		
		client_message = GradientDescentDpClientMessage();
		std::tie(sigmoid_round_4_state,
		 		*client_message.mutable_client_sigmoid_round_4_message()) =
			secure_sigmoid->GenerateSigmoidRoundFourMessage(0, 
					sigmoid_precomputed_value,
	        sigmoid_round_3_state, server_message.server_sigmoid_round_3_message()).value();
		client_end = std::chrono::high_resolution_clock::now();
		pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
	            client_end - client_start).count())/ 1e6;

		server_start = std::chrono::high_resolution_clock::now();
		GPR_ASSERT(cq.Next(&got_tag, &ok));
		GPR_ASSERT(got_tag == (void*) cq_index);
		GPR_ASSERT(ok);
		if (!grpc_status.ok()) {
			std::cerr << "Client: Failed on message round 5 iteration " << i <<
				" with status " <<
				grpc_status.error_code() << " error_message: " <<
			  grpc_status.error_message() << std::endl;
			return 1;
		}
		server_end = std::chrono::high_resolution_clock::now();
		pone_time_incl_comm +=
			(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
				server_start).count())/ 1e6;
		client_start = std::chrono::high_resolution_clock::now();
		
		::grpc::ClientContext client_context5;
		cq_index++;
		ok=false;
		 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<GradientDescentDpServerMessage>>(stub->AsyncHandle(&client_context5, client_message, &cq));
		rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
		
		std::vector<uint64_t> raw_sigmoid_output_shares = 
			secure_sigmoid->GenerateSigmoidResult(0,
		     sigmoid_precomputed_value,
		     sigmoid_round_4_state,
				 server_message.
				 server_sigmoid_round_4_message())
																						.value();
		sigmoid_output = {
			.sigmoid_output = std::move(raw_sigmoid_output_shares)
		};
		client_end = std::chrono::high_resolution_clock::now();
		pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
	            client_end - client_start).count())/ 1e6;
	} else {
	 //////////////////////////////////////////////////////////////////////////
	 //	 6-round Sigmoid with New MIC gate
	 //////////////////////////////////////////////////////////////////////////
	 	applications::SigmoidPrecomputedValueNewMic&
			 sigmoid_precomputed_value = 
					gradient_descent_party_zero->share_provider_->
							GetSigmoidPrecomputedValueNewMic(); 
  	 applications::RoundOneSigmoidNewMicState sigmoid_round_1_state;
  	 applications::RoundTwoSigmoidNewMicState sigmoid_round_2_state;
  	 applications::RoundThreeSigmoidNewMicState sigmoid_round_3_state;
		 applications::RoundThreePointFiveSigmoidNewMicState
			  sigmoid_round_3_point_5_state;
  	 applications::RoundFourSigmoidNewMicState sigmoid_round_4_state;
		 applications::RoundFiveSigmoidNewMicState sigmoid_round_5_state;


		// Run Sigmoid Round 1
		client_message = GradientDescentDpClientMessage();
    std::tie(sigmoid_round_1_state,
		 *client_message.mutable_client_sigmoid_new_mic_round_1_message()) = 
     secure_sigmoid_new_mic->GenerateSigmoidRoundOneMessage(0,
        sigmoid_precomputed_value,
				sigmoid_input.sigmoid_input).value();
		client_end = std::chrono::high_resolution_clock::now();
		pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
		        client_end - client_start).count())/ 1e6;

		server_start = std::chrono::high_resolution_clock::now();
		GPR_ASSERT(cq.Next(&got_tag, &ok));
		GPR_ASSERT(got_tag == (void*) cq_index);
		GPR_ASSERT(ok);
		if (!grpc_status.ok()) {
		std::cerr << "Client: Failed on message round 2 iteration " << i <<
			" with status " <<
			grpc_status.error_code() << " error_message: " <<
		  grpc_status.error_message() << std::endl;
		return 1;
		}
		server_end = std::chrono::high_resolution_clock::now();
		pone_time_incl_comm +=
		(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;

	// Run Sigmoid Round 2
	client_start = std::chrono::high_resolution_clock::now();

	::grpc::ClientContext client_context2;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<GradientDescentDpServerMessage>>(stub->AsyncHandle(&client_context2, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
	client_message = GradientDescentDpClientMessage();
	std::tie(sigmoid_round_2_state,
	 		*client_message.mutable_client_sigmoid_new_mic_round_2_message()) =
		secure_sigmoid_new_mic->GenerateSigmoidRoundTwoMessage(0,
			 sigmoid_precomputed_value,
       sigmoid_round_1_state,
			 server_message.server_sigmoid_new_mic_round_1_message()).value();
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            client_end - client_start).count())/ 1e6;

	server_start = std::chrono::high_resolution_clock::now();
	GPR_ASSERT(cq.Next(&got_tag, &ok));
	GPR_ASSERT(got_tag == (void*) cq_index);
	GPR_ASSERT(ok);
	if (!grpc_status.ok()) {
		std::cerr << "Client: Failed on message round 3 iteration " << i <<
			" with status " <<
			grpc_status.error_code() << " error_message: " <<
		  grpc_status.error_message() << std::endl;
		return 1;
	}
	server_end = std::chrono::high_resolution_clock::now();
	pone_time_incl_comm +=
		(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;

	// Run Sigmoid Round 3
	client_start = std::chrono::high_resolution_clock::now();
	
	::grpc::ClientContext client_context3;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<GradientDescentDpServerMessage>>(stub->AsyncHandle(&client_context3, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
	client_message = GradientDescentDpClientMessage();
	std::tie(sigmoid_round_3_state,
	 		*client_message.mutable_client_sigmoid_new_mic_round_3_message()) =
		secure_sigmoid_new_mic->GenerateSigmoidRoundThreeMessage(0, 
				sigmoid_precomputed_value,
        sigmoid_round_2_state,
				server_message.server_sigmoid_new_mic_round_2_message()).value();
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            client_end - client_start).count())/ 1e6;

	server_start = std::chrono::high_resolution_clock::now();
	GPR_ASSERT(cq.Next(&got_tag, &ok));
	GPR_ASSERT(got_tag == (void*) cq_index);
	GPR_ASSERT(ok);
	if (!grpc_status.ok()) {
		std::cerr << "Client: Failed on message round 4 iteration " << i <<
			" with status " <<
			grpc_status.error_code() << " error_message: " <<
		  grpc_status.error_message() << std::endl;
		return 1;
	}
	server_end = std::chrono::high_resolution_clock::now();
	pone_time_incl_comm +=
		(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;

	// Run Sigmoid Round 3.5
	client_start = std::chrono::high_resolution_clock::now();
	
	::grpc::ClientContext client_context4;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<GradientDescentDpServerMessage>>(stub->AsyncHandle(&client_context4, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
	client_message = GradientDescentDpClientMessage();
	std::tie(sigmoid_round_3_point_5_state,
	 		*client_message.mutable_client_sigmoid_new_mic_round_3_point_5_message()) =
		secure_sigmoid_new_mic->GenerateSigmoidRoundThreePointFiveMessage(0, 
				sigmoid_precomputed_value,
        sigmoid_round_3_state, server_message.server_sigmoid_new_mic_round_3_message()).value();
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            client_end - client_start).count())/ 1e6;

	server_start = std::chrono::high_resolution_clock::now();
	GPR_ASSERT(cq.Next(&got_tag, &ok));
	GPR_ASSERT(got_tag == (void*) cq_index);
	GPR_ASSERT(ok);
	if (!grpc_status.ok()) {
		std::cerr << "Client: Failed on message round 3.5 iteration " << i <<
			" with status " <<
			grpc_status.error_code() << " error_message: " <<
		  grpc_status.error_message() << std::endl;
		return 1;
	}
	server_end = std::chrono::high_resolution_clock::now();
	pone_time_incl_comm +=
		(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;
			 
 	// Run Sigmoid Round 4
 	client_start = std::chrono::high_resolution_clock::now();

	::grpc::ClientContext client_context3_point_5;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<GradientDescentDpServerMessage>>(stub->AsyncHandle(&client_context3_point_5, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
 	client_message = GradientDescentDpClientMessage();
 	std::tie(sigmoid_round_4_state,
 	 		*client_message.mutable_client_sigmoid_new_mic_round_4_message()) =
 		secure_sigmoid_new_mic->GenerateSigmoidRoundFourMessage(0, 
 				sigmoid_precomputed_value,
         sigmoid_round_3_point_5_state, server_message.server_sigmoid_new_mic_round_3_point_5_message()).value();
 	client_end = std::chrono::high_resolution_clock::now();
 	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
             client_end - client_start).count())/ 1e6;

 	server_start = std::chrono::high_resolution_clock::now();
	GPR_ASSERT(cq.Next(&got_tag, &ok));
	GPR_ASSERT(got_tag == (void*) cq_index);
	GPR_ASSERT(ok);
 	if (!grpc_status.ok()) {
 		std::cerr << "Client: Failed on message round 5 iteration " << i <<
 			" with status " <<
 			grpc_status.error_code() << " error_message: " <<
 		  grpc_status.error_message() << std::endl;
 		return 1;
 	}
 	server_end = std::chrono::high_resolution_clock::now();
 	pone_time_incl_comm +=
 		(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
 			server_start).count())/ 1e6;


	// Run Sigmoid Round 5
	client_start = std::chrono::high_resolution_clock::now();
	
 	::grpc::ClientContext client_context5;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<GradientDescentDpServerMessage>>(stub->AsyncHandle(&client_context5, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
	client_message = GradientDescentDpClientMessage();
	std::tie(sigmoid_round_5_state,
	 		*client_message.mutable_client_sigmoid_new_mic_round_5_message()) =
		secure_sigmoid_new_mic->GenerateSigmoidRoundFiveMessage(0, 
				sigmoid_precomputed_value,
        sigmoid_round_4_state, 
				server_message.server_sigmoid_new_mic_round_4_message()).value();
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            client_end - client_start).count())/ 1e6;

	server_start = std::chrono::high_resolution_clock::now();
	GPR_ASSERT(cq.Next(&got_tag, &ok));
	GPR_ASSERT(got_tag == (void*) cq_index);
	GPR_ASSERT(ok);
	if (!grpc_status.ok()) {
		std::cerr << "Client: Failed on message sigmoidd round 5 iteration " << i <<
			" with status " <<
			grpc_status.error_code() << " error_message: " <<
		  grpc_status.error_message() << std::endl;
		return 1;
	}
	server_end = std::chrono::high_resolution_clock::now();
	pone_time_incl_comm +=
		(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;
	client_start = std::chrono::high_resolution_clock::now();
	

	::grpc::ClientContext client_context6;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<GradientDescentDpServerMessage>>(stub->AsyncHandle(&client_context6, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
	std::vector<uint64_t> raw_sigmoid_output_shares = 
		secure_sigmoid_new_mic->GenerateSigmoidResult(0,
	     sigmoid_precomputed_value,
	     sigmoid_round_5_state,
			 server_message.
			 server_sigmoid_new_mic_round_5_message())
																					.value();
	sigmoid_output = {
		.sigmoid_output = std::move(raw_sigmoid_output_shares)
	};
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            client_end - client_start).count())/ 1e6;
		
	}
	//////////////////////////////////////////////////////////////////////////////
	// Last round of gradient descent.
	//////////////////////////////////////////////////////////////////////////////

	// Run round 5 ("X Transpose D")
	client_start = std::chrono::high_resolution_clock::now();
	client_message = GradientDescentDpClientMessage();
	std::tie(state_x_transpose_d,
	 *client_message.mutable_client_x_transpose_d_message()) =
	  gradient_descent_party_zero->GenerateXTransposeDMessage(
	      sigmoid_output, state_masked_x_transpose)
		.value();
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            client_end - client_start).count())/ 1e6;

	server_start = std::chrono::high_resolution_clock::now();
	GPR_ASSERT(cq.Next(&got_tag, &ok));
	GPR_ASSERT(got_tag == (void*) cq_index);
	GPR_ASSERT(ok);
	
	if (!grpc_status.ok()) {
		std::cerr << "Client: Failed on message round 5 iteration " << i <<
			" with status " <<
			grpc_status.error_code() << " error_message: " <<
		  grpc_status.error_message() << std::endl;
		return 1;
	}
	server_end = std::chrono::high_resolution_clock::now();
	pone_time_incl_comm +=
		(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;
	
	
	// Run round 6 ("Reconstruct Gradient")
	client_start = std::chrono::high_resolution_clock::now();
	
	::grpc::ClientContext client_context5;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<GradientDescentDpServerMessage>>(stub->AsyncHandle(&client_context5, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
	client_message = GradientDescentDpClientMessage();
	std::tie(state_reconstruct_gradient, *client_message.mutable_client_reconstruct_gradient_message()) =
	  gradient_descent_party_zero->GenerateReconstructGradientMessage(
	      state_x_transpose_d, server_message.server_x_transpose_d_message(),
		 		server_masked_x_transpose_message)
		.value();
	client_end = std::chrono::high_resolution_clock::now();
	pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
            client_end - client_start).count())/ 1e6;

	server_start = std::chrono::high_resolution_clock::now();
	GPR_ASSERT(cq.Next(&got_tag, &ok));
	GPR_ASSERT(got_tag == (void*) cq_index);
	GPR_ASSERT(ok);
	if (!grpc_status.ok()) {
		std::cerr << "Client: Failed on message round 6 iteration " << i <<
			" with status " <<
			grpc_status.error_code() << " error_message: " <<
		  grpc_status.error_message() << std::endl;
		return 1;
	}
	server_end = std::chrono::high_resolution_clock::now();
	pone_time_incl_comm +=
		(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
			server_start).count())/ 1e6;
	

	// Compute gradient update
	client_start = std::chrono::high_resolution_clock::now();
	

	::grpc::ClientContext client_context6;
	cq_index++;
	ok=false;
	 rpc = std::unique_ptr<grpc::ClientAsyncResponseReader<GradientDescentDpServerMessage>>(stub->AsyncHandle(&client_context6, client_message, &cq));
	rpc->Finish(&server_message, &grpc_status, (void*)cq_index);
	
    Status status = gradient_descent_party_zero->ComputeGradientUpdate(
        state_reconstruct_gradient,
				server_message.server_reconstruct_gradient_message());
    if (!status.ok()) {
		std::cerr << "Client: Failed on compute gradient update with status " <<
			status << std::endl;
      return 1;
    }

		client_end = std::chrono::high_resolution_clock::now();
		pzero_time += (std::chrono::duration_cast<std::chrono::microseconds>(
	            client_end - client_start).count())/ 1e6;
		
		server_start = std::chrono::high_resolution_clock::now();
		GPR_ASSERT(cq.Next(&got_tag, &ok));
		GPR_ASSERT(got_tag == (void*) cq_index);
		GPR_ASSERT(ok);
		if (!grpc_status.ok()) {
			std::cerr << "Client: Failed on end message iteration " << i <<
				" with status " <<
				grpc_status.error_code() << " error_message: " <<
			  grpc_status.error_message() << std::endl;
			return 1;
		}
		server_end = std::chrono::high_resolution_clock::now();
		pone_time_incl_comm +=
			(std::chrono::duration_cast<std::chrono::microseconds>(server_end -
				server_start).count())/ 1e6;

	}
  auto end = std::chrono::high_resolution_clock::now();

  // Add in preprocessing phase. For the online phase, since the initial round for client and server can be done at the same time
  end_to_end_time = (std::chrono::duration_cast<std::chrono::microseconds>(
          end-start).count())
      / 1e6;


	// Print results
	std::cout << "Completed runs" << std::endl << "num_fractional_bits="
		<< absl::GetFlag(FLAGS_num_fractional_bits) << std::endl
	  << "num_ring_bits=" << absl::GetFlag(FLAGS_num_ring_bits) << std::endl
		<< "num_features=" << absl::GetFlag(FLAGS_num_features) << std::endl
	  << "num_examples=" << absl::GetFlag(FLAGS_num_examples) << std::endl
	  << "num_iterations=" << num_iterations << std::endl
	  << "use_new_mic=" << use_new_mic << std::endl
	  << "Client time total (s) =" << pzero_time <<std::endl
	  << "Server time (incl. comm) total (s) = " << pone_time_incl_comm <<std::endl
	      << "End to End time (excluding preprocessing) total (s) = " << end_to_end_time <<std::endl;

  return 0;
}

}  // namespace
}  // namespace logistic_regression_dp
}  // namespace private_join_and_compute

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  return private_join_and_compute::logistic_regression_dp::ExecuteProtocol();
}
