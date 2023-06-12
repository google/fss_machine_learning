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
#include <thread>  // NOLINT

#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "glog/logging.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "include/grpc/grpc_security_constants.h"
#include "include/grpcpp/grpcpp.h"
#include "include/grpcpp/security/server_credentials.h"
#include "include/grpcpp/server_builder.h"
#include "include/grpcpp/server_context.h"
#include "include/grpcpp/support/status.h"
#include "secret_sharing_mpc/gates/correlated_matrix_product.h"
#include "applications/logistic_regression/gradient_descent_dp_rpc_impl.h"
#include "applications/secure_sigmoid_new_mic/secure_sigmoid_new_mic.h"
#include "applications/logistic_regression/gradient_descent_dp.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "absl/strings/string_view.h"
#include "poisson_regression/beaver_triple_utils.h"


ABSL_FLAG(std::string, port, "0.0.0.0:10501", "Port on which to listen");
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

int RunServer() {
  std::cout << "Server: starting... " << std::endl;
  
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
	
  const applications::SecureSigmoidNewMicParameters sigmoid_params_new_mic {
      kLogGroupSize,
      sigmoid_spline_params,
      static_cast<uint8_t>(absl::GetFlag(FLAGS_num_fractional_bits)),
      kTaylorPolynomialDegree,
      kSampleLargeExpParams,
			absl::GetFlag(FLAGS_block_length),
			absl::GetFlag(FLAGS_num_splits)
  };
	
	// For precomputation
  std::unique_ptr<applications::SecureSigmoid> secure_sigmoid = 
		applications::SecureSigmoid::Create(gd_params.num_examples,
			sigmoid_params).value();			
  std::unique_ptr<applications::SecureSigmoidNewMic> secure_sigmoid_new_mic = 
		applications::SecureSigmoidNewMic::Create(gd_params.num_examples,
			sigmoid_params_new_mic).value();
	
	// Purposely incorrect: sample shares of x and y at random.
  std::vector<uint64_t> share_x =
		internal::SampleShareOfZero(
			gd_params.num_examples * gd_params.num_features,
	    gd_params.modulus).value().first;		
  std::vector<uint64_t> share_y = internal::SampleShareOfZero(
			gd_params.num_examples,
	    gd_params.modulus).value().first;
	// Initialize theta to 0.
	std::vector<double> theta(gd_params.num_features, 0);

	// TODO actually make the client communicate the trusted setup.
	// Purposely incorrect: each party generates shares independently.
  // Initialize preprocessed shares in trusted setup.
  std::vector<applications::SigmoidPrecomputedValue> sigmoid_p0;
	std::vector<applications::SigmoidPrecomputedValueNewMic> sigmoid_p0_new_mic;
  std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
      beaver_triple_matrix_b_c_p1;

  // Generates mask for X and for X^T. These masks are part of the correlated
  // matrix product and do not change across iterations
  auto vector_a_masks_x_transpose =
      SampleBeaverTripleMatrixA(gd_params.num_features, gd_params.num_examples,
                                gd_params.modulus).value();
																
	// Initialize DP noise also in trusted setup (set dp noise to 0 for now)
	std::vector<std::vector<uint64_t>> share_noise_p1 (gd_params.num_iterations,
		std::vector<uint64_t>(gd_params.num_features, 0));

  for (size_t idx = 0; idx < gd_params.num_iterations; idx++) {
    // sigmoid(u): u (#examples x 1)
		// Performs precomputations for both MIC options, only the right one will be
		// used.
    std::pair<applications::SigmoidPrecomputedValue, 		applications::SigmoidPrecomputedValue> preCompRes;
    preCompRes = secure_sigmoid->PerformSigmoidPrecomputation().value();
    sigmoid_p0.push_back(std::move(preCompRes.first));
		sigmoid_p0_new_mic.push_back(std::move(
			secure_sigmoid_new_mic->PerformSigmoidPrecomputation().value().second));


    // X.transpose() * d: X.transpose() (#features * #examples),
    // d (#examples x 1).
    auto matrix_shares_3 =
                         SampleBeaverTripleMatrixBandC(
                             vector_a_masks_x_transpose, gd_params.num_features,
                             gd_params.num_examples, 1, 
														 gd_params.modulus).value();
    beaver_triple_matrix_b_c_p1.push_back(matrix_shares_3.first);
  }
  // Initialize LogRegDpShareProvider with preprocessed shares.
	std::unique_ptr<LogRegDPShareProvider> share_provider;
	if (use_new_mic) {
		// Initialize LogRegShareProvider for New MIC.
    share_provider  =
			std::make_unique<LogRegDPShareProvider>(
				LogRegDPShareProvider::CreateNewMic(
          std::move(sigmoid_p0_new_mic),
				  std::get<0>(vector_a_masks_x_transpose),
          beaver_triple_matrix_b_c_p1, share_noise_p1, gd_params.num_examples,
          gd_params.num_features, gd_params.num_iterations).value());
	} else {
		// Initialize LogRegShareProvider with preprocessed shares.
    share_provider  = 
			std::make_unique<LogRegDPShareProvider>(LogRegDPShareProvider::Create(
        sigmoid_p0, std::get<0>(vector_a_masks_x_transpose),
        beaver_triple_matrix_b_c_p1, share_noise_p1, gd_params.num_examples,
        gd_params.num_features, gd_params.num_iterations).value());
	}
	
	std::unique_ptr<GradientDescentDpRpcImpl> service;
	
	if(use_new_mic) {
		service = std::make_unique<GradientDescentDpRpcImpl>(share_x, share_y,
		 theta, std::move(share_provider), fpe_params, gd_params,
		 sigmoid_params_new_mic);
	} else {
		service = std::make_unique<GradientDescentDpRpcImpl>(share_x, share_y,
		 theta, std::move(share_provider), fpe_params, gd_params,
		 sigmoid_params);
	}

  ::grpc::ServerBuilder builder;
  // Consider grpc::SslServerCredentials if not running locally.
  builder.AddListeningPort(absl::GetFlag(FLAGS_port),
                           grpc::InsecureServerCredentials());
	builder.SetMaxReceiveMessageSize(INT_MAX); // consider limiting max message size
  builder.RegisterService(service.get());
  std::unique_ptr<::grpc::Server> grpc_server(builder.BuildAndStart());

  // Run the server on a background thread.
  std::thread grpc_server_thread(
      [](::grpc::Server* grpc_server_ptr) {
        std::cout << "Server: listening on " << absl::GetFlag(FLAGS_port)
                  << std::endl;
        grpc_server_ptr->Wait();
      },
      grpc_server.get());

  while (!service->shut_down()) {
    // Wait for the server to be done, and then shut the server down.
  }

  // Shut down server.
  grpc_server->Shutdown();
  grpc_server_thread.join();
  std::cout << "Server completed protocol and shut down." << std::endl;

  return 0;
}

}  // namespace logistic_regression_dp
}  // namespace private_join_and_compute

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  return private_join_and_compute::logistic_regression_dp::RunServer();
}
