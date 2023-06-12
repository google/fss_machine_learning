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
#include "applications/secure_sigmoid/secure_sigmoid_rpc_impl.h"
#include "applications/secure_sigmoid/secure_sigmoid.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "absl/strings/string_view.h"
#include "poisson_regression/beaver_triple_utils.h"


ABSL_FLAG(std::string, port, "0.0.0.0:10501", "Port on which to listen");
					
// FPE Params 
ABSL_FLAG(int, num_fractional_bits, 20, "num fractional bits");
ABSL_FLAG(int, num_ring_bits, 63, "num ring bits");

// Exponent Params
ABSL_FLAG(int, exponent_bound, 13, "exponent bound");

// Gradient Descent Params
ABSL_FLAG(size_t, num_examples, 10, "num examples");

namespace private_join_and_compute {
namespace applications {

int RunServer() {
  std::cout << "Server: starting... " << std::endl;
	
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

  const applications::SecureSigmoidParameters sigmoid_params {
      kLogGroupSize,
      sigmoid_spline_params,
      static_cast<uint8_t>(absl::GetFlag(FLAGS_num_fractional_bits)),
      kTaylorPolynomialDegree,
      kSampleLargeExpParams
  };
	
	// For precomputation
  std::unique_ptr<applications::SecureSigmoid> secure_sigmoid = 
		applications::SecureSigmoid::Create(
			absl::GetFlag(FLAGS_num_examples),
			sigmoid_params).value();
	
	// Purposely incorrect: sample shares of x, y and theta at random.
  std::vector<uint64_t> sigmoid_input_share =
		internal::SampleShareOfZero(
			absl::GetFlag(FLAGS_num_examples),
	    modulus).value().second;		

    // sigmoid(u): u (#examples x 1)
    std::pair<applications::SigmoidPrecomputedValue, 		applications::SigmoidPrecomputedValue> preCompRes;
    preCompRes = secure_sigmoid->PerformSigmoidPrecomputation().value();

	

  SecureSigmoidRpcImpl service(std::move(sigmoid_input_share),
		std::move(preCompRes.second), sigmoid_params);

  ::grpc::ServerBuilder builder;
  // Consider grpc::SslServerCredentials if not running locally.
  builder.AddListeningPort(absl::GetFlag(FLAGS_port),
                           grpc::InsecureServerCredentials());
	builder.SetMaxReceiveMessageSize(INT_MAX); // consider limiting max message size
  builder.RegisterService(&service);
  std::unique_ptr<::grpc::Server> grpc_server(builder.BuildAndStart());

  // Run the server on a background thread.
  std::thread grpc_server_thread(
      [](::grpc::Server* grpc_server_ptr) {
        std::cout << "Server: listening on " << absl::GetFlag(FLAGS_port)
                  << std::endl;
        grpc_server_ptr->Wait();
      },
      grpc_server.get());

  while (!service.shut_down()) {
    // Wait for the server to be done, and then shut the server down.
  }

  // Shut down server.
  //grpc_server->Shutdown();
  grpc_server_thread.join();
  std::cout << "Server completed protocol and shut down." << std::endl;

  return 0;
}

}  // namespace applications
}  // namespace private_join_and_compute

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  return private_join_and_compute::applications::RunServer();
}
