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
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "include/grpc/grpc_security_constants.h"
#include "include/grpcpp/grpcpp.h"
#include "include/grpcpp/security/server_credentials.h"
#include "include/grpcpp/server_builder.h"
#include "include/grpcpp/server_context.h"
#include "include/grpcpp/support/status.h"
#include "applications/const_round_secure_comparison/const_round_secure_comparison_rpc_impl.h"
#include "applications/const_round_secure_comparison/const_round_secure_comparison.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "absl/strings/string_view.h"
#include "poisson_regression/beaver_triple_utils.h"


ABSL_FLAG(std::string, port, "0.0.0.0:10501", "Port on which to listen");

ABSL_FLAG(size_t, num_inputs, 1000, "num inputs");
ABSL_FLAG(size_t, string_length, 62, "string length");
ABSL_FLAG(size_t, num_pieces, 4, "num pieces");
ABSL_FLAG(size_t, piece_length, 16, "piece_length");




namespace private_join_and_compute {
namespace applications {

int RunServer() {
  std::cout << "Server: starting... " << std::endl;
	
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

  std::pair<SecureComparisonPrecomputedValue, SecureComparisonPrecomputedValue> precomp = seccomp->PerformComparisonPrecomputation().value();

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

  ConstRoundSecureComparisonRpcImpl service(
			std::move(first_bit_share), std::move(comparison_input),
	std::move(precomp.second), std::move(seccomp));

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
  grpc_server->Shutdown();
  grpc_server_thread.join();
  std::cout << "Server completed protocol and shut down." << std::endl;

  return 0;
}

}  // namespace applications
}  // namespace private_join_and_compute

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  return private_join_and_compute::applications::RunServer();
}