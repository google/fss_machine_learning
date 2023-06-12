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
#include "applications/secure_comparison/secure_comparison_rpc_impl.h"
#include "applications/secure_comparison/secure_comparison.h"
#include "poisson_regression/fixed_point_element_factory.h"
#include "poisson_regression/fixed_point_element.h"
#include "poisson_regression/ring_arithmetic_utils.h"
#include "absl/strings/string_view.h"
#include "poisson_regression/beaver_triple_utils.h"


ABSL_FLAG(std::string, port, "0.0.0.0:10501", "Port on which to listen");
					
// FPE Params 
ABSL_FLAG(int, num_fractional_bits, 15, "num fractional bits");
ABSL_FLAG(int, num_ring_bits, 63, "num ring bits");

// Comparison Params
ABSL_FLAG(size_t, num_inputs, 10, "num inputs");
ABSL_FLAG(size_t, block_length, 16, "block length");
ABSL_FLAG(size_t, num_splits, 4, "num splits or blocks");

namespace private_join_and_compute {
namespace secure_comparison {

int RunServer() {
  std::cout << "Server: starting... " << std::endl;
	
  // Setup
	// Purposely incorrect: each party generates precomputation independently.
  auto precomputed_values = 
      internal::ComparisonPrecomputation(
          absl::GetFlag(FLAGS_num_inputs), 
					absl::GetFlag(FLAGS_block_length), 
					absl::GetFlag(FLAGS_num_splits)).value();

  auto short_gates = std::move(std::get<0>(precomputed_values));
  auto precomputed_values_p1 = std::move(std::get<2>(precomputed_values));
	// Input is hardcoded to all 0 for the time being
	std::vector<uint64_t> input(absl::GetFlag(FLAGS_num_inputs), 0);

  SecureComparisonRpcImpl service(
			std::move(input),
			absl::GetFlag(FLAGS_block_length), 
			absl::GetFlag(FLAGS_num_splits),
			std::move(short_gates),
			std::move(precomputed_values_p1));

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

}  // namespace secure_comparison
}  // namespace private_join_and_compute

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  return private_join_and_compute::secure_comparison::RunServer();
}