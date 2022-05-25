#pragma once
#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "core/providers/cpu/cpu_execution_provider.h"
namespace onnxruntime {
namespace training {
namespace api {
namespace utils {

// TODO: consolidate the gradient names with frontend tooling
const std::vector<std::string> GRAD_SUFFIX{"_grad.accumulation.buffer", "_grad", "_grad.accumulation.out"};
const std::string MOMENT_1_SUFFIX{".exp_avg"};
const std::string MOMENT_2_SUFFIX{".exp_avg_sq"};
const std::string LAZY_RESET_GRAD_NAME{"lazy_reset_grad"};
// TODO: don't hard code the state names, should get the state names according to the optimizer types.
const std::vector<std::string> MOMENT_STATE_NAMES{"momentum0", "momentum1"};

// Get names of graph inputs and outputs
void GetGraphInputOutputNames(const std::unique_ptr<onnxruntime::InferenceSession>& session_object,
                              std::vector<std::string>& input_names,
                              std::vector<std::string>& output_names);
// Fetch the parameter name from suffix: name = param_name+suffix,
// returns True if suffix is present in name else False
bool GetParamNameFromSuffix(const std::string& name, const std::string& suffix, std::string& param_name);

// Fetch the parameter name from all possible gradient suffix: name = param_name+suffix
// returns True if suffix is present in name else False
bool GetParamNameFromGradient(const std::string& grad_name, std::string& param_name);

// Allocate OrtValue like the input ortvalue on the same device
Status OrtValueLike(const SessionState& sess_state, const OrtValue& input_val, OrtValue& output_val);

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
