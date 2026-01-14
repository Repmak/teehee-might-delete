#include <iostream>
#include "ORTWrapper.h"

namespace nlp::inference {

    ORTWrapper::ORTWrapper(const std::string& model_path) :
        env(ORT_LOGGING_LEVEL_WARNING, "BERT_Inference"),
        session(env, std::wstring(model_path.begin(), model_path.end()).c_str(), Ort::SessionOptions{nullptr}),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {

    }

    std::vector<float> ORTWrapper::run(
        const std::vector<int64_t>& input_ids,
        const std::vector<int64_t>& mask,
        const std::vector<int64_t>& type_ids
    ) {
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(input_ids.data()), input_ids.size(), input_shape.data(), input_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(mask.data()), mask.size(), input_shape.data(), input_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(type_ids.data()), type_ids.size(), input_shape.data(), input_shape.size()));

        // todo: code below is llm generated!!
        // auto output_tensors = session.Run(
        //     Ort::RunOptions{nullptr},
        //     input_node_names.data(),
        //     input_tensors.data(),
        //     input_tensors.size(),
        //     output_node_names.data(),
        //     output_node_names.size()
        // );
        //
        // // 4. Extract the output data
        // float* float_ptr = output_tensors[0].GetTensorMutableData<float>();
        // size_t output_count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        //
        // return std::vector<float>(float_ptr, float_ptr + output_count);
    }

} // namespace nlp::inference
