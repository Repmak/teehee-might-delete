#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>

namespace nlp::inference {

    class ORTWrapper {
        public:
            ORTWrapper(const std::string& model_path);

            std::vector<float> run(
                const std::vector<int64_t>& input_ids,
                const std::vector<int64_t>& mask,
                const std::vector<int64_t>& type_ids
            );

        private:
            Ort::Env env;
            Ort::Session session;
            Ort::MemoryInfo memory_info;

            std::vector<const char*> input_names = {"input_ids", "attention_mask", "token_type_ids"};
            std::vector<const char*> output_names = {"output"};
    };

} // namespace nlp::inference
