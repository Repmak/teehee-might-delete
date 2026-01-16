#pragma once

#include <string>
#include <vector>
#include <TokenizerInterface.h>
#include "InferenceInterface.h"


namespace nlp::inference {

    class OnnxEngine : public InferenceInterface {
        public:
            OnnxEngine(const std::string& model_path);

            [[nodiscard]] std::vector<std::vector<float>> encode(const std::vector<tokenizer::Token>& tokens) override;

        private:
            Ort::Env env;
            Ort::Session session;
            Ort::MemoryInfo memory_info;

            std::vector<std::string> input_names;
            std::vector<std::string> output_names;
    };

} // namespace nlp::inference
