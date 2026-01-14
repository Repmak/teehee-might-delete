#include <iostream>
#include <iomanip>
#include <optional>
#include "./encoder/include/WordPiece.h"
#include "./inference/include/ORTWrapper.h"

int main() {
    try {
        nlp::encoder::WordPiece encoder(
            std::string(PROJECT_ROOT_PATH) + "/hf_model/tokenizer.json",
            "/model/vocab",
            true,
            true,
            true,
            true,
            128
        );

        const auto& vocab = encoder.get_vocab_list();
        std::unordered_map<std::string, int64_t> string_map = vocab.get_string_to_id_map();

        // std::cout << std::left << std::setw(20) << "Token" << " | " << "ID" << std::endl;
        // std::cout << std::string(30, '-') << std::endl;
        // for (const auto& [token, id] : string_map) {
        //     std::cout << std::left << std::setw(20) << token << " | " << id << std::endl;
        // }

        auto tokens = encoder.encode("Thé quick Browñ fox   jumps over \n the lázy dog");

        // std::cout << "\n--- Tokenization Results (" << tokens.size() << " tokens) ---\n";
        // std::cout << std::left
        //           << std::setw(6)  << "Index"
        //           << std::setw(10) << "ID"
        //           << std::setw(18) << "Token"
        //           << "Role" << "\n";
        // std::cout << std::string(45, '-') << "\n";
        // for (size_t i = 0; i < tokens.size(); ++i) {
        //     std::cout << std::left
        //               << std::setw(6)  << i
        //               << std::setw(10) << tokens[i].id
        //               << std::setw(18) << tokens[i].text << "\n";
        // }

        // nlp::inference::OnnxModel model(std::string(PROJECT_ROOT_PATH) + "/hf_model/model.onnx");
        // std::vector<float> results = model.run(ids, mask, types);
        //
        // for (float val : results) {
        //     std::cout << val << " ";
        // }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
