#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace nlp::tokenizer {

    enum class SpecialTokenType {
        BOS, EOS, PAD, UNK, MASK
    };

    class VocabList {
        public:
            VocabList() = default;

            // Explicitly set an ID to a token.
            bool set_token(const std::string& token, uint32_t id, std::optional<SpecialTokenType> type = std::nullopt);

            [[nodiscard]] std::optional<uint32_t> token_to_id(const std::string& token) const;
            [[nodiscard]] std::optional<std::string> id_to_token(uint32_t id) const;
            [[nodiscard]] size_t size() const { return id_to_string_map_.size(); }

        private:
            std::unordered_map<std::string, uint32_t> string_to_id_map_;
            std::vector<std::string> id_to_string_map_;
            std::unordered_map<SpecialTokenType, uint32_t> special_tokens_;
    };

} // namespace nlp::tokenizer
