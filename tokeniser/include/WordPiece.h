#pragma once

#include <string>
#include <vector>
#include <optional>

#include "ITokenizer.h"

namespace nlp::tokenizer {

    class WordPiece : public ITokenizer {
        public:
            // Initialisation.
            WordPiece(const std::string& vocab_list);

            std::vector<Token> encode(std::string_view text) const override;
            size_t get_vocab_size() const override;
            std::optional<SpecialTokenType> identify_special_token(uint32_t id) const override;

        private:
            std::unique_ptr<VocabList> vocab_list_;

            void build_byte_encoder();

            std::vector<std::string> bpe_merge(std::string_view word) const;

            // Splits text by regex (eg: whitespace, punctuation).
            std::vector<std::string_view> pre_tokenize(std::string_view text) const;
    };

} // namespace nlp::tokenizer
