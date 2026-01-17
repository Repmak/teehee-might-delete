#pragma once

#include <string>
#include <vector>
#include <optional>
#include <iostream>
#include "TokenizerInterface.h"

namespace nlp::tokenizer {

    class MaxMatch : public TokenizerInterface {
        public:
            explicit MaxMatch(const MaxMatchConfig& config);

            [[nodiscard]] std::vector<Token> tokenize(std::string_view text) const override;

            [[nodiscard]] size_t get_vocab_size() const override { return vocab_list_->size(); }
            [[nodiscard]] const VocabList& get_vocab_list() const { return *vocab_list_; }

        private:
            MaxMatchConfig config_;
            std::unique_ptr<VocabList> vocab_list_;

            // Splits text by whitespace and punctuation.
            [[nodiscard]] static std::vector<std::string_view> split_text(std::string_view text);

            // Encode each word into one or more tokens (using MaxMatch algorithm).
            [[nodiscard]] std::vector<Token> encode_word(std::string_view word) const;

            // Truncation and adding special tokens.
            void post_processing(std::vector<Token>& tokens) const;

            // Normalising user input.
            void clean_text_inplace(std::string& text) const;
            void to_lowercase_inplace(std::string& text) const;
            void strip_accents_inplace(std::string& text) const;
            void handle_chinese_chars_inplace(std::string& text) const;
    };

} // namespace nlp::tokenizer
