#pragma once

#include <string>
#include <vector>
#include <string_view>
#include <optional>

#include "VocabList.h"

namespace nlp::tokenizer {

    struct Token {
        uint32_t id;          // The numerical ID according to the model.
        std::string text;     // The original string representation.
        size_t start_offset;  // Start position in the original string.
        size_t end_offset;    // End position in the original string.
        std::optional<SpecialTokenType> special_type;
    };

    class ITokenizer {
        public:
            virtual ~ITokenizer() = default;

            // Encodes raw text into a Token object.
            virtual std::vector<Token> encode(std::string_view text) const = 0;

            // Returns the total vocabulary size.
            virtual size_t get_vocab_size() const = 0;

            // Checks if the token linked to the parameter id is a special token.
            virtual std::optional<SpecialTokenType> identify_special_token(uint32_t id) const = 0;
    };

} // namespace nlp::tokenizer
