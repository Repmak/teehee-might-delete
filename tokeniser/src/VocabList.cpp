#include "VocabList.h"

namespace nlp::tokenizer {

    bool VocabList::set_token(const std::string& token, uint32_t id, std::optional<SpecialTokenType> type) {
        // Check the token and id.
        if (
            string_to_id_map_.contains(token) ||
            (id < id_to_string_map_.size() && !id_to_string_map_[id].empty()))
        {
            return false;
        }

        // Ensure vector is large enough.
        if (id >= id_to_string_map_.size()) id_to_string_map_.resize(id + 1);

        // Set the mappings.
        string_to_id_map_[token] = id;
        id_to_string_map_[id] = token;

        // These will be populated by ids 0, 100, 101, 102, and 103.
        // Refer to the vocab dictionary from tokenizer.json.
        if (type.has_value()) special_tokens_[type.value()] = id;

        return true;
    }

    std::optional<uint32_t> VocabList::token_to_id(const std::string& token) const {
        auto got = string_to_id_map_.find(token);
        if (got == string_to_id_map_.end()) return std::nullopt;
        return got->second;
    }

    std::optional<std::string> VocabList::id_to_token(uint32_t id) const {
        if (id >= id_to_string_map_.size()) return std::nullopt;
        const std::string& token = id_to_string_map_[id];
        if (token.empty()) return std::nullopt;
        return token;
    }

} // namespace nlp::tokenizer
