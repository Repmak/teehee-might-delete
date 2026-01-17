#pragma once
#include <vector>
#include <TokenizerInterface.h>

namespace sentencpp::embedding_utils {

    class VectorMaths {
        public:
            //
            static std::vector<float> mean_pooling(
                const std::vector<std::vector<float>>& token_embeddings,
                const std::vector<tokenizer::Token>& original_tokens
            );

            //
            static std::vector<float> min_pooling(
                const std::vector<std::vector<float>>& token_embeddings,
                const std::vector<tokenizer::Token>& original_tokens
                );

            //
            static std::vector<float> max_pooling(
                const std::vector<std::vector<float>>& token_embeddings,
                const std::vector<tokenizer::Token>& original_tokens
            );

            // Calculates euclidean distance between two vectors.
            static float euclidean_distance(
                const std::vector<float>& a,
                const std::vector<float>& b
            );

            // Calculates the cosine similarity between two vectors.
            static float cosine_similarity(
                const std::vector<float>& vec_a,
                const std::vector<float>& vec_b
            );

            // Calculates softmax for a
            static std::vector<float> calculate_softmax(
                const std::vector<float>& logits
            );
    };

} // namespace sentencpp::embedding_utils
