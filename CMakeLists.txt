cmake_minimum_required(VERSION 4.1)
project(word-piece-tokenizer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(tokeniser/include)

add_executable(word-piece-tokenizer
        tokeniser/include/ByteLevelBPE.h
        tokeniser/include/ITokenizer.h
        tokeniser/include/VocabList.h
        tokeniser/src/ByteLevelBPE.cpp
        main.cpp
        tokeniser/src/WordPiece.cpp
        tokeniser/include/WordPiece.h
        tokeniser/src/VocabList.cpp
)
