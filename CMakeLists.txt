cmake_minimum_required(VERSION 3.20)
project(word-piece-encoder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Search for International Components for Unicode.
find_package(ICU COMPONENTS uc i18n REQUIRED)

# Path setup for onnxruntime library.
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/3rd_party/onnxruntime-osx-arm64-1.23.2")

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

add_executable(${PROJECT_NAME}
        main.cpp

        encoder/src/VocabList.cpp
        encoder/src/WordPiece.cpp
        encoder/include/EncoderInterface.h
        encoder/include/VocabList.h
        encoder/include/WordPiece.h

#        inference/src/ORTWrapper.cpp
#        inference/include/ORTWrapper.h
)

# Tells the compiler where to look for include files.
target_include_directories(${PROJECT_NAME} PRIVATE
        encoder/include
        inference/include
        ${ICU_INCLUDE_DIRS}
        "${ONNXRUNTIME_ROOT}/include"
)

#
target_link_libraries(${PROJECT_NAME} PRIVATE
        ${ONNXRUNTIME_LIB}
        $<$<TARGET_EXISTS:ICU::uc>:ICU::uc>
        $<$<TARGET_EXISTS:ICU::i18n>:ICU::i18n>
        $<$<NOT:$<TARGET_EXISTS:ICU::uc>>:${ICU_LIBRARIES}>  # Fallback for older ICU versions.
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
        PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}"
)
