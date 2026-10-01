cmake_minimum_required(VERSION 3.26)
project(wordpipe_nemo_worker LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(NEMO_SPEECH_SOURCE_DIR "" CACHE PATH "Pinned upstream NeMo-Speech.cpp checkout")
set(NEMO_SPEECH_LIBRARY_DIR "" CACHE PATH "Matching upstream build's bin directory")
if(NOT EXISTS "${NEMO_SPEECH_SOURCE_DIR}/src/asr/recognizer.h")
  message(FATAL_ERROR "Set NEMO_SPEECH_SOURCE_DIR to the matching upstream checkout")
endif()
unset(NEMO_ASR CACHE)
find_library(NEMO_ASR nemo_speech_asr PATHS "${NEMO_SPEECH_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)
find_package(Threads REQUIRED)
add_executable(wordpipe-nemo-worker main.cpp)
target_include_directories(wordpipe-nemo-worker PRIVATE
  "${NEMO_SPEECH_SOURCE_DIR}/src/common"
  "${NEMO_SPEECH_SOURCE_DIR}/src/asr"
  "${NEMO_SPEECH_SOURCE_DIR}/src/asr/features"
  "${NEMO_SPEECH_SOURCE_DIR}/src/asr/encoder"
  "${NEMO_SPEECH_SOURCE_DIR}/src/asr/decoders"
  "${NEMO_SPEECH_SOURCE_DIR}/src/asr/vad"
  "${NEMO_SPEECH_SOURCE_DIR}/src/asr/diar"
  "${NEMO_SPEECH_SOURCE_DIR}/src/asr/postproc"
  "${NEMO_SPEECH_SOURCE_DIR}/src/runtime/ggml"
  "${NEMO_SPEECH_SOURCE_DIR}/ggml/include"
  "${NEMO_SPEECH_SOURCE_DIR}/llama.cpp/vendor"
  "${NEMO_SPEECH_SOURCE_DIR}/llama.cpp/vendor/miniaudio")
target_link_libraries(wordpipe-nemo-worker PRIVATE "${NEMO_ASR}" Threads::Threads ${CMAKE_DL_LIBS})
target_compile_options(wordpipe-nemo-worker PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(wordpipe-nemo-worker PROPERTIES INSTALL_RPATH "$ORIGIN/../lib/nemo" BUILD_RPATH_USE_ORIGIN ON)
install(TARGETS wordpipe-nemo-worker RUNTIME DESTINATION bin)
enable_testing()
add_executable(test-transcript test_transcript.cpp)
add_test(NAME transcript-snapshots COMMAND test-transcript)
add_test(NAME worker-help COMMAND wordpipe-nemo-worker --help)
add_executable(test-session-thread test_session_thread.cpp)
target_link_libraries(test-session-thread PRIVATE Threads::Threads)
add_test(NAME session-lifecycle COMMAND test-session-thread)
