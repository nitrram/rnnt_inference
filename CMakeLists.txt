cmake_minimum_required(VERSION 3.18.1)

# Declares and names the project.
project("onnxinfer")

set(ONNX_LIB_DIR . )

message("current dir: " +  ${CMAKE_CURRENT_SOURCE_DIR})

# set protobufers
set(SPM_PROTO_HDRS sp/pb/sentencepiece.pb.h)
set(SPM_PROTO_SRCS sp/pb/sentencepiece.pb.cc)
set(SPM_MODEL_PROTO_HDRS sp/pb/sentencepiece_model.pb.h)
set(SPM_MODEL_PROTO_SRCS sp/pb/sentencepiece_model.pb.cc)
set(PROTOBUF_LITE_LIBRARY "")
set(PROTOBUF_LITE_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/arena.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/arenastring.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/bytestream.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/coded_stream.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/common.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/extension_set.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/generated_enum_util.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/generated_message_table_driven_lite.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/generated_message_util.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/implicit_weak_message.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/int128.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/io_win32.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/message_lite.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/parse_context.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/repeated_field.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/status.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/statusor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/stringpiece.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/stringprintf.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/structurally_valid.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/strutil.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/time.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/wire_format_lite.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/zero_copy_stream.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/zero_copy_stream_impl.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite/zero_copy_stream_impl_lite.cc)

set(ABSL_FLAGS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/absl/flags/flag.cc)

# set sentencepiece
set(SPM_SRCS
  ${PROTOBUF_LITE_SRCS}
  ${SPM_PROTO_HDRS}
  ${SPM_PROTO_SRCS}
  ${SPM_MODEL_PROTO_HDRS}
  ${SPM_MODEL_PROTO_SRCS}
  ${ABSL_FLAGS_SRCS}
  sp/bpe_model.h
  sp/common.h
  sp/normalizer.h
  sp/freelist.h
  sp/filesystem.h
  sp/init.h
  sp/util.h
  sp/sentencepiece_processor.h
  sp/model_factory.h
  sp/model_interface.h
  sp/error.cc
  sp/bpe_model.cc
  sp/model_factory.cc
  sp/model_interface.cc
  sp/normalizer.cc
  sp/filesystem.cc
  sp/sentencepiece_processor.cc
  sp/util.cc
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party/protobuf-lite)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sp/third_party) #absl
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sp/pb)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ort)


find_package(OpenMP REQUIRED)

set(CXX_SRCS
  wav/wav_buf_simulator.cpp
  wav/wavread.cpp
  common/thr_queue.cpp
  stfft.cpp
  feat.cpp
  beam_search.cpp
  input_processing.cpp
  rnnt_attrs.cpp
  onnx_inference.cpp
)

set(CC_SRCS
  ${SPM_SRCS}
  pulse/pulse_rec.c
)

set_source_files_properties(${CXX_SRCS} PROPERTIES COMPILE_FLAGS "-std=c++17 -fno-permissive")

add_compile_options(
#  -DDEBUG_STFT
#  -DDEBUG_FEAT
#  -DDEBUG_WAV
#  -DDEBUG_BUF
  -DDEBUG_INF
  -pthread #protobuf
  -DHAVE_PTHREAD=1 #protobuf
  -mfma #neon
  -mavx2 #neon
  -Wno-deprecated-declarations
  -O3
  -Wall
  -fPIC
  -Ofast
  -march=znver3
#  -march=native
)

add_executable( # Sets the name of the library.
  onnxinfer
  ${CC_SRCS}
  ${CXX_SRCS}
)

set(ONNXRUNTIME_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libonnxruntime.so)


target_compile_options(onnxinfer PRIVATE "${OpenMP_CXX_FLAGS}")


target_compile_definitions(onnxinfer
  PRIVATE
  ${ORR_DEFINITIONS})

target_link_libraries(
  onnxinfer
  PRIVATE
  ${ONNXRUNTIME_PATH}
  ${OpenMP_CXX_FLAGS}
  pulse-simple
  pulse
)
