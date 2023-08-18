#ifndef TENSORFLOW_LITE_MICRO_MATRIX_MULTI_H_
#define TENSORFLOW_LITE_MICRO_MATRIX_MULTI_H_

#include <cstdint>
#include <limits>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

class MatrixMultiplyOpFloat {
 public:
  static const TFLMRegistration* getRegistration();
  static TFLMRegistration* GetMutableRegistration();
  static void* Init(TfLiteContext* context, const char* buffer, size_t length);
  static void Free(TfLiteContext* context, void* buffer);
  static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node);
  static TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node);

 private:
  static bool freed_;
};

}  // namespace 
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_TEST_HELPER_CUSTOM_OPS_H_