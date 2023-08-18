#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

constexpr int kInputTensor_0 = 0;
constexpr int kInputTensor_1 = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  float scale;                // quantization scale for tensor 1
  size_t num_rows_1;          // number of rows in tensor 1
  size_t num_columns_1;       // number of columns in tensor 1
  size_t num_columns_2;       // number of columns in tensor 2
};

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteTensor* tensor_1,
                             const TfLiteTensor* tensor_2,
                             const TfLiteTensor* output) {
  node->user_data = context->AllocatePersistentBuffer(context, sizeof(OpData));
  OpData* op_data = static_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, op_data != nullptr);

  if (tensor_1->type == kTfLiteInt8 && tensor_2->type == kTfLiteFloat32) {
    TF_LITE_ENSURE_EQ(context, tensor_1->params.zero_point, 0);
    op_data->scale = tensor_1->params.scale;
  }

  op_data->num_rows_1 = tensor_1->dims->data[0];
  op_data->num_columns_1 = tensor_1->dims->data[1];
  op_data->num_columns_2 = tensor_2->dims->data[1];

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input_1 =
      micro_context->AllocateTempInputTensor(node, kInputTensor_0);
  TF_LITE_ENSURE(context, input_1 != nullptr);
  TF_LITE_ENSURE(context, input_1->type == kTfLiteFloat32 || input_1->type == kTfLiteInt8);

  TfLiteTensor* input_2 =
      micro_context->AllocateTempInputTensor(node, kInputTensor_1);
  TF_LITE_ENSURE(context, input_2 != nullptr);
  TF_LITE_ENSURE(context, input_2->type == kTfLiteFloat32 || input_2->type == kTfLiteInt8);

  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE(context, output->type == kTfLiteFloat32 || input_2->type == kTfLiteInt8);

  TF_LITE_ENSURE_OK(context, CalculateOpData(context, node, input_1, input_2, output));

  micro_context->DeallocateTempTfLiteTensor(input_1);
  micro_context->DeallocateTempTfLiteTensor(input_2);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input_1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor_0);
  const TfLiteEvalTensor* input_2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor_1);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  OpData& op_data = *static_cast<OpData*>(node->user_data);

  const size_t num_rows_1 = op_data.num_rows_1;
  const size_t num_columns_1 = op_data.num_columns_1;
  const size_t num_columns_2 = op_data.num_columns_2;

  if (input_1->type == kTfLiteFloat32 && input_2->type == kTfLiteFloat32) {
    const float* input_1_data = tflite::micro::GetTensorData<float>(input_1);
    const float* input_2_data = tflite::micro::GetTensorData<float>(input_2);
    float* output_data = tflite::micro::GetTensorData<float>(output);

    // Perform matrix multiplication
    for (size_t i = 0; i < num_rows_1; i++) {
      for (size_t j = 0; j < num_columns_2; j++) {
        float sum = 0.0;
        for (size_t k = 0; k < num_columns_1; k++) {
          sum += input_1_data[i * num_columns_1 + k] * input_2_data[k * num_columns_2 + j];
        }
        output_data[i * num_columns_2 + j] = sum;
      }
    }
  } else if (input_1->type == kTfLiteInt8 && input_2->type == kTfLiteFloat32) {
    const int8_t* input_1_data = tflite::micro::GetTensorData<int8_t>(input_1);
    const float* input_2_data = tflite::micro::GetTensorData<float>(input_2);
    float* output_data = tflite::micro::GetTensorData<float>(output);

    // Perform matrix multiplication with dequantization
    for (size_t i = 0; i < num_rows_1; i++) {
      for (size_t j = 0; j < num_columns_2; j++) {
        float sum = 0.0;
        for (size_t k = 0; k < num_columns_1; k++) {
          sum += static_cast<float>(input_1_data[i * num_columns_1 + k]) *
                 input_2_data[k * num_columns_2 + j];
        }
        output_data[i * num_columns_2 + j] = sum;
      }
    }
  } else if (input_1->type == kTfLiteInt8 && input_2->type == kTfLiteInt8) {
    const int8_t* input_1_data = tflite::micro::GetTensorData<int8_t>(input_1);
    const int8_t* input_2_data = tflite::micro::GetTensorData<int8_t>(input_2);
    int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

    // Perform matrix multiplication with dequantization
    for (size_t i = 0; i < num_rows_1; i++) {
      for (size_t j = 0; j < num_columns_2; j++) {
        int sum = 0;
        for (size_t k = 0; k < num_columns_1; k++) {
          sum += static_cast<int8_t>(input_1_data[i * num_columns_1 + k]) *
                 input_2_data[k * num_columns_2 + j];
        }
        output_data[i * num_columns_2 + j] = sum;
      }
    }
  } else {
    MicroPrintf("MATRIX_MULTI only supports FLOAT32 and INT8 inputs, got %s and %s.",
                TfLiteTypeGetName(input_1->type), TfLiteTypeGetName(input_2->type));
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace 

TFLMRegistration Register_MATRIX_MULTI() {
  return tflite::micro::RegisterOp(nullptr, Prepare, Eval);
}

}  // namespace tflite
