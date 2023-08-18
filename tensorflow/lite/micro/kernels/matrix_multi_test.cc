
#include <algorithm>
#include <iterator>
#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

constexpr float kTestTolerance = 7.41e-03;
constexpr int kNumInputs = 2;
constexpr int kNumOutputs = 1;
constexpr int kInputTensorIndex_0 = 0;
constexpr int kInputTensorIndex_1 = 1;
constexpr int kOutputTensorIndex = 2;

void ExecuteMatrixMultiTest(TfLiteTensor* tensors, int tensors_count) {
  int kInputArrayData[] = {kNumInputs, kInputTensorIndex_0,
                           kInputTensorIndex_1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
  int kOutputArrayData[] = {kNumOutputs, kOutputTensorIndex};
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  const TFLMRegistration registration = tflite::Register_MATRIX_MULTI();
  micro::KernelRunner runner(registration, tensors, tensors_count, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

// template <typename T1, typename T2, typename T3>
// void TestMatrixMulti(int* input_dims_data[kNumInputs], const T1* input_data_0,
//                      const T2* input_data_1, int* expected_dims,
//                      const T3* expected_data, T3* output_data) {
template <typename T>
void TestMatrixMulti(int* input_dims_data[kNumInputs], const T* input_data_0,
                     const T* input_data_1, int* expected_dims,
                     const T* expected_data, T* output_data) {
  TfLiteIntArray* input_dims_0 = IntArrayFromInts(input_dims_data[0]);
  TfLiteIntArray* input_dims_1 = IntArrayFromInts(input_dims_data[1]);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int output_count = ElementCount(*output_dims);

  TfLiteTensor tensors[] = {
    CreateTensor(input_data_0, input_dims_0),
    CreateTensor(input_data_1, input_dims_1),
    CreateTensor(output_data, output_dims),
  };
  constexpr int tensors_count = std::extent<decltype(tensors)>::value;
  ExecuteMatrixMultiTest(tensors, tensors_count);

  // check output data against expected
  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], kTestTolerance);
  }

  // check output dimensions (relocated) against original dimensions
  TF_LITE_MICRO_EXPECT_EQ(output_dims->size,
                          tensors[kOutputTensorIndex].dims->size);
  for (int i = 0; i < output_dims->size; i++) {
    TF_LITE_MICRO_EXPECT_EQ(output_dims->data[i],
                            tensors[kOutputTensorIndex].dims->data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(MatrixMultiOpTestSimpleint) {
  int kInputDims_0[] = {2, 2};
  int kInputDims_1[] = {2, 2};
  int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
  int kOutputDims[] = {2, 2};

  constexpr float kInput_0[] = {1, 2, 3, 4};
  constexpr float kInput_1[] = {5, 6, 7, 8};
  constexpr float kExpect[] = {19, 22, 43, 50};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestMatrixMulti(kInputDims, kInput_0, kInput_1, kOutputDims,
                                    kExpect, output_data);
}

// TF_LITE_MICRO_TEST(EmbeddingLookupOpTestSimpleFloat) {
//   int kInputDims_0[] = {1, 3};
//   int kInputDims_1[] = {3, 8};
//   int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
//   int kOutputDims[] = {1, 8};

//   constexpr float kInput_0[] = {1, 0, 2};
//   constexpr float kInput_1[] = {
//       0.00, 0.01, 0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
//       1.00, 1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
//       2.00, 2.01, 2.02, 2.03, 2.10, 2.11, 2.12, 2.13  // Row 2
//   };
//   constexpr float kExpect[] = {
//       4.00, 4.03, 4.04, 4.06, 4.30, 4.33, 4.36, 4.39
//   };
//   constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
//   float output_data[kOutputCount];

//   tflite::testing::TestMatrixMulti(kInputDims, kInput_0, kInput_1,
//                                        kOutputDims, kExpect, output_data);
// }

// TF_LITE_MICRO_TEST(MatrixMultiOpTestSimplefloat2) {
//   // 定义输入矩阵的维度
//   int kInputDims_0[] = {2, 4};
//   int kInputDims_1[] = {4, 3};
//   int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
//   // 指定输入矩阵和期望输出的值
//   constexpr float kInput_0[] = {
//     1.0, 2.0, 3.0, 4.0,
//     5.0, 6.0, 7.0, 8.0
//   };
//   constexpr float kInput_1[] = {
//     1.0, 2.0, 3.0,
//     4.0, 5.0, 6.0,
//     7.0, 8.0, 9.0,
//     10.0, 11.0, 12.0
//   };
//   constexpr float kExpect[] = {
//     70.0, 80.0, 90.0,
//     158.0, 184.0, 210.0
//   };

//   // 定义输出矩阵的维度和输出数据
//   int kOutputDims[] = {2, 3};
// //   float output_data[6];
//   constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
//   float output_data[kOutputCount];

//   // 调用测试函数进行测试
//   tflite::testing::TestMatrixMulti(kInputDims,kInput_0, kInput_1,
//                                    kOutputDims, kExpect, output_data);
// }


// TF_LITE_MICRO_TEST(EmbeddingLookupOpTestSimpleInt8) {
//   int kInputDims_0[] = {1, 3};
//   int kInputDims_1[] = {3, 3, 2, 4};
//   int* kInputDims[tflite::testing::kNumInputs] = {kInputDims_0, kInputDims_1};
//   int kOutputDims[] = {3, 3, 2, 4};

//   constexpr int8_t kInput_0[] = {1, 0, 2};
//   constexpr int8_t kInput_1[] = {
//       0,   1,   2,   3,   10,  11,  12,  13,   // Row 0
//       100, 101, 102, 103, 110, 111, 112, 113,  // Row 1
//       -56, -55, -54, -53, -46, -45, -44, -43,  // Row 2
//   };
//   constexpr int8_t kExpect[] = {
//       100, 101, 102, 103, 110, 111, 112, 113,  // Row 1
//       0,   1,   2,   3,   10,  11,  12,  13,   // Row 0
//       -56, -55, -54, -53, -46, -45, -44, -43,  // Row 2
//   };
//   constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
//   int8_t output_data[kOutputCount];

//   tflite::testing::TestMatrixMulti(kInputDims, kInput_0, kInput_1,
//                                        kOutputDims, kExpect, output_data);
// }

TF_LITE_MICRO_TESTS_END
