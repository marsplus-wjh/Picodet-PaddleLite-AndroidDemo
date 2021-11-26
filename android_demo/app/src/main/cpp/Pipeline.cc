// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include "Pipeline.h"

Detector::Detector(const std::string &modelDir, const std::string &labelPath,
                   const int cpuThreadNum, const std::string &cpuPowerMode,
                   int inputWidth, int inputHeight,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd, float scoreThreshold,
                   const std::string &configPath, float nmsThreshold, const std::vector<int> &inputFpnStride)
        : inputWidth_(inputWidth), inputHeight_(inputHeight), inputMean_(inputMean),
          inputStd_(inputStd), scoreThreshold_(scoreThreshold), nmsThreshold_(nmsThreshold), inputFpnStride_(inputFpnStride) {
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(modelDir + "/model.nb");
  config.set_threads(cpuThreadNum);
  config.set_power_mode(ParsePowerMode(cpuPowerMode));
  predictor_ =
          paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
                  config);
  labelList_ = LoadLabelList(labelPath);
  colorMap_ = GenerateColorMap(labelList_.size());
  configs_ = LoadConfigs(configPath);
}

std::vector<std::string> Detector::LoadLabelList(const std::string &labelPath) {
  std::ifstream file;
  std::vector<std::string> labels;
  file.open(labelPath);
  while (file) {
    std::string line;
    std::getline(file, line);
    labels.push_back(line);
  }
  file.clear();
  file.close();
  return labels;
}

std::vector<std::string> Detector::LoadConfigs(const std::string &configPath) {
  std::ifstream file;
  std::vector<std::string> configs;
  file.open(configPath);
  while (file) {
    std::string line;
    std::getline(file, line);
    configs.push_back(line);
  }
  file.clear();
  file.close();
  return configs;
}

std::vector<cv::Scalar> Detector::GenerateColorMap(int numOfClasses) {
  std::vector<cv::Scalar> colorMap = std::vector<cv::Scalar>(numOfClasses);
  for (int i = 0; i < numOfClasses; i++) {
    int j = 0;
    int label = i;
    int R = 0, G = 0, B = 0;
    while (label) {
      R |= (((label >> 0) & 1) << (7 - j));
      G |= (((label >> 1) & 1) << (7 - j));
      B |= (((label >> 2) & 1) << (7 - j));
      j++;
      label >>= 3;
    }
    colorMap[i] = cv::Scalar(R, G, B);
  }
  return colorMap;
}

void Detector::Preprocess(const cv::Mat &rgbaImage) {
  // Set the data of input image
  auto inputTensor = predictor_->GetInput(0);
  std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_}; //inputHeight_ = im_shape_[0] inputWidth_ = im_shape_[1]
  inputTensor->Resize(inputShape);
  auto inputData = inputTensor->mutable_data<float>();
  cv::Mat resizedRGBAImage;
  cv::resize(rgbaImage, resizedRGBAImage,
             cv::Size(inputShape[3], inputShape[2]));
  cv::Mat resizedRGBImage;
  cv::cvtColor(resizedRGBAImage, resizedRGBImage, cv::COLOR_BGRA2RGB);
  resizedRGBImage.convertTo(resizedRGBImage, CV_32FC3, 1.0 / 255.0f);

  NHWC3ToNC3HW(reinterpret_cast<const float *>(resizedRGBImage.data), inputData,
               inputMean_.data(), inputStd_.data(), inputShape[3],
               inputShape[2]);
}

float fast_exp(float x) {
  union {
      uint32_t i;
      float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}

template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length) {
  const _Tp alpha = *std::max_element(src, src + length);
  _Tp denominator{0};
  for (int i = 0; i < length; ++i) {
    dst[i] = fast_exp(src[i] - alpha);
    denominator += dst[i];
  }
  for (int i = 0; i < length; ++i) {
    dst[i] /= denominator;
  }
  return 0;
}

void Detector::PicoDetPostProcess(std::vector<ObjectResult>* results,
                                  std::vector<const float *> outs,
                                  std::vector<int> fpn_stride,
                                  std::vector<int> im_shape,
                                  std::vector<float> scale_factor,
                                  float score_threshold,
                                  float nms_threshold,
                                  int num_class,
                                  int reg_max) {
  std::vector<std::vector<ObjectResult>> bbox_results;
  bbox_results.resize(num_class);
  int in_h = im_shape[0], in_w = im_shape[1];
  for (int i = 0; i < fpn_stride.size(); ++i) {
    int feature_h = in_h / fpn_stride[i];
    int feature_w = in_w / fpn_stride[i];
    for (int idx = 0; idx < feature_h * feature_w; idx++) {
      const float *scores = outs[i] + (idx * num_class);

      int row = idx / feature_w;
      int col = idx % feature_w;
      float score = 0;
      int cur_label = 0;
      for (int label = 0; label < num_class; label++) {
        if (scores[label] > score) {
          score = scores[label];
          cur_label = label;
        }
      }
      if (score > score_threshold) {
        const float *bbox_pred = outs[i + fpn_stride.size()]
                                 + (idx * 4 * (reg_max + 1));
        bbox_results[cur_label].push_back(Detector::disPred2Bbox(bbox_pred,
                                                                 cur_label, score, col, row, fpn_stride[i], im_shape, reg_max));
      }
    }
  }
  for (int i = 0; i < (int)bbox_results.size(); i++) {
    Detector::nms(bbox_results[i], nms_threshold);

    for (auto box : bbox_results[i]) {
      box.rect[0] = box.rect[0] / scale_factor[1];
      box.rect[2] = box.rect[2] / scale_factor[1];
      box.rect[1] = box.rect[1] / scale_factor[0];
      box.rect[3] = box.rect[3] / scale_factor[0];
      results->push_back(box);
    }
  }
}

void Detector::nms(std::vector<ObjectResult> &input_boxes, float nms_threshold) {
  std::sort(input_boxes.begin(),
            input_boxes.end(),
            [](ObjectResult a, ObjectResult b) { return a.confidence > b.confidence; });
  std::vector<float> vArea(input_boxes.size());
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    vArea[i] = (input_boxes.at(i).rect[2] - input_boxes.at(i).rect[0] + 1)
               * (input_boxes.at(i).rect[3] - input_boxes.at(i).rect[1] + 1);
  }
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    for (int j = i + 1; j < int(input_boxes.size());) {
      float xx1 = (std::max)(input_boxes[i].rect[0], input_boxes[j].rect[0]);
      float yy1 = (std::max)(input_boxes[i].rect[1], input_boxes[j].rect[1]);
      float xx2 = (std::min)(input_boxes[i].rect[2], input_boxes[j].rect[2]);
      float yy2 = (std::min)(input_boxes[i].rect[3], input_boxes[j].rect[3]);
      float w = (std::max)(float(0), xx2 - xx1 + 1);
      float h = (std::max)(float(0), yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= nms_threshold) {
        input_boxes.erase(input_boxes.begin() + j);
        vArea.erase(vArea.begin() + j);
      }
      else {
        j++;
      }
    }
  }
}

// PicoDet decode
ObjectResult Detector::disPred2Bbox(const float *&dfl_det, int label, float score,
                                    int x, int y, int stride, std::vector<int> im_shape,
                                    int reg_max) {
  float ct_x = (x + 0.5) * stride;
  float ct_y = (y + 0.5) * stride;
  std::vector<float> dis_pred;
  dis_pred.resize(4);
  for (int i = 0; i < 4; i++) {
    float dis = 0;
    float* dis_after_sm = new float[reg_max + 1];
    activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);
    for (int j = 0; j < reg_max + 1; j++) {
      dis += j * dis_after_sm[j];
    }
    dis *= stride;
    dis_pred[i] = dis;
    delete[] dis_after_sm;
  }
  int xmin = (int)(std::max)(ct_x - dis_pred[0], .0f);
  int ymin = (int)(std::max)(ct_y - dis_pred[1], .0f);
  int xmax = (int)(std::min)(ct_x + dis_pred[2], (float)im_shape[0]);
  int ymax = (int)(std::min)(ct_y + dis_pred[3], (float)im_shape[1]);

  ObjectResult result_item;
  result_item.rect = {xmin, ymin, xmax, ymax};
  result_item.class_id = label;
  result_item.confidence = score;

  // TODO: add some infos
  result_item.class_name = result_item.class_id >= 0 && result_item.class_id < labelList_.size()
                           ? labelList_[result_item.class_id]
                           : "Unknow";
  result_item.fill_color = result_item.class_id >= 0 && result_item.class_id < colorMap_.size()
                           ? colorMap_[result_item.class_id]
                           : cv::Scalar(0, 0, 0);

  return result_item;
}

void Detector::Predict(const cv::Mat &rgbaImage, std::vector<ObjectResult> &results,
                       double *preprocessTime, double *predictTime,
                       double *postprocessTime) {
  auto t = GetCurrentTime();

  t = GetCurrentTime();
  Preprocess(rgbaImage);
  *preprocessTime = GetElapsedTime(t);
  LOGD("Detector preprocess costs %f ms", *preprocessTime);

  t = GetCurrentTime();
  predictor_->Run();
  *predictTime = GetElapsedTime(t);
  LOGD("Detector predict costs %f ms", *predictTime);

  t = GetCurrentTime();
  std::vector<float> scale_factor = {
          static_cast<float>(inputHeight_) / static_cast<float>(rgbaImage.rows),
          static_cast<float>(inputWidth_) / static_cast<float>(rgbaImage.cols)
  };
  std::vector<int> im_shape;
  im_shape.push_back(inputHeight_);
  im_shape.push_back(inputWidth_);
  auto output_names = predictor_->GetOutputNames();
  std::vector<const float *> outs;
  int num_class;
  int reg_max;
  for (int i = 0; i < output_names.size(); i++) {
    auto output_tensor = predictor_->GetTensor(output_names[i]);
    const float* outptr = output_tensor->data<float>();
    std::vector<int64_t> output_shape = output_tensor->shape();
    if (i == 0) {
      num_class = output_shape[2];
    }
    if (i == inputFpnStride_.size()) {
      reg_max = output_shape[2] / 4 - 1;
    }
    outs.push_back(outptr);
  }
  PicoDetPostProcess(&results, outs, inputFpnStride_, im_shape, scale_factor, scoreThreshold_, nmsThreshold_, num_class, reg_max);
  *postprocessTime = GetElapsedTime(t);
  LOGD("Detector postprocess costs %f ms", *postprocessTime);
}

Pipeline::Pipeline(const std::string &modelDir, const std::string &labelPath,
                   const int cpuThreadNum, const std::string &cpuPowerMode,
                   int inputWidth, int inputHeight,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd, float scoreThreshold,
                   const std::string &configPath, float nmsThreshold, const std::vector<int> &inputFpnStride) {
  detector_.reset(new Detector(modelDir, labelPath, cpuThreadNum, cpuPowerMode,
                               inputWidth, inputHeight, inputMean, inputStd,
                               scoreThreshold,
                               configPath, nmsThreshold, inputFpnStride));
}

void Pipeline::VisualizeResults(const std::vector<ObjectResult> &results,
                                cv::Mat *rgbaImage) {
  int w = rgbaImage->cols;
  int h = rgbaImage->rows;
  for (int i = 0; i < results.size(); i++) {
    ObjectResult object = results[i];
    cv::Rect boundingBox =
            cv::Rect(object.rect[0], object.rect[1],
                     object.rect[2] - object.rect[0], object.rect[3] - object.rect[1]) &
            cv::Rect(0, 0, w - 1, h - 1); // image.cols, image.rows
    // Configure text size
    std::string text = object.class_name + " ";
    text += std::to_string(static_cast<int>(object.confidence * 100)) + "%";
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 1.5f;
    float fontThickness = 1.0f;
    cv::Size textSize =
            cv::getTextSize(text, fontFace, fontScale, fontThickness, nullptr);
    // Draw roi object, text, and background
    cv::rectangle(*rgbaImage, boundingBox, object.fill_color, 2);
    cv::rectangle(*rgbaImage,
                  cv::Point2d(boundingBox.x,
                              boundingBox.y - round(textSize.height * 1.25f)),
                  cv::Point2d(boundingBox.x + boundingBox.width, boundingBox.y),
                  object.fill_color, -1);
    cv::putText(*rgbaImage, text, cv::Point2d(boundingBox.x, boundingBox.y),
                fontFace, fontScale, cv::Scalar(255, 255, 255), fontThickness);
  }
}

void Pipeline::VisualizeStatus(double readGLFBOTime, double writeGLTextureTime,
                               double preprocessTime, double predictTime,
                               double postprocessTime, cv::Mat *rgbaImage) {
  char text[255];
  cv::Scalar fontColor = cv::Scalar(255, 255, 255);
  int fontFace = cv::FONT_HERSHEY_PLAIN;
  double fontScale = 1.f;
  float fontThickness = 1;
  sprintf(text, "Read GLFBO time: %.1f ms", readGLFBOTime);
  cv::Size textSize =
          cv::getTextSize(text, fontFace, fontScale, fontThickness, nullptr);
  textSize.height *= 1.25f;
  cv::Point2d offset(10, textSize.height + 15);
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Write GLTexture time: %.1f ms", writeGLTextureTime);
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Preprocess time: %.1f ms", preprocessTime);
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Predict time: %.1f ms", predictTime);
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Postprocess time: %.1f ms", postprocessTime);
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
}

bool Pipeline::Process(int inTexureId, int outTextureId, int textureWidth,
                       int textureHeight, std::string savedImagePath) {
  static double readGLFBOTime = 0, writeGLTextureTime = 0;
  double preprocessTime = 0, predictTime = 0, postprocessTime = 0;

  // Read pixels from FBO texture to CV image
  cv::Mat rgbaImage;
  CreateRGBAImageFromGLFBOTexture(textureWidth, textureHeight, &rgbaImage,
                                  &readGLFBOTime);
  // Feed the image, run inference and parse the results
  std::vector<ObjectResult> results;
  detector_->Predict(rgbaImage, results, &preprocessTime, &predictTime,
                     &postprocessTime);

  // Visualize the objects to the origin image
  VisualizeResults(results, &rgbaImage);

  // Visualize the status(performance data) to the origin image
  VisualizeStatus(readGLFBOTime, writeGLTextureTime, preprocessTime,
                  predictTime, postprocessTime, &rgbaImage);

  // Dump modified image if savedImagePath is set
  if (!savedImagePath.empty()) {
    cv::Mat bgrImage;
    cv::cvtColor(rgbaImage, bgrImage, cv::COLOR_RGBA2BGR);
    imwrite(savedImagePath, bgrImage);
  }

  // Write back to texture2D
  WriteRGBAImageBackToGLTexture(rgbaImage, outTextureId, &writeGLTextureTime);
  return true;
}
