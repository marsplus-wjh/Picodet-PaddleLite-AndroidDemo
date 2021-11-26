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

#pragma once

#include "Utils.h"
#include "paddle_api.h"
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>

// Object Detection Result for PicoDet
struct ObjectResult {
    // Rectangle coordinates of detected object: left, right, top, down
    std::vector<int> rect;
    // Class id of detected object
    int class_id;
    std::string class_name;
    // Confidence of detected object
    float confidence;
    cv::Scalar fill_color;
};

class Detector {
public:
    explicit Detector(const std::string &modelDir, const std::string &labelPath,
                      const int cpuThreadNum, const std::string &cpuPowerMode,
                      int inputWidth, int inputHeight,
                      const std::vector<float> &inputMean,
                      const std::vector<float> &inputStd, float scoreThreshold,
                      const std::string &configPath, float nmsThreshold, const std::vector<int> &inputFpnStride);

    void Predict(const cv::Mat &rgbImage, std::vector<ObjectResult> &results,
                 double *preprocessTime, double *predictTime,
                 double *postprocessTime);

private:
    std::vector<std::string> LoadLabelList(const std::string &path);

    std::vector<std::string> LoadConfigs(const std::string &path);

    std::vector<cv::Scalar> GenerateColorMap(int numOfClasses);

    std::vector<float>scale_factor;

    void Preprocess(const cv::Mat &rgbaImage);

    void PicoDetPostProcess(std::vector<ObjectResult> *results,
                            std::vector<const float *> outs,
                            std::vector<int> fpn_stride,
                            std::vector<int> im_shape,
                            std::vector<float> scale_factor,
                            float score_threshold,
                            float nms_threshold,
                            int num_class,
                            int reg_max);

    ObjectResult disPred2Bbox(const float *&dfl_det, int label, float score,
                              int x, int y, int stride, std::vector<int> im_shape,
                              int reg_max);

    void nms(std::vector<ObjectResult> &input_boxes, float nms_threshold);

private:
    int inputWidth_;
    int inputHeight_;
    std::vector<float> inputMean_;
    std::vector<float> inputStd_;
    float scoreThreshold_;
    float nmsThreshold_;
    std::vector<std::string> labelList_;
    std::vector<cv::Scalar> colorMap_;
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
    std::vector<std::string> configs_;
    std::vector<int> inputFpnStride_;
};

class Pipeline {
public:
    Pipeline(const std::string &modelDir, const std::string &labelPath,
             const int cpuThreadNum, const std::string &cpuPowerMode,
             int inputWidth, int inputHeight, const std::vector<float> &inputMean,
             const std::vector<float> &inputStd, float scoreThreshold,
             const std::string &configPath, float nmsThreshold, const std::vector<int> &inputFpnStride);

    bool Process(int inTextureId, int outTextureId, int textureWidth,
                 int textureHeight, std::string savedImagePath);

private:
    // Read pixels from FBO texture to CV image
    void CreateRGBAImageFromGLFBOTexture(int textureWidth, int textureHeight,
                                         cv::Mat *rgbaImage,
                                         double *readGLFBOTime) {
      auto t = GetCurrentTime();
      rgbaImage->create(textureHeight, textureWidth, CV_8UC4);
      glReadPixels(0, 0, textureWidth, textureHeight, GL_RGBA, GL_UNSIGNED_BYTE,
                   rgbaImage->data);
      *readGLFBOTime = GetElapsedTime(t);
      LOGD("Read from FBO texture costs %f ms", *readGLFBOTime);
    }

    // Write back to texture2D
    void WriteRGBAImageBackToGLTexture(const cv::Mat &rgbaImage, int textureId,
                                       double *writeGLTextureTime) {
      auto t = GetCurrentTime();
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, textureId);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, rgbaImage.cols, rgbaImage.rows,
                      GL_RGBA, GL_UNSIGNED_BYTE, rgbaImage.data);
      *writeGLTextureTime = GetElapsedTime(t);
      LOGD("Write back to texture2D costs %f ms", *writeGLTextureTime);
    }

    // Visualize the results to origin image
    void VisualizeResults(const std::vector<ObjectResult> &results, cv::Mat *rgbaImage);

    // Visualize the status(performace data) to origin image
    void VisualizeStatus(double readGLFBOTime, double writeGLTextureTime,
                         double preprocessTime, double predictTime,
                         double postprocessTime, cv::Mat *rgbaImage);

private:
    std::shared_ptr<Detector> detector_;
};
