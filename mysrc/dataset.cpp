
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <array>
#include <fstream>
#include <sstream>
#include <future>
#include <thread>
#include <glog/logging.h>
#include "dataset.hpp"

using std::endl;
using std::cout;
using std::string;
using std::stringstream;
using std::ifstream;
using std::ios;
using std::array;
using std::vector;
using cv::Mat;


DataSet::DataSet(const std::vector<string>& files, array<int, 2> size, bool nchw, bool inplace) 
    : size(size), inplace(inplace), nchw(nchw), img_paths(files){
    set_default_states();
    n_samples = static_cast<int>(files.size());
}

int DataSet::get_n_samples() {
    return n_samples;
}

void DataSet::get_one_by_idx(int idx, float* data, int* locations) {
    CHECK(data != nullptr) << "memory not allocated, implement error\n";
    string impth = img_paths[idx];
    Mat image = cv::imread(impth, cv::ImreadModes::IMREAD_COLOR);
    float ratio = 1.0f;
    cv::Point dxy;
    cv::Mat dst; int width = size[1], height = size[0];
    letter_box(image, cv::Size(width, height), dst, ratio, dxy);
    cv::Mat blob = cv::dnn::blobFromImage(dst, 1 / 255.0, cv::Size(width, height), cv::Scalar(0, 0, 0), true, false).reshape(1, 1);
    std::vector<float> value = (std::vector<float>)(blob);
    memcpy(&data[0], &value[0], value.size()*sizeof(float));

    int x{ 0 }, y{0};
    getxy_from_fullname(impth, x, y);
    locations[0] = x; locations[1] = y;
}

void DataSet::letter_box(const cv::Mat& img, cv::Size new_shape, cv::Mat& dst, float& r, cv::Point& d, cv::Scalar color,
    bool auto_mode, bool scaleup, int stride) {
    //# Resizeand pad image while meeting stride - multiple constraints
    float width = img.cols, height = img.rows;

    //# Scale ratio(new / old)
    r = std::min(new_shape.width / width, new_shape.height / height);
    if (!scaleup) // # only scale down, do not scale up (for better val mAP)
        r = std::min(r, 1.0f);

    //# Compute padding
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    int dw = new_shape.width - new_unpadW;  //# wh padding
    int dh = new_shape.height - new_unpadH;
    if (auto_mode) { //# minimum rectangle, wh padding
        dw %= stride;
        dh %= stride;
    }
    dw /= 2, dh /= 2; //# divide padding into 2 sides
    d.x = dw, d.y = dh;

    resize(img, dst, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_LINEAR);
    int top = int(round(dh - 0.1));
    int bottom = int(round(dh + 0.1));
    int left = int(round(dw - 0.1));
    int right = int(round(dw + 0.1));
    copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT, color); //# add border
}

void DataSet::getxy_from_fullname(const cv::String& fullname, int& x, int& y) {
    //remove the extension and path
    auto filename = fullname.substr(fullname.find_last_of("/\\") + 1);
    filename = filename.substr(0, filename.find_last_of("."));

    //parse the string
    int delimeter_pos = filename.find_first_of("_");
    std::stringstream x_str(filename.substr(0, delimeter_pos)), y_str(filename.substr(delimeter_pos + 1));
    x_str >> x; y_str >> y;
}

void DataSet::set_default_states() {
    inplace = true;
    nchw = true;
}

