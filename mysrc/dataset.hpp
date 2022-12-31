#ifndef _DATASET_HPP_
#define _DATASET_HPP_

#include <array>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using std::string;
using std::array;
using std::vector;
using cv::Mat;


class DataSet {
public:
    vector<string> img_paths;
    vector<int64_t> labels;
    int n_samples;
    array<int, 2> size; //height, width
    bool inplace{ true };
    bool nchw{ true };

    DataSet(const std::vector<string>& files, array<int, 2> size = { 224, 224 }, bool nchw = true, bool inplace = true);
    //DataSet(string rootpth, string fname, array<int, 2> size = { 224, 224 }, bool nchw = true, bool inplace = true);
    DataSet() { set_default_states(); }

    // void init(string rootpth, string fname, array<int, 2> size={224, 224}, bool nchw=true, int ra_n=2, int ra_m=9, bool inplace=true);
    void parse_annos(string imroot, string annfile);
    cv::Mat TransTrain(cv::Mat& im);
    cv::Mat TransVal(cv::Mat& im);
    //void get_one_by_idx(int idx, float* data, int64_t& lb);
    void get_one_by_idx(int idx, float* data, int* locations);
    void letter_box(const cv::Mat& img, cv::Size new_shape, cv::Mat& dst, float& r, cv::Point& d, cv::Scalar color = cv::Scalar(114, 114, 114),
        bool auto_mode = false, bool scaleup = true, int stride = 32);
    void getxy_from_fullname(const cv::String& fullname, int& x, int& y);

    void Mat2Mem(cv::Mat& im, float* res);
    int get_n_samples();
    void set_default_states();

    void _train();
    void _eval();
    void _set_rand_aug(int ra_n = 2, int ra_m = 9);
};


#endif
