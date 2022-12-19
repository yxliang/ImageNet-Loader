
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
#include <opencv2/opencv.hpp>

//#include "pipeline.hpp"
//#include "random.hpp"
#include "dataloader.hpp"
#include "blocking_queue.hpp"


using std::endl;
using std::cout;
using std::vector;
using std::string;
using std::array;
using std::ifstream;
using std::stringstream;
using std::ios;
using cv::Mat;


// functions of Batch
Batch::Batch(vector<float> *dt, vector<int> dsz, vector<int64_t> *lbs, vector<int> lsz)
    : data(dt), dsize(dsz), labels(lbs), lsize(lsz) {}
//
//WSI_Batch::WSI_Batch(vector<cv::Mat>* dt, vector<int> dsz)
//    : data(dt), dsize(dsz) {}


/* 
 * methods for DataLoaderNP
 *  */
Batch DataLoaderNp::_get_batch() {
    // auto t1 = std::chrono::steady_clock::now();
    //CHECK (!pos_end()) << "want more samples than n_samples, there can be some logical problem\n";

    Batch spl;

    int n_batch = batchsize;
    if (pos + batchsize > n_samples) n_batch = n_samples - pos;
    int single_size = width * height * 3;
    vector<float> *data = new vector<float>(n_batch * single_size);
    vector<int64_t> *labels = new vector<int64_t>(n_batch);
    int bs_thread = n_batch / num_workers + 1;
    auto thread_func = [&](int thread_idx) {
        for (int i{0}; i < bs_thread; ++i) {
            int pos_thread = i * num_workers + thread_idx;
            if (pos_thread >= n_batch) break;
            dataset.get_one_by_idx(
                    indices[pos + pos_thread],
                    &((*data)[pos_thread * single_size]),
                    (*labels)[pos_thread]
                    );
        }
    };

    // auto t2 = std::chrono::steady_clock::now();
    vector<std::future<void>> tpool(num_workers);
    // vector<std::future<void>> tpool;
    for (int i{0}; i < num_workers; ++i) {
        // tpool[i] = std::async(std::launch::async, thread_func, i);
        tpool[i] = std::move(thread_pool.submit(thread_func, i));
    }
    for (int i{0}; i < num_workers; ++i) {
        tpool[i].get();
    }
    pos += n_batch;
    //cout << "pos: " << pos << "; n_batch: " << n_batch << endl;
    // auto t3 = std::chrono::steady_clock::now();

    vector<int> dsize;
    if (nchw) {
        dsize = {n_batch, 3, height, width};
    } 
    if (!nchw){
        dsize = {n_batch, height, width, 3};
    }
    vector<int> lsize{n_batch};
    // auto t4 = std::chrono::steady_clock::now();
    //
    // cout << "prepare thread_func and memory: "
    //     << std::chrono::duration<double, std::milli>(t2 - t1).count() << endl;
    // cout << "processing: "
    //     << std::chrono::duration<double, std::milli>(t3 - t2).count() << endl;
    spl.data = data;
    spl.labels = labels;
    spl.dsize.swap(dsize);
    spl.lsize.swap(lsize);
    return spl;
}


//WSI_Batch DataLoaderWSI::_get_batch() {
//    // auto t1 = std::chrono::steady_clock::now();
//    //CHECK (!pos_end()) << "want more samples than n_samples, there can be some logical problem\n";
//
//    WSI_Batch spl;
//
//    int n_batch = batchsize;
//    if (pos + batchsize > n_samples) n_batch = n_samples - pos;
//    int single_size = width * height * 3;
//    vector<cv::Mat> data = new vector<float>(n_batch * single_size);
//    vector<int64_t>* labels = new vector<int64_t>(n_batch);
//    int bs_thread = n_batch / num_workers + 1;
//    auto thread_func = [&](int thread_idx) {
//        for (int i{ 0 }; i < bs_thread; ++i) {
//            int pos_thread = i * num_workers + thread_idx;
//            if (pos_thread >= n_batch) break;
//            dataset.GetItem(
//                indices[pos + pos_thread],
//                &((*data)[pos_thread * single_size])
//            );
//        }
//    };
//
//    // auto t2 = std::chrono::steady_clock::now();
//    vector<std::future<void>> tpool(num_workers);
//    // vector<std::future<void>> tpool;
//    for (int i{ 0 }; i < num_workers; ++i) {
//        // tpool[i] = std::async(std::launch::async, thread_func, i);
//        tpool[i] = std::move(thread_pool.submit(thread_func, i));
//    }
//    for (int i{ 0 }; i < num_workers; ++i) {
//        tpool[i].get();
//    }
//    pos += n_batch;
//    cout << "pos: " << pos << "; n_batch: " << n_batch << endl;
//    // auto t3 = std::chrono::steady_clock::now();
//
//    vector<int> dsize;
//    if (nchw) {
//        dsize = { n_batch, 3, height, width };
//    }
//    if (!nchw) {
//        dsize = { n_batch, height, width, 3 };
//    }
//    vector<int> lsize{ n_batch };
//    // auto t4 = std::chrono::steady_clock::now();
//    //
//    // cout << "prepare thread_func and memory: "
//    //     << std::chrono::duration<double, std::milli>(t2 - t1).count() << endl;
//    // cout << "processing: "
//    //     << std::chrono::duration<double, std::milli>(t3 - t2).count() << endl;
//    spl.data = data;
//    spl.labels = labels;
//    spl.dsize.swap(dsize);
//    spl.lsize.swap(lsize);
//    return spl;
//}


void Normalize(Mat& im, array<float, 3> mean, array<float, 3> std, float* p_res, double pca_std, bool nchw) {
    /* merge 1/255, pca-noise, mean/var and layout change operations together, so see if this can be faster */

    // for pca noise
    vector<float> rgb(3, 0);

    // for mean/var
    for (int i{ 0 }; i < 3; ++i) {
        std[i] = 1.f / std[i];
    }

    float scale = static_cast<float>(1. / 255.);
    im.forEach<cv::Vec3b>([&](cv::Vec3b& pix, const int* pos) {
        for (int i{ 0 }; i < 3; ++i) {
            float tmp = static_cast<float>(pix[i]) * scale;
            tmp += rgb[2 - i];
            tmp = (tmp - mean[i]) * std[i];
            int offset;
            if (nchw) {
                offset = i * im.rows * im.cols + pos[0] * im.cols + pos[1];
            }
            else {
                offset = pos[0] * im.cols * 3 + pos[1] * 3 + i;
            }
            p_res[offset] = tmp;
        }
        });
}

// member function of TransformTrain
DataSet::DataSet(string rootpth, string fname, array<int, 2> size, bool nchw, bool inplace) : size(size), inplace(inplace), nchw(nchw) {
    set_default_states();
    parse_annos(rootpth, fname);
}

int DataSet::get_n_samples() {
    return n_samples;
}

void DataSet::get_one_by_idx(int idx, float* data, int64_t& label) {
    CHECK(data != nullptr) << "memory not allocated, implement error\n";
    string impth = img_paths[idx];
    //cout << impth << endl;
    Mat im = cv::imread(impth, cv::ImreadModes::IMREAD_COLOR);
    CHECK(!im.empty()) << "image " << impth << "does not exists\n";
    //if (is_train) {
    //    im = TransTrain(im);
    //} else {
    //    im = TransVal(im);
    //}
    Normalize(im, { 0.485f, 0.456f, 0.406f }, { 0.229f, 0.224f, 0.225f }, data, 0, true);
     //Mat2Mem(im, data);
    //label = labels[idx];
}

void DataSet::Mat2Mem(Mat& im, float* res) {
    CHECK(res != nullptr) << "res should not be nullptr\n";
    int row_size = im.cols * 3;
    int chunk_size = row_size * sizeof(float);
    if (nchw) {
        int plane_size = im.rows * im.cols;
        for (int h{ 0 }; h < im.rows; ++h) {
            float* ptr = im.ptr<float>(h);
            int offset_w = 0;
            for (int w{ 0 }; w < im.cols; ++w) {
                for (int c{ 0 }; c < 3; ++c) {
                    int offset = c * plane_size + h * im.cols + w;
                    res[offset] = ptr[offset_w];
                    ++offset_w;
                }
            }
        }
    }
    else {
        for (int h{ 0 }; h < im.rows; ++h) {
            float* ptr = im.ptr<float>(h);
            int offset_res = row_size * h;
            memcpy((void*)(res + offset_res), (void*)ptr, chunk_size);
        }
    }
}

void DataSet::set_default_states() {
    inplace = true;
    nchw = true;
}

void DataSet::parse_annos(string imroot, string annfile) {
    ifstream fin(annfile, ios::in);
    CHECK(fin) << "file does not exists: " << annfile << endl;
    stringstream ss;
    fin >> ss.rdbuf(); // std::noskipws
    CHECK(!(fin.fail() && fin.eof())) << "error when read ann file\n";
    fin.close();

    n_samples = 0;
    string buf;
    while (std::getline(ss, buf)) { ++n_samples; };
    ss.clear(); ss.seekg(0);

    img_paths.resize(n_samples);
    for (int i{ 0 }; i < n_samples; ++i) {
        ss >> buf;
        img_paths[i] = buf;
    }

    //labels.resize(n_samples);
    //int tmp = 0;
    //// if (imroot[imroot.size()-1] == '/') {tmp = 1;}
    //if (imroot.back() == '/') {tmp = 1;}
    //for (int i{0}; i < n_samples; ++i) {
    //    ss >> buf >> labels[i];
    //    int num_split = tmp;
    //    if (buf[0] == '/') ++num_split;
    //    if (num_split == 0) {
    //        img_paths[i] = imroot + "/" + buf;
    //    } else if (num_split == 1) {
    //        img_paths[i] = imroot + buf;
    //    } else {
    //        img_paths[i] = imroot + buf.substr(1);
    //    }
    //}
}