
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

#include "pipeline.hpp"
#include "random.hpp"
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


/* 
 * methods for DataLoaderNP
 *  */
Batch DataLoaderNp::_get_batch() {

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

    vector<std::future<void>> tpool(num_workers);

    for (int i{0}; i < num_workers; ++i) {
        tpool[i] = std::move(thread_pool.submit(thread_func, i));
    }
    for (int i{0}; i < num_workers; ++i) {
        tpool[i].get();
    }
    pos += n_batch;

    vector<int> dsize;
    if (nchw) {
        dsize = {n_batch, 3, height, width};
    } 
    if (!nchw){
        dsize = {n_batch, height, width, 3};
    }
    vector<int> lsize{n_batch};

    spl.data = data;
    spl.labels = labels;
    spl.dsize.swap(dsize);
    spl.lsize.swap(lsize);
    return spl;
}



