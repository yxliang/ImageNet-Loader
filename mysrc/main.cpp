#include "dataloader.hpp"
#include <iostream>
#include <vector>
//using std::endl;
//using std::cout;
//using std::vector;
using namespace cv;
using namespace std;  //error

const std::string KEYS =
"{im_dir                    |D:\\Pinxin\\Dev\\export_tct_ngc_12.20\\TCTAnnotated(non-gynecologic)/TCT_NGC-BJXK-20210910/BJXFK-XA-575822/  | images fold         }"
"{batch_size                |16                                                                               |batch_size      }"
"{num_worker                |2                                                                                  |num_worker    }"
"{height                    |1280                                                                                |height      }"
"{width                     |1280                                                                                |width      }"
;

int main(int argc, char* argv[]) {
	cv::CommandLineParser parser(argc, argv, KEYS);
	string im_dir = parser.get<string>("im_dir");
	int batch_size = parser.get<int>("batch_size");
	int num_worker = parser.get<int>("num_worker");
	int height = parser.get<int>("height");
	int width = parser.get<int>("width");
	num_worker = std::min(num_worker, batch_size);
	bool nchw(true), shuffle(false), drop_last(false);
	std::vector<std::string> image_files;
	try {
		cv::glob(im_dir + "\\*.jpg", image_files, true);
	}
	catch (...) {
		std::cerr << "List image error";
		return 0;
	}
	
	DataLoaderWSI dl(image_files, batch_size, { height, width }, nchw, shuffle, num_worker, false);
	//for (int i{ 0 }; i < 10; ++i) {
	//	std::cout << dl.dataset.img_paths[i] << std::endl;
	//}
	//for (int i{ 0 }; i < 10; ++i) {
	//	std::cout << dl.indices[i] << std::endl;
	//}

	std::cout << "run get batch: " << std::endl;
	int64 t;
	dl.start_prefetcher();
	t = cv::getTickCount();
	Batch batch;
	int i{ 0 };
	while (!dl._is_end()) {
		++i;
		t = cv::getTickCount();
		batch = dl._next_batch();
		std::cout << i << "th batch. " << (cv::getTickCount() - t) * 1000 / cv::getTickFrequency() << "ms." << " batch size: " << batch.dsize[0] << std::endl;

		if (batch.data) delete batch.data;
		if (batch.locations) delete batch.locations;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	return 0;
}
