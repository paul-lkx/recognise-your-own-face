#include "opencv2/face/facerec.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/core.hpp"
#include <iostream>  
#include <fstream>
#include "sstream"


using namespace cv;
using namespace cv::face;
using namespace std;

static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	// 创建和返回一个归一化后的图像矩阵:  
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}


static void read_csv(const string& filename, vector<Mat>& images,
                     vector<int>& labels, std::map<int, string>& labelsInfo, char separator = ';') {
    ifstream csv(filename.c_str());
    if (!csv) CV_Error(Error::StsBadArg, "No valid input file was given, please check the given filename.");
    string line, path, classlabel, info;
    while (getline(csv, line)) {
        stringstream liness(line);
        path.clear(); classlabel.clear(); info.clear();
        getline(liness, path, separator);
        getline(liness, classlabel, separator);
        getline(liness, info, separator);
        if(!path.empty() && !classlabel.empty()) {
            cout << "Processing " << path << endl;
            int label = atoi(classlabel.c_str());
            if(!info.empty())
                labelsInfo.insert(std::make_pair(label, info));
            // 'path' can be file, dir or wildcard path
            String root(path.c_str());
            vector<String> files;
            glob(root, files, true);
            for(vector<String>::const_iterator f = files.begin(); f != files.end(); ++f) {
                cout << "\t" << *f << endl;
                Mat img = imread(*f, IMREAD_GRAYSCALE);
                static int w=-1, h=-1;
                static bool showSmallSizeWarning = true;
                if(w>0 && h>0 && (w!=img.cols || h!=img.rows)) cout << "\t* Warning: images should be of the same size!" << endl;
                if(showSmallSizeWarning && (img.cols<50 || img.rows<50)) {
                    cout << "* Warning: for better results images should be not smaller than 50x50!" << endl;
                    showSmallSizeWarning = false;
                }
                images.push_back(img);
                labels.push_back(label);
            }
        }
    }
}

int main() {

	//读取你的CSV文件路径.  
	//string fn_csv = string(argv[1]);  
	string fn_csv = "../at.txt";

	// 2个容器来存放图像数据和对应的标签  
	vector<Mat> images;
	vector<int> labels;
    std::map <int, string> labelsInfo;
	// 读取数据. 如果文件不合法就会出错  
	// 输入的文件名已经有了.  
	try
	{
        read_csv(fn_csv, images, labels, labelsInfo);
//        std::cout << images.size() << std::endl;
	}
	catch (cv::Exception& e)
	{
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// 文件有问题，我们啥也做不了了，退出了
		exit(1);
	}
	// 如果没有读取到足够图片，也退出.  
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}

	// 下面的几行代码仅仅是从你的数据集中移除最后一张图片  
	//[gm:自然这里需要根据自己的需要修改，他这里简化了很多问题]  
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();
	// 下面几行创建了一个特征脸模型用于人脸识别，  
	// 通过CSV文件读取的图像和标签训练它。  
	// T这里是一个完整的PCA变换  
	//如果你只想保留10个主成分，使用如下代码  
	//      cv::createEigenFaceRecognizer(10);  
	//  
	// 如果你还希望使用置信度阈值来初始化，使用以下语句：  
	//      cv::createEigenFaceRecognizer(10, 123.0);  
	//  
	// 如果你使用所有特征并且使用一个阈值，使用以下语句：  
	//      cv::createEigenFaceRecognizer(0, 123.0);  

	Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
	model->train(images, labels);
	model->save("MyFacePCAModel.xml");

	Ptr<FisherFaceRecognizer> model1 = FisherFaceRecognizer::create();
	model1->train(images, labels);
	model1->save("MyFaceFisherModel.xml");

	Ptr<LBPHFaceRecognizer> model2 = LBPHFaceRecognizer::create();
	model2->train(images, labels);
	model2->save("MyFaceLBPHModel.xml");

	// 下面对测试图像进行预测，predictedLabel是预测标签结果  
	int predictedLabel = model->predict(testSample);
	int predictedLabel1 = model1->predict(testSample);
	int predictedLabel2 = model2->predict(testSample);

	// 还有一种调用方式，可以获取结果同时得到阈值:  
	//      int predictedLabel = -1;  
	//      double confidence = 0.0;  
	//      model->predict(testSample, predictedLabel, confidence);  

	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	string result_message1 = format("Predicted class = %d / Actual class = %d.", predictedLabel1, testLabel);
	string result_message2 = format("Predicted class = %d / Actual class = %d.", predictedLabel2, testLabel);
	cout << result_message << endl;
	cout << result_message1 << endl;
	cout << result_message2 << endl;

	getchar();
	//waitKey(0);
	return 0;
}