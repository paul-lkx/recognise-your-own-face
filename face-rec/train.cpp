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
	// �����ͷ���һ����һ�����ͼ�����:  
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

	//��ȡ���CSV�ļ�·��.  
	//string fn_csv = string(argv[1]);  
	string fn_csv = "../at.txt";

	// 2�����������ͼ�����ݺͶ�Ӧ�ı�ǩ  
	vector<Mat> images;
	vector<int> labels;
    std::map <int, string> labelsInfo;
	// ��ȡ����. ����ļ����Ϸ��ͻ����  
	// ������ļ����Ѿ�����.  
	try
	{
        read_csv(fn_csv, images, labels, labelsInfo);
//        std::cout << images.size() << std::endl;
	}
	catch (cv::Exception& e)
	{
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// �ļ������⣬����ɶҲ�������ˣ��˳���
		exit(1);
	}
	// ���û�ж�ȡ���㹻ͼƬ��Ҳ�˳�.  
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}

	// ����ļ��д�������Ǵ�������ݼ����Ƴ����һ��ͼƬ  
	//[gm:��Ȼ������Ҫ�����Լ�����Ҫ�޸ģ���������˺ܶ�����]  
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();
	// ���漸�д�����һ��������ģ����������ʶ��  
	// ͨ��CSV�ļ���ȡ��ͼ��ͱ�ǩѵ������  
	// T������һ��������PCA�任  
	//�����ֻ�뱣��10�����ɷ֣�ʹ�����´���  
	//      cv::createEigenFaceRecognizer(10);  
	//  
	// ����㻹ϣ��ʹ�����Ŷ���ֵ����ʼ����ʹ��������䣺  
	//      cv::createEigenFaceRecognizer(10, 123.0);  
	//  
	// �����ʹ��������������ʹ��һ����ֵ��ʹ��������䣺  
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

	// ����Բ���ͼ�����Ԥ�⣬predictedLabel��Ԥ���ǩ���  
	int predictedLabel = model->predict(testSample);
	int predictedLabel1 = model1->predict(testSample);
	int predictedLabel2 = model2->predict(testSample);

	// ����һ�ֵ��÷�ʽ�����Ի�ȡ���ͬʱ�õ���ֵ:  
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