#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <string>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier{
    public:
        Classifier(const string& model_file,
                   const string& trained_file,
                   const string& label_file,
                   const string& detector_file);

        void Execute(const cv::Mat& img);
  
    private:
        std::vector<float> Predict(const cv::Mat& img);

        void WrapInputLayer(std::vector<cv::Mat>* input_channels);

        void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

        std::vector<Prediction> Classify(std::vector<float> output_probs, int N = 7);
                        
        std::vector<cv::Rect> detectFace(const cv::Mat& img);
  
    private:
        shared_ptr<Net> net_;
        cv::Size input_geometry_;
        int num_channels_;
        std::vector<string> labels_;
        cv::CascadeClassifier detector_;
};

/* Face detection by cascade classifier */
std::vector<cv::Rect> Classifier::detectFace(const cv::Mat& img){
    std::vector<cv::Rect> faces;
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY );
    detector_.detectMultiScale(gray, faces, 1.3, 5);
    return faces;
}

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& label_file,
                       const string& detector_file){
    Caffe::set_mode(Caffe::GPU);
    /* Load the network. */
    net_.reset(new Net(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load labels. */
    std::ifstream labels(label_file.c_str());
    CHECK(labels) << "Unable to open labels file " << label_file;
    string line;
    while (std::getline(labels, line))
    labels_.push_back(string(line));

    Blob* output_layer = net_->output_blobs()[0];
    CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";

    /* Load face detector. */
    CHECK(detector_.load(detector_file)) << "Unable to load face detector from " << detector_file;
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N){
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

/* Core function that processes input image and visualizes result */
void Classifier::Execute(const cv::Mat& img){
    cv::Mat img2 = img.clone();
    int offset_x = 20, offset_y = 40;
    std::vector<cv::Rect> faces = detectFace(img);
    int max_len_prob = 120;
    
    /* Process all detected faces */
    for(std::vector<cv::Rect>::const_iterator r = faces.begin(); r != faces.end(); ++r) {
        cv::rectangle(img2, cv::Point(r->x, r->y), cv::Point(r->x+r->width, r->y+r->height),
                      cv::Scalar(0,255,0));
        cv::Mat face;
        cv::cvtColor(img, face, cv::COLOR_BGR2GRAY);
        face = face(cv::Rect(r->x-offset_x,r->y-offset_y,
                             r->width+2*offset_x,r->height+2*offset_y));
        std::vector<float> output = Predict(face);
        std::vector<Prediction> predictions = Classify(output, labels_.size());
        cv::putText(img2, predictions[0].first, cv::Point(r->x,r->y-12),
                    cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,0,255), 1.4);

        /* Visualize output probs */
        for(size_t i = 0; i < output.size(); ++i){
            cv::putText(img2, labels_[i], cv::Point(r->x+r->width+10,r->y+i*25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1.4);
            cv::line(img2, cv::Point(r->x+r->width+90,r->y-6+i*25),
                           cv::Point(r->x+r->width+90+int(max_len_prob*output[i]),r->y-6+i*25),
                           cv::Scalar(0,255,0), 3, 8, 0);
        }
    }
    cv::imshow("image", img2);
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(std::vector<float> output_probs, int N){
    N = std::min<int>(labels_.size(), N);
    std::vector<int> maxN = Argmax(output_probs, N);
    std::vector<Prediction> predictions;
    for (int i = 0; i < N; ++i){
        int idx = maxN[i];
        predictions.push_back(std::make_pair(labels_[idx], output_probs[idx]));
    }
    return predictions;
}

std::vector<float> Classifier::Predict(const cv::Mat& img){
    Blob* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data<float>();
    const float* end = begin + output_layer->channels();
    return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data<float>();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);
    
    sample_float = (sample_float/255-0.5)*2.0;
    
    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    
    cv::split(sample_float, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data<float>())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    cv::CommandLineParser parser(argc, argv,
        "{net-proto|fer2013_mini_XCEPTION.102-0.66/deploy.prototxt|}"
        "{net-weight|fer2013_mini_XCEPTION.102-0.66/weights.caffemodel|}"
        "{net-labels|fer2013_mini_XCEPTION.102-0.66/labels.txt|}"
        "{detector|haarcascade_frontalface_alt.xml|}"
        "{video||}"
        "{image||}"
    );
    /* Parsing arguments */
    string model_file    = parser.get<string>("net-proto");
    string trained_file  = parser.get<string>("net-weight");
    string label_file    = parser.get<string>("net-labels");
    string detector_file = parser.get<string>("detector");
    string image_file    = parser.get<string>("image");
    string video_file    = parser.get<string>("video");
    
    /* Input check */
    if(image_file.empty() && video_file.empty())
    {
        std::cout << "ERROR: Empty input" << std::endl;
        return -1;
    }

    /* Model construction */
    Classifier classifier(model_file, trained_file, label_file, detector_file);

    cv::namedWindow("image", cv::WINDOW_AUTOSIZE);

    /* Video processing */
    if(!video_file.empty())
    {
        cv::VideoCapture cap(video_file);
        if(!cap.isOpened())
        {
            std::cout << "ERROR: Could not open " << video_file << std::endl;
            cap.release();
            cv::destroyAllWindows();
            return -1;
        }

        cv::Mat frame;
        for(;;)
        {
            cap >> frame;
            if(frame.empty()) break;
            classifier.Execute(frame);
            if(cv::waitKey(15) >= 0) break;
        }
        cap.release();
    }
    
    /* Image processing */
    if(!image_file.empty())
    {
        cv::Mat img = cv::imread(image_file, -1);
        if(img.empty())
        {
            std::cout << "ERROR: Could not open " << image_file << std::endl;
            cv::destroyAllWindows();
            return -1;
        }
        classifier.Execute(img);
        cv::waitKey(0);
    }
    cv::destroyAllWindows();
    return 0;
}
