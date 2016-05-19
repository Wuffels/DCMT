#ifndef TRACKOBJECT_H
#define TRACKOBJECT_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <queue>


#define MAX_POINTS 300
#define INF 10000000.0


class TrackObject
{
public:
    int counter;
    int d, a;
    int start_weigth;
    float dist_coef;
    float point_avg_dist;
    float tresh1, tresh2;

    std::string detectorType;
    std::string descriptorType;
    std::string matcherType;
    int descriptorLength;
    int thrOutlier;
    float thrConf;
    float thrRatio;

    bool estimateScale;
    bool estimateRotation;


    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
    cv::Ptr<cv::DescriptorMatcher> descriptorMatcher;

    std::vector<bool> *isActive;

    int backgroundPointsNum;
    cv::Mat selectedFeatures;
    std::vector<int> *selectedClasses;
    cv::Mat featuresDatabase;

    cv::Mat backgroundDatabase;

    std::vector<int> classesDatabase;

    std::vector<int> *weigthsDatabase;

    std::vector<std::vector<float>* > squareForm;
    std::vector<std::vector<float>* > angles;
    cv::Point2f center;

    cv::Point2f topLeft;
    cv::Point2f topRight;
    cv::Point2f bottomRight;
    cv::Point2f bottomLeft;

    cv::Rect_<float> boundingbox;
    bool hasResult;

    cv::Point2f centerToTopLeft;
    cv::Point2f centerToTopRight;
    cv::Point2f centerToBottomRight;
    cv::Point2f centerToBottomLeft;

    std::vector<cv::Point2f> *springs;

    cv::Mat im_prev;
    std::vector<std::pair<cv::KeyPoint,int> > activeKeypoints;
    std::vector<std::pair<cv::KeyPoint,int> > trackedKeypoints;

    std::vector<cv::KeyPoint> otherKeypoints;
    std::vector<std::pair <cv::KeyPoint, int> > notMatchedKeypoints;//

    unsigned int nbInitialKeypoints;

    std::vector<cv::Point2f> votes;

    TrackObject();
    ~TrackObject();
    cv::Mat getAverage(cv::Mat arr);
    float D (cv::Mat arr) ;
    void recalcCenter(float scaleEstimate, float rotationEstimate);
    void addPoint(cv::KeyPoint point, cv::Mat descriptor);
    void dropPoint(int idx);
    void initialise(cv::Mat im_gray0, cv::Point2f topleft, cv::Point2f bottomright);
    void estimate(const std::vector<std::pair<cv::KeyPoint, int> >& keypointsIN, float& scaleEstimate, float& medRot, std::vector<std::pair<cv::KeyPoint, int> >& keypoints);
    //void TrackObject::estimate(const std::vector<std::pair<cv::KeyPoint, int> >& keypointsIN, cv::Point2f& center, float& scaleEstimate, float& medRot, std::vector<std::pair<cv::KeyPoint, int> >& keypoints);
std::vector<std::pair<cv::KeyPoint, int> > outliers;

    void processFrame(cv::Mat im_gray);
    void addFlags(std::vector<std::pair<cv::KeyPoint,int> > activeKeypoints);
    void drawWeigthsMap(cv::Mat im_gray, std::vector<cv::KeyPoint> keypoints, std::vector<std::vector<cv::DMatch> > selectedMatchesAll, int tau);
};
class Cluster
{
public:
    int first, second;//cluster id
    float dist;
    int num;
};

void inout_rect(const std::vector<cv::KeyPoint>& keypoints, cv::Point2f topleft, cv::Point2f bottomright, std::vector<cv::KeyPoint>& in, std::vector<cv::KeyPoint>& out);
void track(cv::Mat im_prev, cv::Mat im_gray, const std::vector<std::pair<cv::KeyPoint, int> >& keypointsIN, std::vector<std::pair<cv::KeyPoint, int> >& keypointsTracked, std::vector<unsigned char>& status, int THR_FB = 20);
cv::Point2f rotate(cv::Point2f p, float rad);
#endif // TRACKOBJECT_H
