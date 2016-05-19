#include <stdio.h>
#include <iostream>
#include <queue>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "functions.h"
#define _USE_MATH_DEFINES
#define MAX_WEIGTH 200
#define ADD_WEIGTH 1
#define MAX_POINTS 300
#define INF 10000000.0
#if __cplusplus < 201103L //test if c++11

    #include <limits>

    #ifndef NAN
    //may not be correct on all compilator, DON'T USE the flag FFAST-MATH

        #define NAN std::numeric_limits<float>::quiet_NaN()

        template <typename T>
        bool isnan(T d)
        {
          return d != d;
        }
    #endif

#endif

void fillModelParams (cv::Point2f &topleft, cv::Point2f &bottomright, float &point_avg_dist,
                            std::vector<std::vector<float>* > squareForm, std::vector<std::vector<float>* > angles,
                            std::vector<cv::Point2f> *springs, cv::Point2f &center, std::vector<cv::KeyPoint> &selected_keypoints,
                            cv::Point2f &centerToTopLeft, cv::Point2f &centerToTopRight, cv::Point2f &centerToBottomRight, cv::Point2f &centerToBottomLeft);

void fillDB (cv::Mat &selectedFeatures, std::vector<int> *selectedClasses, cv::Mat &featuresDatabase,
                     cv::Mat &backgroundDatabase, std::vector<int> &classesDatabase,
                     int backgroundPointsNum);
void recalcCenter_(float scaleEstimate, float rotationEstimate,  std::vector<bool> *isActive,
                    cv::Point2f &topLeft, cv::Point2f &topRight, cv::Point2f &bottomRight, cv::Point2f &bottomLeft,
                    std::vector<cv::Point2f> *springs, cv::Point2f &center,
                    cv::Point2f &centerToTopLeft, cv::Point2f &centerToTopRight, cv::Point2f &centerToBottomRight, cv::Point2f &centerToBottomLeft);
void getActivePoints (cv::Mat &activeFeaturesDB, cv::Mat &activeSelectedFeatures, cv::Mat &backgrdDb, cv::Mat &featuresDatabase,
                              std::vector<int> &featuresIdxs, std::vector<int> &selectedfeaturesIdxs, std::vector<bool> *isActive, int backgroundPointsNum);
void deletePoints (int backgroundPointsNum, cv::Mat &selectedFeatures, cv::Mat &featuresDatabase,
                           std::vector<int> *weigthsDatabase, std::vector<std::pair<cv::KeyPoint,int> > &activeKeypoints,
                           std::vector<bool> *isActive, bool &hasResult);
void getBestPoints_(std::vector<cv::KeyPoint> &selected_keypoints);
void recalcCenter_(float scaleEstimate, float rotationEstimate,  std::vector<bool> *isActive,
                    cv::Point2f &topLeft, cv::Point2f &topRight, cv::Point2f &bottomRight, cv::Point2f &bottomLeft,
                    std::vector<cv::Point2f> *springs, cv::Point2f &center,
                    cv::Point2f &centerToTopLeft, cv::Point2f &centerToTopRight, cv::Point2f &centerToBottomRight, cv::Point2f &centerToBottomLeft);

void addPoint_(cv::KeyPoint &point, cv::Mat &descriptor, std::vector<bool> *isActive, int backgroundPointsNum,
               std::vector<cv::Point2f> *springs, cv::Point2f &center, cv::Mat &featuresDatabase, cv::Mat &selectedFeatures,
               std::vector<std::vector<float>* > squareForm, std::vector<std::vector<float>* > angles,
               std::vector<std::pair<cv::KeyPoint,int> > &activeKeypoints, std::vector<int> *weigthsDatabase);

void dropPoint_(int idx, std::vector<bool> *isActive, int backgroundPointsNum, cv::Mat &featuresDatabase, cv::Mat &selectedFeatures);

void estimate_(const std::vector<std::pair<cv::KeyPoint, int> >& keypointsIN, float& scaleEstimate, float& medRot,
               std::vector<std::pair<cv::KeyPoint, int> >& keypoints, cv::Point2f &center, int thrOutlier,
               std::vector<std::vector<float>* > squareForm, std::vector<std::vector<float>* > angles,
               bool estimateScale, bool estimateRotation, std::vector<cv::Point2f> *springs);

void initialise_(cv::Ptr<cv::FeatureDetector> &detector, cv::Ptr<cv::DescriptorExtractor> &descriptorExtractor ,cv::Ptr<cv::DescriptorMatcher> &descriptorMatcher,
                std::string &detectorType, std::string &matcherType, int &backgroundPointsNum, int descriptorLength,
                cv::Mat &selectedFeatures, std::vector<int> *selectedClasses, cv::Mat &featuresDatabase,
                cv::Mat &backgroundDatabase, std::vector<int> &classesDatabase, std::vector<int> *weigthsDatabase,
                cv::Mat &im_gray0, cv::Point2f &topleft, cv::Point2f &bottomright, float &point_avg_dist, unsigned int &nbInitialKeypoints,
                std::vector<std::vector<float>* > squareForm, std::vector<std::vector<float>* > angles,
                std::vector<std::pair<cv::KeyPoint,int> > &activeKeypoints,
                std::vector<bool> *isActive, std::vector<cv::Point2f> *springs, cv::Point2f &center, cv::Mat &im_prev,
                cv::Point2f &centerToTopLeft, cv::Point2f &centerToTopRight, cv::Point2f &centerToBottomRight, cv::Point2f &centerToBottomLeft);

void processFrame_(cv::Mat &im_gray, cv::Ptr<cv::FeatureDetector> &detector, cv::Ptr<cv::DescriptorExtractor> &descriptorExtractor,
                   cv::Ptr<cv::DescriptorMatcher> &descriptorMatcher, int &backgroundPointsNum, int descriptorLength,
                   cv::Mat &selectedFeatures, std::vector<int> *selectedClasses, cv::Mat &featuresDatabase,
                   std::vector<int> &classesDatabase, std::vector<int> *weigthsDatabase,
                   float &point_avg_dist, unsigned int &nbInitialKeypoints,
                   std::vector<std::vector<float>* > squareForm, std::vector<std::vector<float>* > angles, std::vector<cv::Point2f> &votes,
                   std::vector<std::pair<cv::KeyPoint,int> > &activeKeypoints,std::vector<std::pair<cv::KeyPoint,int> > &trackedKeypoints,
                   std::vector<bool> *isActive, std::vector<cv::Point2f> *springs, cv::Point2f &center, cv::Mat &im_prev,
                   cv::Point2f &centerToTopLeft, cv::Point2f &centerToTopRight, cv::Point2f &centerToBottomRight, cv::Point2f &centerToBottomLeft,
                   int thrOutlier, float thrConf, float thrRatio, bool estimateScale, bool estimateRotation, cv::Rect_<float> &boundingbox,
                   bool &hasResult, float dist_coef, float tresh1, float tresh2,
                   cv::Point2f &topLeft, cv::Point2f &topRight, cv::Point2f &bottomRight, cv::Point2f &bottomLeft);
