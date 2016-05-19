#include "trackobject.h"
#include "functions.h"
#include "track_clear_function.h"
#include <stdio.h>
#define _USE_MATH_DEFINES
#define MAX_WEIGTH 200
#define ADD_WEIGTH 1
#include <iostream>
#include <cmath>
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

TrackObject::TrackObject()
{
    a = 0;
    d = 0;
    counter = 0;

    detectorType = "BRISK";
    descriptorType = "BRISK";
    matcherType = "BruteForce-Hamming";
    thrOutlier = 20;
    thrConf = 0.75;
    thrRatio = 0.8;
    descriptorLength = 512;
    estimateScale = true;
    estimateRotation = true;
    nbInitialKeypoints = 0;

    selectedClasses = new std::vector<int> (MAX_POINTS);

    weigthsDatabase = new std::vector<int>(MAX_POINTS, -1);

    for (int i = 0; i < MAX_POINTS; i++) {
        angles.push_back(new std::vector<float>(MAX_POINTS));
        squareForm.push_back(new std::vector<float>(MAX_POINTS));
        selectedClasses->at(i) = i + 1;
    }

    springs = new std::vector<cv::Point2f> (MAX_POINTS, cv::Point2f(INF, INF));

    isActive = new std::vector<bool> (MAX_POINTS, false);
}
TrackObject::~TrackObject()
{
    delete selectedClasses;

    delete weigthsDatabase;

    for (int i = 0; i < MAX_POINTS; i++) {
        delete angles.at(i);
        delete squareForm.at(i);
    }

    delete springs;

    delete isActive;
}

void TrackObject::recalcCenter(float scaleEstimate, float rotationEstimate)
{
    recalcCenter_(scaleEstimate, rotationEstimate, isActive, topLeft, topRight, bottomRight, bottomLeft, springs, center,
                  centerToTopLeft, centerToTopRight, centerToBottomRight, centerToBottomLeft);
}
void TrackObject::addPoint(cv::KeyPoint point, cv::Mat descriptor)
{
    addPoint_(point, descriptor,isActive, backgroundPointsNum, springs,center, featuresDatabase, selectedFeatures, squareForm, angles,
              activeKeypoints, weigthsDatabase);
}
void TrackObject::dropPoint(int idx)
{
    dropPoint_(idx, isActive, backgroundPointsNum, featuresDatabase, selectedFeatures);
}

//float TrackObject::D (cv::Mat arr) {
//    cv::Mat avg = getAverage(arr.clone());
//    int counter = 0;
//    for (int i = 0; i < arr.rows; i++) {
//        for (int j = 0; j < arr.cols; j++) {
//            unsigned char val = (arr.at<unsigned char> (i, j));
//            unsigned char c = val ^ (avg.at<unsigned char> (0, j));
//            counter += Count(c);
//        }
//    }
//    return (float)counter/arr.rows;
//}

void TrackObject::initialise(cv::Mat im_gray0, cv::Point2f topleft, cv::Point2f bottomright)
{
    initialise_(detector, descriptorExtractor , descriptorMatcher,
             detectorType, matcherType,  backgroundPointsNum,  descriptorLength,
             selectedFeatures, selectedClasses,  featuresDatabase,
             backgroundDatabase, classesDatabase, weigthsDatabase,
             im_gray0, topleft, bottomright, point_avg_dist, nbInitialKeypoints,
             squareForm, angles, activeKeypoints,
             isActive, springs, center, im_prev,
             centerToTopLeft, centerToTopRight, centerToBottomRight, centerToBottomLeft);
}

void TrackObject::estimate(const std::vector<std::pair<cv::KeyPoint, int> >& keypointsIN, float& scaleEstimate, float& medRot, std::vector<std::pair<cv::KeyPoint, int> >& keypoints)
{
    estimate_(keypointsIN, scaleEstimate, medRot, keypoints, center, thrOutlier,
              squareForm, angles, estimateScale, estimateRotation, springs);
}

void TrackObject::processFrame(cv::Mat im_gray)
{
    processFrame_(im_gray, detector, descriptorExtractor,
                       descriptorMatcher, backgroundPointsNum, descriptorLength,
                       selectedFeatures, selectedClasses, featuresDatabase,
                       classesDatabase, weigthsDatabase,
                       point_avg_dist, nbInitialKeypoints,
                       squareForm, angles,
                       activeKeypoints,
                       isActive, springs, center, im_prev,
                       centerToTopLeft, centerToTopRight, centerToBottomRight, centerToBottomLeft,
                       thrOutlier, thrConf, thrRatio, estimateScale, estimateRotation, boundingbox,
                       hasResult, dist_coef, tresh1, tresh2,
                       topLeft, topRight, bottomRight, bottomLeft);
}
