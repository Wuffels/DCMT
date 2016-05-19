#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <stdio.h>
#include <iostream>
#include <queue>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

void inout_rect(const std::vector<cv::KeyPoint>& keypoints, cv::Point2f topleft, cv::Point2f bottomright, std::vector<cv::KeyPoint>& in, std::vector<cv::KeyPoint>& out);
void track(cv::Mat im_prev, cv::Mat im_gray, const std::vector<std::pair<cv::KeyPoint, int> >& keypointsIN, std::vector<std::pair<cv::KeyPoint, int> >& keypointsTracked, std::vector<unsigned char>& status);
cv::Point2f rotate(cv::Point2f p, float rad);
bool compare_response(const cv::KeyPoint &first, const cv::KeyPoint &second);
int Count(unsigned char a);
int calcDist (cv::Mat avg_object, cv::Mat avg_background);
float fris (float d1, float d2);
typedef std::pair<int,int> PairInt;
typedef std::pair<float,int> PairFloat;
template<typename T>
bool comparatorPair ( const std::pair<T,int>& l, const std::pair<T,int>& r)
{
    return l.first < r.first;
}
template<typename T>
bool comparatorPairDesc ( const std::pair<T,int>& l, const std::pair<T,int>& r)
{
    return l.first > r.first;
}
template <typename T>
T sign(T t)
{
    if( t == 0 )
        return T(0);
    else
        return (t < 0) ? T(-1) : T(1);
}
template<typename T>
T median(std::vector<T> list)
{
    T val;
    std::nth_element(&list[0], &list[0]+list.size()/2, &list[0]+list.size());
    val = list[list.size()/2];
    if(list.size()%2==0)
    {
        std::nth_element(&list[0], &list[0]+list.size()/2-1, &list[0]+list.size());
        val = (val+list[list.size()/2-1])/2;
    }
    return val;
}
bool comparatorPairSecond(const std::pair<int, int>& l, const std::pair<int, int>& r);
std::vector<int> argSortInt(const std::vector<int>& list);
std::vector<int> findPoints(std::vector<std::vector<int> > pairs, int idx);
std::vector<int> calcHist(std::vector<cv::Point2f> votes, int treshold);
std::vector<bool> in1d(const std::vector<int>& a, const std::vector<int>& b);
cv::Mat getAverage(cv::Mat arr, int descriptorLength);

#endif // FUNCTIONS_H

