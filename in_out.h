#ifndef IN_OUT
#define IN_OUT


#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <queue>
#include "trackobject.h"
class Input
{
public:
    bool videofile, jpg, show;
    int x1, x2, y1, y2;
    int frame_counter;
    int select_flag;
    int cursor;
    cv::Point2f p1, p2;
    std::string name;
    cv::Mat img;
    cv::VideoCapture cap;
    std::ifstream ifs;
    std::ifstream ifs_orig;
    std::string buf;
    std::ofstream fout;
    Input();
    bool init(int argc, char* argv[], TrackObject *tracker);
    bool read();
    bool getCoor ();

};

#endif // IN_OUT
