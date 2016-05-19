#include <QCoreApplication>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QDebug>
#include "in_out.h"

Input::Input()
{
}

bool Input::init(int argc, char *argv[], TrackObject *tracker)
{
    ifs.open(argv[2]);
    tracker->dist_coef = atof(argv[3]);
    tracker->tresh1 = atof(argv[4]);
    tracker->tresh2 = atof(argv[5]);

    if(atoi(argv[7])==1)
        show = true;
    else show = false;

    frame_counter = 0;
    select_flag = 1;

    if(cap.open(argv[1])){
        select_flag = 1;
        videofile = true;
        return true;
    } else {
        videofile = false;
        name = std::string(argv[1]) + "\\00000001.png";
        img = cv::imread(name);
        if(img.data==NULL){
            name = std::string(argv[1]) + "\\00000001.jpg";
            img = cv::imread(name);
            if(img.data==NULL)
                return false;
        }
    }

    return true;
}

bool Input::getCoor ()
{
    std::string buf;
    std::getline(ifs, buf);

    char * pch = strtok((char*)buf.c_str(), ",");
    float coor[4];
    int i = 0;

    while (pch!=NULL)
    {
        if((strcmp("NaN", pch)==0) || (strcmp("Nan", pch)==0) || (strcmp("nan", pch)==0)) {
            return false;
            break;
        }

        float num = std::stof(pch, NULL);
        coor[i] = num;
        pch=strtok(NULL, ",");
        i++;
    }

    p1 = cv::Point2f(coor[0], coor[1]);
    p2 = cv::Point2f((coor[0]+coor[2]), (coor[1]+coor[3]));

    return true;
}

bool Input::read()
{
    if(videofile)
        return cap.read(img);

    frame_counter++;
    char str[8];
    itoa(frame_counter, str, 10);
    std::string frame_num(str);
    int cursor = name.length() - 4 - frame_num.length();
    name.replace(cursor, frame_num.length(), frame_num);
    img = cv::imread(name);
    if(img.data==NULL) return false;
}
