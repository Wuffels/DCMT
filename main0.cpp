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
#include "trackobject.h"


cv::Point point1,point2;
cv::Rect rect;
cv::Mat img;
int select_flag,drag;

void mouseHandler(int event, int x, int y, int flags, void* param)
{
    if (event == CV_EVENT_LBUTTONDOWN && !drag)
    {
        /* left button clicked. ROI selection begins */
        point1 = cv::Point(x, y);
        drag = 1;
    }

    if (event == CV_EVENT_MOUSEMOVE && drag)
    {
        /* mouse dragged. ROI being selected */
        cv::Mat img1 = img.clone();
        point2 = cv::Point(x, y);
        cv::rectangle(img1, point1, point2, CV_RGB(255, 0, 0), 3, 8, 0);
        cv::imshow("XXX", img1);
    }

    if (event == CV_EVENT_LBUTTONUP && drag)
    {
        point2 = cv::Point(x, y);
        rect = cv::Rect(point1.x,point1.y,x-point1.x,y-point1.y);
        drag = 0;
       // roiImg = img(rect);
    }

    if (event == CV_EVENT_LBUTTONUP)
    {
       /* ROI selected */
        select_flag = 1;
        drag = 0;
    }
}
//параметры на вход: имя_программы.exe имя_файла(папки)_где_видео имя_файла_с_координатами.txt dist tresh1 tresh2 show
//координаты в файле  - левый верхний угол и длины сторон
//dist - порог на геом расстояние
//tresh1 и tresh2 - промежуток для FRiS-функции
//show - 0 - не показывать кадр, 1 - показывать
int main(int argC, char*argV[])
{
//    printf("@%lf\n", cv::getTickFrequency());
//    return 0;
    //cv::VideoWriter out("D:/res_weigths.avi",CV_FOURCC('D','I','V','X'),20,cv::Size(290,217));
    TrackObject cmt;
    Input in;
    if(!in.init(argC, argV, &cmt)) {
        printf("Camera not opened\r\n");
        return 0;
    }

    select_flag = 0;
    drag = 0;
    cmt.counter = 0;

    std::ofstream fout;

    fout.open(argV[6],std::ios::app);

    double avg_dist = 0;
    double avg_coef=0;
    int not_found_counter = 0;
    int frame_counter = 0;
    long t1 = 0, t2 = 0;

    do{
        cv::Mat image,gray;
        if(!in.read())
            break;
        image = in.img;
        frame_counter++;
        cv::cvtColor(image,gray, CV_RGB2GRAY);

        if (in.select_flag == 1) {
            if(!in.getCoor()){
                printf("Incorrect filename\r\n");
                return 0;
            }
            //t1=cv::getTickCount();
            cmt.initialise(gray,in.p1,in.p2);
            //t2=cv::getTickCount();
            //printf("!%ld\n", t2-t1);
            if(cmt.featuresDatabase.rows == 0) {
                in.select_flag = 0;
                continue;
            }
            in.select_flag = 2;
        } else if (in.select_flag == 2) {
            //t1=cv::getTickCount();
                cmt.processFrame(gray);
               // t2=cv::getTickCount();
                //printf("%ld\n", t2-t1);

                for(unsigned int i = 0; i<cmt.activeKeypoints.size(); i++)
                    cv::circle(gray, cmt.activeKeypoints[i].first.pt, 3, cv::Scalar(255,255,255));

                cv::line(gray, cmt.topLeft, cmt.topRight, cv::Scalar(255,255,255));
                cv::line(gray, cmt.topRight, cmt.bottomRight, cv::Scalar(255,255,255));
                cv::line(gray, cmt.bottomRight, cmt.bottomLeft, cv::Scalar(255,255,255));
                cv::line(gray, cmt.bottomLeft, cmt.topLeft, cv::Scalar(255,255,255));

                if(in.getCoor()) {
                    //cv::rectangle(gray, in.p1 , in.p2, cv::Scalar(0,0,0));
                    cmt.counter++;
                    if(!cmt.hasResult) {
                        not_found_counter++;
                    } else {
                        float top, bottom, left, right;
                        cv::Point2f pp1 = in.p1, pp2 = in.p2;
//                        float l, r, b, t;
//                        l = cmt.topLeft.x;
//                        r = cmt.topRight.x;
//                        t = cmt.topRight.y;
//                        b = cmt.bottomLeft.y;
//                        left = std::max(cmt.topLeft.x, in.p1.x);
//                        right = std::min(cmt.topRight.x, in.p2.x);
//                        bottom = std::min(cmt.bottomLeft.y, in.p2.y);
//                        top = std::max(cmt.topRight.y, in.p1.y);

                        //вычисление минимальног не повернутого bounding box, вмещающего в себя bbox, полученный алгоритмом
                        float l, l1, l2, r, r1, r2, b, b1, b2, t, t1, t2;
                        l1 = std::min(cmt.topLeft.x, cmt.topRight.x);
                        l2 = std::min(cmt.bottomLeft.x, cmt.bottomRight.x);
                        l = std::min(l1,l2);

                        r1 = std::max(cmt.topLeft.x, cmt.topRight.x);
                        r2 = std::max(cmt.bottomLeft.x, cmt.bottomRight.x);
                        r = std::max(r1,r2);

                        t1 = std::min(cmt.topLeft.y, cmt.topRight.y);
                        t2 = std::min(cmt.bottomLeft.y, cmt.bottomRight.y);
                        t = std::min(t1,t2);

                        b1 = std::max(cmt.topLeft.y, cmt.topRight.y);
                        b2 = std::max(cmt.bottomLeft.y, cmt.bottomRight.y);
                        b = std::max(b1,b2);

                        cv::rectangle(gray, cv::Point2f(l, t), cv::Point2f(r, b), cv::Scalar(0,0,0));

                        //вычисление площади пересечения
                        left = std::max(l, in.p1.x);
                        right = std::min(r, in.p2.x);
                        bottom = std::min(b, in.p2.y);
                        top = std::max(t, in.p1.y);

                        float diffS = 0;
                        if(right > left && bottom > top){
                            diffS = (right-left)*(bottom-top);
                        }

                        not_found_counter += (diffS==0)?1:0;

                        //float boxS = (in.p2.x-in.p1.x)*(in.p2.y-in.p1.y) + (cmt.topRight.x - cmt.topLeft.x)*(cmt.bottomLeft.y - cmt.topLeft.y) - diffS;
                        float boxS = 0;
                        if(r > l && b > t){
                            boxS = (r-l)*(b-t);
                        }
                        float allS = (in.p2.x-in.p1.x)*(in.p2.y-in.p1.y) + boxS - diffS;
                        avg_coef += (diffS/allS);
                         float center_x1 = (in.p2.x-in.p1.x)/2;
                         float center_y1 = (in.p2.y-in.p1.y)/2;
                         float center_x2 = (cmt.topRight.x - cmt.topLeft.x)/2;
                         float center_y2 = (cmt.bottomLeft.y - cmt.topLeft.y)/2;
                         float dx = center_x1-center_x2;
                         float dy = center_y1-center_y2;
                         double dist = sqrt(pow(dx,2)+pow(dy,2));
                         avg_dist+=dist;
                    }
                }

                //printf("%d\n", cmt.counter);
                //printf("%d\n", cmt.counter);
//                cv::Mat weigths(gray.rows, gray.cols, CV_8UC1);
//                cv::Mat weigths_color;
//                cv::cvtColor(gray,weigths_color,CV_GRAY2RGB);
//                weigths.setTo(0);
//                for(int i = 0; i<cmt.isActive->size(); i++) {
//                    if(cmt.isActive->at(i)) {
//                        cv::Point2f p = cmt.center + cmt.springs->at(i);
//                        int x = p.x;
//                        int y = p.y;
//                        if(cmt.weigthsDatabase->at(i) >= 0) {
//                            //cv::circle(weigths, cv::Point2d(x,y), 3, cmt.weigthsDatabase->at(i) + 55, -3);
//                            cv::circle(weigths_color, cv::Point2d(x,y), 3, cv::Scalar(0,cmt.weigthsDatabase->at(i) + 55,0), -3);
//                            //weigths.col(x).row(y) = cmt.weigthsDatabase->at(i) + 55;
//                        }
//                    }
//                }
////                for(unsigned int i = 0; i<cmt.activeKeypoints.size(); i++)
////                    cv::circle(weigths_color, cmt.activeKeypoints[i].first.pt, 3, cv::Scalar(255,255,255));
//                cv::imshow("weigths", weigths_color);
//                out.write(weigths_color);
                if(in.show) cv::imshow("XXX",gray);
            } else {
                cv::imshow("XXX",gray);
            }
    } while(27 != cv::waitKey(1));

    fout << std::endl << argV[1] << std::endl;
    fout << "_________________"<<std::endl << cmt.dist_coef << " " << cmt.tresh1 << " " << cmt.tresh2 << std::endl << "_________________"<< std::endl;;
    fout << (float)(cmt.counter - not_found_counter)/cmt.counter << std::endl;
    fout << (avg_dist/(cmt.counter - not_found_counter)) << std::endl;
    fout << (float)((avg_coef)/cmt.counter) << std::endl;
    fout << (cmt.a) << std::endl << cmt.d << std::endl;
    fout.close();
    //printf("\n\n!!!!!! %d %d\n", cmt.a, cmt.d);
    //printf("%d %d\n", cmt.counter, not_found_counter);
    return 0;
}

