#include "functions.h"
#define THR_FB 20

void inout_rect(const std::vector<cv::KeyPoint>& keypoints, cv::Point2f topleft, cv::Point2f bottomright,
                std::vector<cv::KeyPoint>& in, std::vector<cv::KeyPoint>& out)
{
    for(unsigned int i = 0; i < keypoints.size(); i++)
    {
        if(keypoints[i].pt.x > topleft.x && keypoints[i].pt.y > topleft.y && keypoints[i].pt.x < bottomright.x && keypoints[i].pt.y < bottomright.y)
            in.push_back(keypoints[i]);
        else out.push_back(keypoints[i]);
    }
}

void track(cv::Mat im_prev, cv::Mat im_gray, const std::vector<std::pair<cv::KeyPoint, int> >& keypointsIN,
           std::vector<std::pair<cv::KeyPoint, int> >& keypointsTracked, std::vector<unsigned char>& status)
{
    //Status of tracked keypoint - True means successfully tracked
    status = std::vector<unsigned char>();

    //If at least one keypoint is active
    if(keypointsIN.size() > 0)
    {
        std::vector<cv::Point2f> pts;
        std::vector<cv::Point2f> pts_back;
        std::vector<cv::Point2f> nextPts;
        std::vector<unsigned char> status_back;
        std::vector<float> err;
        std::vector<float> err_back;
        std::vector<float> fb_err;
        for(unsigned int i = 0; i < keypointsIN.size(); i++)
            pts.push_back(cv::Point2f(keypointsIN[i].first.pt.x,keypointsIN[i].first.pt.y));

        //Calculate forward optical flow for prev_location
        cv::calcOpticalFlowPyrLK(im_prev, im_gray, pts, nextPts, status, err);
        //Calculate backward optical flow for prev_location
        cv::calcOpticalFlowPyrLK(im_gray, im_prev, nextPts, pts_back, status_back, err_back);

        //Calculate forward-backward error
        for(unsigned int i = 0; i < pts.size(); i++)
        {
            cv::Point2f v = pts_back[i]-pts[i];
            fb_err.push_back(sqrt(v.dot(v)));
        }

        //Set status depending on fb_err and lk error
        for(unsigned int i = 0; i < status.size(); i++)
            status[i] = (fb_err[i] <= THR_FB) & status[i];

        keypointsTracked = std::vector<std::pair<cv::KeyPoint, int> >();
        for(unsigned int i = 0; i < pts.size(); i++)
        {
            std::pair<cv::KeyPoint, int> p = keypointsIN[i];
            if(status[i])
            {
                p.first.pt = nextPts[i];
                keypointsTracked.push_back(p);
            }
        }
    }
    else keypointsTracked = std::vector<std::pair<cv::KeyPoint, int> >();
}

cv::Point2f rotate(cv::Point2f p, float rad)
{
    if(rad == 0)
        return p;
    float s = sin(rad);
    float c = cos(rad);
    return cv::Point2f(c*p.x-s*p.y,s*p.x+c*p.y);
}
bool compare_response(const cv::KeyPoint &first, const cv::KeyPoint &second)
{
    return (first.response > second.response);
}
int Count(unsigned char a)
{
   int count=0;
   while(a)
    {
      count += a&0x01;
      a /= 2;
    }
 return   count;
}
int calcDist (cv::Mat avg_object, cv::Mat avg_background) {
    int dist = 0;
    for(int j = 0; j < avg_object.cols; j++) {
        unsigned char val = (avg_object.at<unsigned char> (0, j)) ^ (avg_background.at<unsigned char> (0, j));
        dist += Count(val);
    }
    return dist;
}
float fris (float d1, float d2)
{
//    float x = d1/(d1+d2);
//    float t = 0.05;
//    float w = t*std::sin(4*3.1415*x);
//    float f = 1-2*x+w;
//    return f;
    return (d1-d2)/(d1+d2);
}
typedef std::pair<int,int> PairInt;
typedef std::pair<float,int> PairFloat;
bool comparatorPairSecond(const std::pair<int, int>& l, const std::pair<int, int>& r)
{
    return l.second < r.second;
}
std::vector<int> argSortInt(const std::vector<int>& list)
{
    std::vector<int> result(list.size());
    std::vector<std::pair<int, int> > pairList(list.size());
    for(unsigned int i = 0; i < list.size(); i++)
        pairList[i] = std::make_pair(i, list[i]);
    std::sort(&pairList[0], &pairList[0]+pairList.size(), comparatorPairSecond);
    for(unsigned int i = 0; i < list.size(); i++)
        result[i] = pairList[i].first;
    return result;
}
std::vector<int> findPoints(std::vector<std::vector<int> > pairs, int idx)
{
    std::vector<bool> used(pairs.size(), false);
    std::vector <int> findedPoints;
    std::queue<int> nextPoints;
    for(unsigned int i = 0; i < pairs.at(idx).size(); i++) {
        int p = pairs[idx][i];
        nextPoints.push(p);
        findedPoints.push_back(p);
        used[p] = true;
    }
    while(!nextPoints.empty()) {
        int p = nextPoints.front();
        nextPoints.pop();
        std::vector<int> points = pairs[p];
        for(unsigned int i = 0; i < points.size(); i++) {
            int point = points[i];
            if(!(used[point])){
                nextPoints.push(point);
                findedPoints.push_back(point);
                used[point] = true;
            }
        }
    }
    return findedPoints;
}
std::vector<int> calcHist(std::vector<cv::Point2f> votes, int treshold)
{
    std::vector<std::vector<int> > pairs;
    std::vector<int> hist;
    int max = 0;
    unsigned int idx = 0;

    for(unsigned int i = 0; i < votes.size(); i++) {
        hist.push_back(0);
        for(unsigned int j = 0; j < votes.size(); j++) {
            cv::Point2f p = votes[i]-votes[j];
            float dist = sqrt(p.dot(p));
            if(dist < treshold) {
                hist.at(i)++;
                while(pairs.size() <= i) {
                    std::vector <int> vect;
                    pairs.push_back(vect);
                }
                pairs.at(i).push_back(j);
            }
        }
    }

    for(unsigned int i = 0; i < hist.size(); i++) {
        int num = hist.at(i) ;
        if(num > max) {
            max = num;
            idx = i;
        }
    }
    hist.clear();
    if((pairs.size())==0) return hist;
    return findPoints(pairs,idx);
}

//todo : n*log(n) by sorting the second array and dichotomic search instead of n^2
std::vector<bool> in1d(const std::vector<int>& a, const std::vector<int>& b)
{
    std::vector<bool> result;
    for(unsigned int i = 0; i < a.size(); i++)
    {
        bool found = false;
        for(unsigned int j = 0; j < b.size(); j++)
            if(a[i] == b[j])
            {
                found = true;
                break;
            }
        result.push_back(found);
    }
    return result;
}
cv::Mat getAverage(cv::Mat arr, int descriptorLength)
{
    cv::Mat res(arr.row(0).clone());
    std::vector<int> count(descriptorLength, 0);
    int counter = 0;
    for (int i = 0; i < arr.rows; i++) {
        for (int j = 0; j < arr.cols; j++) {
            unsigned char val = (unsigned char) (arr.at<unsigned char> (i, j));
            for(int k = 0; k < 8; k++) {
                unsigned char binVal = val & 0x1;
                val >>= 1;
                count[counter + 8 - k - 1] += (int)binVal;
            }
            counter+=8;
        }
        counter = 0;
    }

    for(int i = 0; i < descriptorLength; i++) {
        if(count[i] >= ((arr.rows+1)/2))
            count[i] = 1;
        else count[i] = 0;
    }
    counter = 0;

    for(int i = 0; i < descriptorLength; i+=8) {
        res.at<unsigned char> (0, counter) = 0;
        res.row(0).col(counter) |= (count[i+7]);
        res.row(0).col(counter) |= (count[i+6]*2);
        res.row(0).col(counter) |= (count[i+5]*4);
        res.row(0).col(counter) |= (count[i+4]*8);
        res.row(0).col(counter) |= (count[i+3]*16);
        res.row(0).col(counter) |= (count[i+2]*32);
        res.row(0).col(counter) |= (count[i+1]*64);
        res.row(0).col(counter) |= (count[i]*128);
        unsigned char r = (unsigned char)res.at<unsigned char> (0, counter);
        counter++;
    }

    return res;
}
