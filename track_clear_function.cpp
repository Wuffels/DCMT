#include "track_clear_function.h"

//Функция recalcCenter_ пересчитывает центр объекта и все координаты объекта относительно центра
//ВХОД:
//float scaleEstimate, float rotationEstimate - масштаб и поворот объекта на текущем кадре
//std::vector<bool> *isActive - массив в котором указано под какими номерами хранятся точки объекта,
//а под какими ничего не хранится и можно добавить новые точки
//cv::Point2f &topLeft, cv::Point2f &topRight, cv::Point2f &bottomRight, cv::Point2f &bottomLeft -
//координаты углов bounding box
//std::vector<cv::Point2f> *springs - координаты точек объекта относительно центра объекта
//cv::Point2f &center -  координаты центра объекта
//cv::Point2f &centerToTopLeft, cv::Point2f &centerToTopRight, cv::Point2f &centerToBottomRight,
//cv::Point2f &centerToBottomLeft - координаты углов bounding box относительно центра объекта
//РЕЗУЛЬТАТ:
//Новые значения для
//topLeft, topRight, bottomRight, bottomLeft, springs, center,
//centerToTopLeft, сenterToTopRight, centerToBottomRight, centerToBottomLeft
//В соответствии с scaleEstimate, rotationEstimate, isActive
void recalcCenter_(float scaleEstimate, float rotationEstimate,  std::vector<bool> *isActive,
                    cv::Point2f &topLeft, cv::Point2f &topRight, cv::Point2f &bottomRight, cv::Point2f &bottomLeft,
                    std::vector<cv::Point2f> *springs, cv::Point2f &center,
                    cv::Point2f &centerToTopLeft, cv::Point2f &centerToTopRight, cv::Point2f &centerToBottomRight, cv::Point2f &centerToBottomLeft)
{
    //Find the new center of selected keypoints
    cv::Point2f newCenter(0,0);
    int activeNum = 0;

    for(unsigned int i = 0; i < MAX_POINTS; i++) {
        if(isActive->at(i) ) {
            newCenter += springs->at(i) + center;
            activeNum++;
        }
    }
    newCenter *= (1.0/activeNum);

    //Calculate springs of each keypoint
    for(unsigned int i = 0; i < MAX_POINTS; i++) {
        if(isActive->at(i)) {
            springs->at(i) = springs->at(i) + center - newCenter;
        }
    }
    center = newCenter;
    topLeft = center + scaleEstimate*rotate(centerToTopLeft, rotationEstimate);
    topRight = center + scaleEstimate*rotate(centerToTopRight, rotationEstimate);
    bottomLeft = center + scaleEstimate*rotate(centerToBottomLeft, rotationEstimate);
    bottomRight = center + scaleEstimate*rotate(centerToBottomRight, rotationEstimate);
}

//Функция addPoint_ добавляет новую точку в модель объекта под первым незанятым номером(если  таких нет то не добавляет)
//ВХОД:
//cv::KeyPoint &point - новая особая точка
//cv::Mat &descriptor -  дескриптор новой точки
//std::vector<bool> *isActive - массив в котором указано под какими номерами хранятся точки объекта, а под какими ничего не хранится и можно добавить новые точки
//int backgroundPointsNum - количество особых точек фона
//std::vector<cv::Point2f> *springs - координаты точек объекта относительно центра объекта
//cv::Point2f &center - центр объекта
//cv::Mat &featuresDatabase - база всех дескрипторов
//cv::Mat &selectedFeatures - база дескрипторов объекта
//std::vector<std::vector<float>* > squareForm - матрица попарных расстояний
//std::vector<std::vector<float>* > angles - матрица попарных углов
//std::vector<std::pair<cv::KeyPoint,int> > &activeKeypoints - массив пар вида (особая точка из модели объекта обнаруженная на текущем кадре; номер этой точки в модели)
//std::vector<int> *weigthsDatabase - база весов точек
//РЕЗУЛЬТАТ:
//std::vector<bool> *isActive - соответствуюшее значение меняется на true
//std::vector<cv::Point2f> *springs - заполняется соответствуюшее значение
//cv::Mat &featuresDatabase, cv::Mat &selectedFeatures - в соответствующую строку записывается дескриптор новой точки
//std::vector<std::vector<float>* > squareForm, std::vector<std::vector<float>* > angles - заменяются соответствующие строки и столбцы матриц
//std::vector<std::pair<cv::KeyPoint,int> > &activeKeypoints - добавляется новая точка с ее номером
//std::vector<int> *weigthsDatabase - под соответствующим номер ставится начальный вес точки

void addPoint_(cv::KeyPoint &point, cv::Mat &descriptor, std::vector<bool> *isActive, int backgroundPointsNum,
               std::vector<cv::Point2f> *springs, cv::Point2f &center, cv::Mat &featuresDatabase, cv::Mat &selectedFeatures,
               std::vector<std::vector<float>* > squareForm, std::vector<std::vector<float>* > angles,
               std::vector<std::pair<cv::KeyPoint,int> > &activeKeypoints, std::vector<int> *weigthsDatabase)
{
    //Get the first free number
    int idx = 0;
    while(idx < MAX_POINTS && isActive->at(idx)) {
        idx++;
    }
    if(idx == MAX_POINTS) {//no free points
        return;
    }
    isActive->at(idx) = true;
    descriptor.row(0).copyTo(featuresDatabase.row(idx + backgroundPointsNum));
    descriptor.row(0).copyTo(selectedFeatures.row(idx));

    for(unsigned int i = 0; i < MAX_POINTS; i++)
    {
        if(!isActive->at(i)) continue;

        float dx = springs->at(i).x + center.x - point.pt.x;
        float dy = springs->at(i).y + center.y - point.pt.y;
        float dist = sqrt(pow(dx,2) + pow(dy,2));
        if(idx!=i) {
            squareForm.at(idx)->at(i) = dist;
            squareForm.at(i)->at(idx) = dist;
            angles.at(idx)->at(i) = atan2(dy, dx);
            angles.at(i)->at(idx) = atan2(dy, dx);
        } else{
            squareForm.at(idx)->at(i) = 0;
            angles.at(idx)->at(i) = 0;
        }
    }
    springs->at(idx) = point.pt - center;
    activeKeypoints.push_back(std::make_pair(point, idx+1));
    weigthsDatabase->at(idx) = ADD_WEIGTH;
}

//Функция dropPoint_ удаляет точку под соответствующим номером
//ВХОД:
//int idx - номер удаляемой точки
//std::vector<bool> *isActive - массив в котором указано под какими номерами хранятся точки объекта, а под какими ничего не хранится и можно добавить новые точки
//int backgroundPointsNum - количество особых точек фона
//cv::Mat &featuresDatabase - база всех дескрипторов
//cv::Mat &selectedFeatures - база дескрипторов объекта
//РЕЗУЛЬТАТ:
//std::vector<bool> *isActive - в соответствуюшее значение ставится false
//cv::Mat &featuresDatabase, cv::Mat &selectedFeatures - соответствующие строки зануляются
void dropPoint_(int idx, std::vector<bool> *isActive, int backgroundPointsNum, cv::Mat &featuresDatabase, cv::Mat &selectedFeatures)
{
    isActive->at(idx)= false;

    featuresDatabase.row(idx + backgroundPointsNum).setTo(0);
    selectedFeatures.row(idx).setTo(0);
}

//Функция estimate_ пересчитывает параметры модели в соответствии с изменением объекта
//ВХОД:
//const std::vector<std::pair<cv::KeyPoint, int> >& keypointsIN - особые  точки модели объекта, найденные на текущем кадре
//float& scaleEstimate, float& medRot - изменение масштаба и угла поворота объекта
//std::vector<std::pair<cv::KeyPoint, int> >& keypoints
//cv::Point2f &center - координаты центра объекта
//int thrOutlier -  порог отсева точек в процессе кластеризации
//std::vector<std::vector<float>* > squareForm - матрица попарных расстояний
//std::vector<std::vector<float>* > angles - матрица попарных углов
//bool estimateScale, bool estimateRotation - параметры, указывающие следует ли учитывать масштаб и угол поворота
//std::vector<cv::Point2f> *springs - координаты точек объекта относительно центра объекта
//РЕЗУЛЬТАТ:
//float& scaleEstimate, float& medRot - вычисляются для текущего кадра
//std::vector<std::pair<cv::KeyPoint, int> >& keypoints - заполняется сортированными по номерам точками из keypointsIN
//cv::Point2f &center< std::vector<cv::Point2f> *springs - пересчитываются

void estimate_(const std::vector<std::pair<cv::KeyPoint, int> >& keypointsIN, float& scaleEstimate, float& medRot,
               std::vector<std::pair<cv::KeyPoint, int> >& keypoints, cv::Point2f &center, int thrOutlier,
               std::vector<std::vector<float>* > squareForm, std::vector<std::vector<float>* > angles,
               bool estimateScale, bool estimateRotation, std::vector<cv::Point2f> *springs)
{
    center = cv::Point2f(NAN,NAN);
    scaleEstimate = NAN;
    medRot = NAN;

    //At least 2 keypoints are needed for scale
    if(keypointsIN.size() > 1)
    {
        //sorting points in according to the the original indexes. result in keypoints
        std::vector<PairInt> list;

        for(unsigned int i = 0; i < keypointsIN.size(); i++)
            list.push_back(std::make_pair(keypointsIN[i].second, i));

        std::sort(&list[0], &list[0]+list.size(), comparatorPair<int>);

        for(unsigned int i = 0; i < list.size(); i++)
            keypoints.push_back(keypointsIN[list[i].second]);

        bool keypointsFlag = false;
        std::vector<float> scaleChange;
        std::vector<float> angleDiffs;
        for(unsigned int i = 0; i < list.size(); i++){
            for(unsigned int j = 0; j < list.size(); j++) {
                if(i != j && keypoints[i].second != keypoints[j].second) {
                    keypointsFlag = true;
                    cv::Point2f p = keypoints[j].first.pt - keypoints[i].first.pt;
                    //This distance might be 0 for some combinations,
                    //as it can happen that there is more than one keypoint at a single location
                    float dist = sqrt(p.dot(p));//L2 норма
                    float origDist = squareForm.at(keypoints[i].second-1)->at(keypoints[j].second-1);
                    scaleChange.push_back(dist/origDist);
                    //Compute angle
                    float angle = atan2(p.y, p.x);
                    float origAngle = angles.at(keypoints[i].second-1)->at(keypoints[j].second-1);
                    float angleDiff = angle - origAngle;
                    //Fix long way angles
                    if(fabs(angleDiff) > CV_PI)
                        angleDiff -= sign(angleDiff) * 2 * CV_PI;
                    angleDiffs.push_back(angleDiff);
                }
            }
        }
        if(keypointsFlag) {
            scaleEstimate = median(scaleChange);
            if(!estimateScale)
                scaleEstimate = 1;
            medRot = median(angleDiffs);
            if(!estimateRotation)
                medRot = 0;

            //estimate the the center coordinates
            std::vector<cv::Point2f> votes = std::vector<cv::Point2f> ();

            for(unsigned int i = 0; i < keypoints.size(); i++)
            {
                votes.push_back(keypoints[i].first.pt - scaleEstimate * rotate(springs->at(keypoints[i].second-1), medRot));
            }

            std::vector<std::pair<cv::KeyPoint, int> > newKeypoints;
            std::vector<cv::Point2f> newVotes;

            std::vector<int> newKeypointsNumbers = calcHist(votes, thrOutlier);
            if(newKeypointsNumbers.size()!= 0) {
                for(unsigned int i = 0; i < newKeypointsNumbers.size(); i++)
                {
                    int p = newKeypointsNumbers[i];
                    newKeypoints.push_back(keypoints[p]);
                    newVotes.push_back(votes[p]);
                }
                keypoints = newKeypoints;
                center = cv::Point2f(0,0);
                for(unsigned int i = 0; i < newVotes.size(); i++)
                    center += newVotes[i];
                    center *= (1.0/newVotes.size());
            }
        }
    }
}

//Функция getBestPoints_ сортирует  особые точки по убыванию response и
//оставляет в массиве только первые MAX_POINTS штук
void getBestPoints_(std::vector<cv::KeyPoint> &selected_keypoints)
{
    //choose the best MAX_POINTS keypoints
    std::sort(selected_keypoints.begin(), selected_keypoints.end(), compare_response);

    if(selected_keypoints.size() > MAX_POINTS) {
        while(selected_keypoints.size() > MAX_POINTS)
            selected_keypoints.pop_back();
    }
}

//Функция fillDB заполняет базы дескрипторов
//ВХОД:
//cv::Mat &selectedFeatures - дескрипторы особых точек объекта
//std::vector<int> *selectedClasses - номера особых точек объекта
//cv::Mat &featuresDatabase - дескрипторы всех особых точек (фона и объекта)
//cv::Mat &backgroundDatabase - дескрипторы особых точек фона
//std::vector<int> &classesDatabase - номера всех точек(классы)
//int backgroundPointsNum -  кол-во особых точек фона
//РЕЗУЛЬТАТ:
//cv::Mat &featuresDatabase - дескрипторы всех особых точек фона, затем объекта, затем нули(так чтобы кол-во точек для объекта равно MAX_POINTS)
//std::vector<int> &classesDatabase - номера всех точек фона, затем объекта

void fillDB (cv::Mat &selectedFeatures, std::vector<int> *selectedClasses, cv::Mat &featuresDatabase,
             cv::Mat &backgroundDatabase, std::vector<int> &classesDatabase,
             int backgroundPointsNum)
{
    //Assign each keypoint a class starting from 1, background is 0
    std::vector<int> backgroundClasses;
    for(unsigned int i = 0; i < backgroundPointsNum; i++)
        backgroundClasses.push_back(0);

    //Stack background features and selected features into database
    featuresDatabase = cv::Mat(backgroundDatabase.rows+MAX_POINTS, std::max(backgroundDatabase.cols,selectedFeatures.cols), backgroundDatabase.type());
    if(backgroundDatabase.cols > 0)
        backgroundDatabase.copyTo(featuresDatabase(cv::Rect(0,0,backgroundDatabase.cols, backgroundDatabase.rows)));
    unsigned char a[3];

    cv::Mat zeroDesc(cv::Mat::zeros(MAX_POINTS - selectedFeatures.rows, selectedFeatures.cols, featuresDatabase.type()));
    selectedFeatures.push_back(zeroDesc);

    if(selectedFeatures.cols > 0)
        selectedFeatures.copyTo(featuresDatabase(cv::Rect(0,backgroundDatabase.rows,selectedFeatures.cols, MAX_POINTS)));
    for(int i = 0; i < 3; i++) {
        a[i] = (unsigned char)featuresDatabase.at<unsigned char> (i, 0);
    }
    //Same for classes
    classesDatabase = std::vector<int>();
    for(unsigned int i = 0; i < backgroundClasses.size(); i++)
        classesDatabase.push_back(backgroundClasses.at(i));

    for(unsigned int i = 0; i < selectedClasses->size(); i++){
        classesDatabase.push_back(selectedClasses->at(i));
     }
}

void fillModelParams (cv::Point2f &topleft, cv::Point2f &bottomright, float &point_avg_dist,
                    std::vector<std::vector<float>* > squareForm, std::vector<std::vector<float>* > angles,
                    std::vector<cv::Point2f> *springs, cv::Point2f &center, std::vector<cv::KeyPoint> &selected_keypoints,
                    cv::Point2f &centerToTopLeft, cv::Point2f &centerToTopRight, cv::Point2f &centerToBottomRight, cv::Point2f &centerToBottomLeft)
{
    double avg = 0;
    for(unsigned int i = 0; i < selected_keypoints.size(); i++)
    {
        for(unsigned int j = 0; j < selected_keypoints.size(); j++)
        {
            float dx = selected_keypoints[j].pt.x-selected_keypoints[i].pt.x;
            float dy = selected_keypoints[j].pt.y-selected_keypoints[i].pt.y;
            double dist = sqrt(pow(dx,2)+pow(dy,2));
            avg+=dist;
            squareForm.at(i)->at(j) = dist;
            angles.at(i)->at(j) = atan2(dy, dx);
            squareForm.at(j)->at(i) = dist;
            angles.at(j)->at(i) = atan2(dy, dx); //wtf????
        }
    }

    point_avg_dist = avg/(selected_keypoints.size()*(selected_keypoints.size()-1));

//Find the center of selected keypoints

    for(unsigned int i = 0; i < selected_keypoints.size(); i++) {
        center += selected_keypoints[i].pt;
    }

    center *= (1.0/selected_keypoints.size());

    //Remember the rectangle coordinates relative to the center
    centerToTopLeft = topleft - center;
    centerToTopRight = cv::Point2f(bottomright.x, topleft.y) - center;
    centerToBottomRight = bottomright - center;
    centerToBottomLeft = cv::Point2f(topleft.x, bottomright.y) - center;

    //Calculate springs of each keypoint
    for(unsigned int i = 0; i < selected_keypoints.size(); i++)
        springs->at(i) = selected_keypoints[i].pt - center;
}

void initialise_(cv::Ptr<cv::FeatureDetector> &detector, cv::Ptr<cv::DescriptorExtractor> &descriptorExtractor ,cv::Ptr<cv::DescriptorMatcher> &descriptorMatcher,
                std::string &detectorType, std::string &matcherType, int &backgroundPointsNum, int descriptorLength,
                cv::Mat &selectedFeatures, std::vector<int> *selectedClasses, cv::Mat &featuresDatabase,
                cv::Mat &backgroundDatabase, std::vector<int> &classesDatabase, std::vector<int> *weigthsDatabase,
                cv::Mat &im_gray0, cv::Point2f &topleft, cv::Point2f &bottomright, float &point_avg_dist, unsigned int &nbInitialKeypoints,
                std::vector<std::vector<float>* > squareForm, std::vector<std::vector<float>* > angles,
                std::vector<std::pair<cv::KeyPoint,int> > &activeKeypoints,
                std::vector<bool> *isActive, std::vector<cv::Point2f> *springs, cv::Point2f &center, cv::Mat &im_prev,
                cv::Point2f &centerToTopLeft, cv::Point2f &centerToTopRight, cv::Point2f &centerToBottomRight, cv::Point2f &centerToBottomLeft)
{
    cv::Mat avg_object,avg_background;

    //Initialise detector, descriptor, matcher
    detector = cv::FeatureDetector::create(detectorType.c_str()); //cv::Algorithm::create<cv::FeatureDetector>(detectorType.c_str());
    descriptorExtractor = cv::DescriptorExtractor::create(detectorType.c_str()); //cv::Algorithm::create<cv::DescriptorExtractor>(descriptorType.c_str());
    descriptorMatcher = cv::DescriptorMatcher::create(matcherType.c_str());

    //Get initial keypoints in whole image
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(im_gray0, keypoints);

    //Remember keypoints that are in the rectangle as selected keypoints
    std::vector<cv::KeyPoint> selected_keypoints;
    std::vector<cv::KeyPoint> background_keypoints;

    inout_rect(keypoints, topleft, bottomright, selected_keypoints, background_keypoints);

    backgroundPointsNum = background_keypoints.size();

//____________________________________выбор наилучших точек
    getBestPoints_(selected_keypoints);
//____________________________________выбор наилучших точек
    if(selected_keypoints.size() == 0)
    {
        printf("No keypoints found in selection");
        return;
    }

    descriptorExtractor->compute(im_gray0, selected_keypoints, selectedFeatures);

    //Remember keypoints that are not in the rectangle as background keypoints
    cv::Mat background_features;

    descriptorExtractor->compute(im_gray0, background_keypoints, background_features);
    backgroundDatabase = background_features;
//____________________________________заполнение баз дескрипторов и классов
    fillDB (selectedFeatures, selectedClasses, featuresDatabase, backgroundDatabase, classesDatabase,
                 backgroundPointsNum);
//____________________________________заполнение баз дескрипторов и классов

//Get all distances between selected keypoints in squareform and get all angles between selected keypoints
    fillModelParams (topleft, bottomright, point_avg_dist, squareForm, angles, springs, center, selected_keypoints,
                        centerToTopLeft, centerToTopRight, centerToBottomRight, centerToBottomLeft);

    //Set start image for tracking
    im_prev = im_gray0.clone();

    //Make keypoints 'active' keypoints
    //____________________нелинейная  фрис
    //    std::vector<std::vector<cv::DMatch> > selectedMatchesAll, backgroundMatches;
    //    descriptorMatcher->knnMatch(selectedFeatures, selectedFeatures, selectedMatchesAll, 2);
    //    descriptorMatcher->knnMatch(selectedFeatures, background_features, backgroundMatches, 1);
    //    activeKeypoints = std::vector<std::pair<cv::KeyPoint,int> >();
    //    for(unsigned int i = 0; i < selected_keypoints.size(); i++) {
    //        float fris_val = fris(backgroundMatches[i][0].distance, selectedMatchesAll[i][1].distance);
    //        weigthsDatabase->at(i) = (MAX_WEIGTH*(fris_val + 3))/4;
    //        //weigthsDatabase->at(i) = (MAX_WEIGTH*(fris_val + 7))/8;
    //        isActive->at(i) = true;
    //        activeKeypoints.push_back(std::make_pair(selected_keypoints[i], selectedClasses->at(i)));
    //    }
    //____________________нелинейная фрис
    activeKeypoints = std::vector<std::pair<cv::KeyPoint,int> >();
    avg_background = getAverage(background_features, descriptorLength);
    avg_object = getAverage(selectedFeatures, descriptorLength);
    int dist = calcDist(avg_object, avg_background);
    for(unsigned int i = 0; i < selected_keypoints.size(); i++) {
    //        int d1 = calcDist(selectedFeatures.row(i),avg_background);
    //        int d2 = calcDist(selectedFeatures.row(i),avg_object);
    //        float f = fris(d1, d2);
    //        weigthsDatabase->at(i) = (MAX_WEIGTH*(3*f + 5))/8;//25%
    //        weigthsDatabase->at(i) = (MAX_WEIGTH*(f + 3))/4;//50%
    //        weigthsDatabase->at(i) = (MAX_WEIGTH*(2*f + 3))/5;//20%
    //        weigthsDatabase->at(i) = (MAX_WEIGTH*(f + 7))/8;//75%
    //        weigthsDatabase->at(i) = (MAX_WEIGTH*(5*f + 14))/90;//10%
        weigthsDatabase->at(i) = (-dist + 662)/3;
        isActive->at(i) = true;
        activeKeypoints.push_back(std::make_pair(selected_keypoints[i], selectedClasses->at(i)));
    }

    //Remember number of initial keypoints
    nbInitialKeypoints = selected_keypoints.size();
}

void getActivePoints (cv::Mat &activeFeaturesDB, cv::Mat &activeSelectedFeatures, cv::Mat &backgrdDb, cv::Mat &featuresDatabase,
                      std::vector<int> &featuresIdxs, std::vector<int> &selectedfeaturesIdxs, std::vector<bool> *isActive, int backgroundPointsNum)
{
    activeFeaturesDB.push_back(backgrdDb);

    for(int i = 0; i < MAX_POINTS; i++) {
        if(isActive->at(i)) {
            activeFeaturesDB.push_back(featuresDatabase.row(i+backgroundPointsNum));
            activeSelectedFeatures.push_back(featuresDatabase.row(i+backgroundPointsNum));
            selectedfeaturesIdxs.push_back(i);
            featuresIdxs.push_back(i+backgroundPointsNum);
        }
    }
}

void matchPoints (int descriptorLength, int thrOutlier, float thrConf, float thrRatio,
                  std::vector<int> *selectedClasses, std::vector<cv::KeyPoint> &keypoints,
                  std::vector<int> &featuresIdxs, std::vector<int> &selectedfeaturesIdxs,
                  std::vector<int> &classesDatabase, std::vector<cv::Point2f> &transformedSprings,
                  std::vector<std::vector<cv::DMatch> > &matchesAll, std::vector<std::vector<cv::DMatch> > &selectedMatchesAll,
                  std::vector<std::pair<cv::KeyPoint,int> > &activeKeypoints,std::vector<std::pair<cv::KeyPoint,int> > &trackedKeypoints,
                  std::vector<bool> *isActive, std::vector<cv::Point2f> *springs, cv::Point2f &center)

{
    for(unsigned int i = 0; i < keypoints.size(); i++)
    {
        cv::KeyPoint keypoint = keypoints[i];

        //First: Match over whole image
        //Compute distances to all descriptors
        std::vector<cv::DMatch> matches = matchesAll[i];

        //Convert distances to confidences, do not weight
        std::vector<float> combined;
        for(unsigned int j = 0; j < matches.size(); j++)
            combined.push_back(1 - matches[j].distance / descriptorLength);

        std::vector<int>& classes = classesDatabase;

        //Get best index
        int bestInd = featuresIdxs[matches[0].trainIdx];

        //Compute distance ratio according to Lowe
        float ratio = (1-combined[0]) / (1-combined[1]);

        //Extract class of best match
        int keypoint_class = classes[bestInd];

        //If distance ratio is ok and absolute distance is ok and keypoint class is not background
        if(ratio < thrRatio && combined[0] > thrConf && keypoint_class != 0) {
            activeKeypoints.push_back(std::make_pair(keypoint, keypoint_class));

        }
//In a second step, try to match difficult keypoints
            //If structural constraints are applicable
            if(!(isnan(center.x) | isnan(center.y)))
            {
                //Compute distances to initial descriptors
                std::vector<cv::DMatch> matches = selectedMatchesAll[i];
                std::vector<float> distances(matches.size()), distancesTmp(matches.size());
                std::vector<int> trainIndex(matches.size());
                for(unsigned int j = 0; j < matches.size(); j++)
                {
                    distancesTmp[j] = matches[j].distance;
                    trainIndex[j] = matches[j].trainIdx;
                }
                //Re-order the distances based on indexing
                std::vector<int> idxs = argSortInt(trainIndex);
                for(unsigned int j = 0; j < idxs.size(); j++)
                    distances[j] = distancesTmp[idxs[j]];

                //Convert distances to confidences
                std::vector<float> confidences(matches.size());
                for(unsigned int j = 0; j < matches.size(); j++)
                    confidences[j] = 1 - distances[j] / descriptorLength;

                //Compute the keypoint location relative to the object center
                cv::Point2f relative_location = keypoint.pt - center;

                //Compute the distances to all springs
                std::vector<float> displacements(springs->size());
                std::vector<float> oldIdxToNew(springs->size(), 0);//inverse index mapping
                for(unsigned int j = 0; j < selectedfeaturesIdxs.size(); j++) {
                    oldIdxToNew.at(selectedfeaturesIdxs[j]) = j;
                }

                for(unsigned int j = 0; j < springs->size(); j++) {
                    if(isActive->at(j)) {
                        cv::Point2f p = (transformedSprings[j] - relative_location);
                        displacements[oldIdxToNew[j]] = sqrt(p.dot(p));
                    }
                }

                //For each spring, calculate weight
                std::vector<float> combined(confidences.size());
                for(unsigned int j = 0; j < confidences.size(); j++)
                    combined[j] = (displacements[j] < thrOutlier)*confidences[j];

                std::vector <int>* classes = selectedClasses;

                //Sort in descending order по степеи похожести с сохранением индексов
                std::vector<PairFloat> sorted_conf(combined.size());
                for(unsigned int j = 0; j < combined.size(); j++)
                    sorted_conf[j] = std::make_pair(combined[j], j);
                std::sort(&sorted_conf[0], &sorted_conf[0]+sorted_conf.size(), comparatorPairDesc<float>);
                //Get best and second best index
                int bestInd = sorted_conf[0].second;
                int secondBestInd = sorted_conf[1].second;

                //Compute distance ratio according to Lowe
                float ratio = (1-combined[bestInd]) / (1-combined[secondBestInd]);

                //Extract class of best match
                int keypoint_class = classes->at(selectedfeaturesIdxs[bestInd]);

                //If distance ratio is ok and absolute distance is ok and keypoint class is not background
                if(ratio < thrRatio && combined[bestInd] > thrConf && keypoint_class != 0)
                {
                    for(int i = activeKeypoints.size()-1; i >= 0; i--)
                    {
                        if(activeKeypoints[i].second == keypoint_class)
                            activeKeypoints.erase(activeKeypoints.begin()+i);
                    }
                    activeKeypoints.push_back(std::make_pair(keypoint, keypoint_class));
                }
            }
    }

//If some keypoints have been tracked
    if(trackedKeypoints.size() > 0)
    {
        //Extract the keypoint classes
        std::vector<int> tracked_classes(trackedKeypoints.size());
        for(unsigned int i = 0; i < trackedKeypoints.size(); i++)
            tracked_classes[i] = trackedKeypoints[i].second;

        //If there already are some active keypoints
        if(activeKeypoints.size() > 0)
        {
            //Add all tracked keypoints that have not been matched
            std::vector<int> associated_classes(activeKeypoints.size());
            for(unsigned int i = 0; i < activeKeypoints.size(); i++)
                associated_classes[i] = activeKeypoints[i].second;

            std::vector<bool> notmissing = in1d(tracked_classes, associated_classes);
            for(unsigned int i = 0; i < trackedKeypoints.size(); i++)
                if(!notmissing[i])
                    activeKeypoints.push_back(trackedKeypoints[i]);
        }
        else activeKeypoints = trackedKeypoints;
    }
}

void deletePoints (int backgroundPointsNum, cv::Mat &selectedFeatures, cv::Mat &featuresDatabase,
                   std::vector<int> *weigthsDatabase, std::vector<std::pair<cv::KeyPoint,int> > &activeKeypoints,
                   std::vector<bool> *isActive, bool &hasResult)
{
    hasResult = true;
    for(unsigned int i = 0; i < activeKeypoints.size(); i++) {
        if(weigthsDatabase->at(activeKeypoints[i].second-1) != MAX_WEIGTH) {
            weigthsDatabase->at(activeKeypoints[i].second-1)+=2;
        }
    }

    for(unsigned int i = 0; i < weigthsDatabase->size(); i++) {
        if(isActive->at(i)) {
            if(weigthsDatabase->at(i) != MAX_WEIGTH) {
                weigthsDatabase->at(i)--;
            }
            if(weigthsDatabase->at(i) < 0) {
              dropPoint_(i,isActive,backgroundPointsNum, featuresDatabase, selectedFeatures);
              //d++;
            }
        }
    }
}

void addNewPoints(std::vector<cv::KeyPoint> &keypoints, int &backgroundPointsNum,
                  cv::Mat &selectedFeatures, cv::Mat &featuresDatabase, std::vector<int> &selectedfeaturesIdxs,
                  std::vector<int> *weigthsDatabase, std::vector<std::vector<cv::DMatch> > &selectedMatchesAll,
                  std::vector<std::vector<cv::DMatch> > &backgroundMatches, cv::Mat &features,
                  float &point_avg_dist, float dist_coef, float tresh1, float tresh2,
                  std::vector<std::vector<float>* > squareForm, std::vector<std::vector<float>* > angles,
                  std::vector<std::pair<cv::KeyPoint,int> > &activeKeypoints,
                  std::vector<bool> *isActive, std::vector<cv::Point2f> *springs, cv::Point2f &center,
                  cv::Point2f &topLeft, cv::Point2f &topRight, cv::Point2f &bottomRight, cv::Point2f &bottomLeft)
{
    std::vector<cv::Point2f> arr;
    arr.push_back(topLeft);
    arr.push_back(topRight);
    arr.push_back(bottomRight);
    arr.push_back(bottomLeft);
//добавление точек
    for(unsigned int j = 0; j < keypoints.size(); j++) {
        double in = cv::pointPolygonTest(arr, keypoints[j].pt, false);
        if(in >= 0) {
            float fris_val = fris(backgroundMatches[j][0].distance, selectedMatchesAll[j][0].distance);
            if(fris_val >= tresh1 && fris_val <= tresh2) {
                int idx = selectedfeaturesIdxs[selectedMatchesAll[j][0].trainIdx];
                cv::Point2f keypt = keypoints[j].pt;
                cv::Point2f keypt2 = (springs->at(idx));
                float dx = keypt.x-center.x-keypt2.x;
                float dy = keypt.y-center.y-keypt2.y;
                double dist = sqrt(pow(dx,2)+pow(dy,2));
                if (dist >= (dist_coef*point_avg_dist)) {
                    //a++;
                    addPoint_(keypoints[j], features.row(j), isActive, backgroundPointsNum, springs, center, featuresDatabase,
                              selectedFeatures, squareForm, angles, activeKeypoints, weigthsDatabase);
                }
           }
        }
    }
}

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
                   cv::Point2f &topLeft, cv::Point2f &topRight, cv::Point2f &bottomRight, cv::Point2f &bottomLeft)
{
    trackedKeypoints = std::vector<std::pair<cv::KeyPoint, int> >();
    std::vector<unsigned char> status;

    track(im_prev, im_gray, activeKeypoints, trackedKeypoints, status);

    std::vector<std::pair<cv::KeyPoint, int> > trackedKeypoints2;

    float scaleEstimate;
    float rotationEstimate;

    //estimate(trackedKeypoints, scaleEstimate, rotationEstimate, trackedKeypoints2);
    estimate_(trackedKeypoints, scaleEstimate, rotationEstimate, trackedKeypoints2, center, thrOutlier,
              squareForm, angles, estimateScale, estimateRotation, springs);

    trackedKeypoints = trackedKeypoints2;

    //Detection
    //Detect keypoints, compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat features;
    detector->detect(im_gray, keypoints);
    descriptorExtractor->compute(im_gray, keypoints, features);

    //Create list of active keypoints
    activeKeypoints = std::vector<std::pair<cv::KeyPoint, int> >();

    //Get the best two matches for each feature
    std::vector<std::vector<cv::DMatch> > matchesAll/*compare all features*/ ,
            selectedMatchesAll/*compare object features*/, backgroundMatches;

    cv::Mat activeFeaturesDB, activeSelectedFeatures;
    std::vector<int> featuresIdxs(backgroundPointsNum, 0), selectedfeaturesIdxs;
    cv::Mat backgrdDb = featuresDatabase.rowRange(0,backgroundPointsNum);

//____________________________________выбор активных точек
    getActivePoints(activeFeaturesDB, activeSelectedFeatures, backgrdDb, featuresDatabase,
                          featuresIdxs, selectedfeaturesIdxs, isActive, backgroundPointsNum);
//____________________________________выбор активных точек

    descriptorMatcher->knnMatch(features, activeFeaturesDB, matchesAll, 2);
    descriptorMatcher->knnMatch(features, backgrdDb, backgroundMatches, 2);


    //Get all matches for selected features
    if(!isnan(center.x) && !isnan(center.y))
        descriptorMatcher->knnMatch(features, activeSelectedFeatures, selectedMatchesAll, selectedFeatures.rows);

    std::vector<cv::Point2f> transformedSprings(springs->size());
    int idx = 0;
    while(idx < MAX_POINTS) {
        if(isActive->at(idx)) {
            transformedSprings[idx] = scaleEstimate * rotate(springs->at(idx), -rotationEstimate);
        }
        idx++;
    }

    //For each keypoint and its descriptor
    matchPoints (descriptorLength, thrOutlier, thrConf, thrRatio,
                      selectedClasses, keypoints,
                      featuresIdxs, selectedfeaturesIdxs,
                      classesDatabase, transformedSprings,
                      matchesAll, selectedMatchesAll,
                      activeKeypoints, trackedKeypoints,
                      isActive, springs, center);

    //Update object state estimate
    im_prev = im_gray;
    topLeft = cv::Point2f(NAN,NAN);
    topRight = cv::Point2f(NAN,NAN);
    bottomLeft = cv::Point2f(NAN,NAN);
    bottomRight = cv::Point2f(NAN,NAN);

    boundingbox = cv::Rect_<float>(NAN,NAN,NAN,NAN);
    hasResult = false;
//изменение веса и удаление
    if(!(isnan(center.x) | isnan(center.y)) && activeKeypoints.size() > nbInitialKeypoints / 10)
    {
        deletePoints (backgroundPointsNum, selectedFeatures, featuresDatabase,
                           weigthsDatabase, activeKeypoints,
                           isActive, hasResult);

        recalcCenter_(scaleEstimate, rotationEstimate, isActive, topLeft, topRight, bottomRight, bottomLeft, springs, center, centerToTopLeft,
                      centerToTopRight, centerToBottomRight, centerToBottomLeft);

        addNewPoints(keypoints, backgroundPointsNum,
                          selectedFeatures, featuresDatabase, selectedfeaturesIdxs,
                          weigthsDatabase, selectedMatchesAll,
                          backgroundMatches, features,
                          point_avg_dist, dist_coef, tresh1, tresh2,
                          squareForm, angles,
                          activeKeypoints,
                          isActive, springs, center,
                          topLeft, topRight, bottomRight, bottomLeft);

        recalcCenter_(scaleEstimate, rotationEstimate, isActive, topLeft, topRight, bottomRight, bottomLeft, springs, center, centerToTopLeft,
                      centerToTopRight, centerToBottomRight, centerToBottomLeft);

        float minx = std::min(std::min(topLeft.x,topRight.x),std::min(bottomRight.x, bottomLeft.x));
        float miny = std::min(std::min(topLeft.y,topRight.y),std::min(bottomRight.y, bottomLeft.y));
        float maxx = std::max(std::max(topLeft.x,topRight.x),std::max(bottomRight.x, bottomLeft.x));
        float maxy = std::max(std::max(topLeft.y,topRight.y),std::max(bottomRight.y, bottomLeft.y));

        boundingbox = cv::Rect_<float>(minx, miny, maxx-minx, maxy-miny);
    }
}
