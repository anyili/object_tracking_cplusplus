#include <iostream>
#include <thread>
#include "libs/libowi.h"

#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;
using namespace chrono;


const String CAPTURE_WINDOW_NAME = "Face Detection";
const double IMAGE_SCALE = 2.0;
const int SHIFT_TOLERANCE_X = 50;
const int SHIFT_TOLERANCE_Y = 20;
const unsigned int RELAX_TIME = 2000;


void DrawROI(Mat &image, const Rect &area, const Scalar &color, int thickNess, CV_OUT Point *points) {
    Point pt1(int(area.x * IMAGE_SCALE), int(area.y * IMAGE_SCALE));
    Point pt2(int((area.x + area.width) * IMAGE_SCALE), int((area.y + area.height) * IMAGE_SCALE));
    rectangle(image, pt1, pt2, color, thickNess);
    points[0] = pt1;
    points[1] = pt2;
}


void detectAndDraw(Mat &originalImage, CascadeClassifier &faceCascade, CascadeClassifier &fistCascade, CV_OUT Point &center) {

    vector<Rect> faces, fists;
    Mat grayImage, reducedImage;

    cvtColor(originalImage, grayImage, COLOR_BGR2GRAY);
    resize(grayImage, reducedImage, Size(), 1.0 / IMAGE_SCALE, 1.0 / IMAGE_SCALE);
    equalizeHist(reducedImage, reducedImage);
    faceCascade.detectMultiScale(reducedImage, faces, 1.2, 3,
                                  0|CASCADE_SCALE_IMAGE, Size(40, 40));
    for (auto face: faces) {
        Point points[2];
        DrawROI(originalImage, face, Scalar(255, 0, 0), 3, points);
    }

    fistCascade.detectMultiScale(reducedImage, fists, 1.2, 3,
                                  0|CASCADE_SCALE_IMAGE, Size(40, 40));


    for (auto fist: fists) {
        Point points[2];
        DrawROI(originalImage, fist, Scalar(0, 255, 0), 3, points);
        center.x = points[0].x + (points[1].x - points[0].x) / 2;
        center.y = points[0].y + (points[1].y - points[0].y) / 2;
    }
    imshow(CAPTURE_WINDOW_NAME, originalImage);
}

int getShift (int original, int current, int tolerance,
              CV_OUT int &newOriginal) {
    int delta = 0;
    if (abs(current - original) < tolerance) {
        newOriginal = original;
    } else {
        newOriginal = current;
        delta = current - original;
    }
    return delta;
}

void moveArmX(int delta) {
    if (delta > 0) {
        owi_base_clockwise();
        this_thread::sleep_for(1s);
        owi_base_off();
    }else if (delta < 0){
        owi_base_counterclockwise();
        this_thread::sleep_for(1s);
        owi_base_off();
    }
    if (delta != 0) {
        cout << "moving in x " << delta << endl;
    }
}

void moveArmY(int delta) {
    if (delta > 0) {
        owi_m4_forward();
        this_thread::sleep_for(1s);
        owi_m4_off();
    }else if (delta < 0){
        owi_m4_reverse();
        this_thread::sleep_for(1s);
        owi_m4_off();
    }
    if (delta != 0) {
        cout << "moving in y " << delta << endl;
    }
}

void moveArmTestX(int delta) {
    if (delta > 0) {
        this_thread::sleep_for(1s);
    }else if (delta < 0){
        this_thread::sleep_for(1s);
    }
    if (delta != 0) {
        cout << "moving in x " << delta << endl;
    }
}

void moveArmTestY(int delta) {
    if (delta > 0) {
        this_thread::sleep_for(1s);
    }else if (delta < 0){
        this_thread::sleep_for(1s);
    }
    if (delta != 0) {
        cout << "moving in y " << delta << endl;
    }
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        cout << "need face and fist haar cascade files" << endl;
        return 1;
    }
    owi_init();

    string faceCascadePath = argv[1];
    string fistCascadePath = argv[2];

    VideoCapture cap(0);
    Mat frame;
    CascadeClassifier faceCascade(faceCascadePath);
    CascadeClassifier fistCascade(fistCascadePath);

    namedWindow(CAPTURE_WINDOW_NAME, WINDOW_AUTOSIZE);

    int originalX = -1, originalY = -1;

    long endTime;
    long startTime = system_clock::now().time_since_epoch() / milliseconds(1);

    while(true) {
        cap >> frame;
        flip(frame, frame, 1);
        Point center(-1, -1);
        detectAndDraw(frame, faceCascade, fistCascade, center);
        if (center.x != -1 && center.y != -1) {
            if (originalX == -1) {
                originalX = center.x;
            } else {
                int delta = getShift(originalX, center.x, SHIFT_TOLERANCE_X, originalX);
                thread(moveArmX, delta).detach();
            }
            if (originalY == -1) {
                originalY = center.y;
            } else {
                int delta = getShift(originalY, center.y, SHIFT_TOLERANCE_Y, originalY);
                thread(moveArmY, delta).detach();
            }

            startTime = system_clock::now().time_since_epoch() / milliseconds(1);
        } else {
            endTime = system_clock::now().time_since_epoch() / milliseconds(1);
            if ((endTime - startTime) > RELAX_TIME) {
                    originalX = -1;
                    originalY = -1;
            }
        }
        if( (char) waitKey(1) == 'q')
            break;
    }
    cap.release();
    destroyAllWindows();

    return 0;
}