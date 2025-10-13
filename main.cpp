#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;
void makeSkinMask(Mat& Image, Scalar lower, Scalar upper,
    int ksize, int openIter, int closeIter);
void detect3FrameMotion(Mat& Image, Mat& out, int minNeighbors);
void thresHand(Mat& Image, Mat& handMask, double margin);

int main(int argc, char** argv)
{

    VideoCapture cap(0);
    /*cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);*/

    if (!cap.isOpened()) {
        cout << "Can't open the camera" << endl;
        return -1;
    }

    Scalar lower(0, 133, 77);
    Scalar upper(255, 173, 127);
    /*Scalar lower(95, 130, 40);
    Scalar upper(255, 155, 130);*/ //둘 중에 더 잘 검출되는 범위 사용해주세요!
    Mat img;
    Mat EdgeMask;
    Mat out;
    vector<Point> EdgePoints;
    vector<vector<Point>> EdgeGroups;

    while (1) {
        cap >> img;
        if (img.empty()) break;

        namedWindow("camera", WINDOW_AUTOSIZE);

        Mat mask = img.clone();
        Mat handMask = Mat::zeros(img.size(), CV_8UC1);

        makeSkinMask(mask, lower, upper, 3, 1, 1);
        detect3FrameMotion(mask, out, 4); // 주변 3*3 픽셀에서 움직인 픽셀 개수가 4개 이상이면 탐지
        thresHand(mask, handMask, 0.003);

        Mat finalMask = Mat::zeros(img.size(), CV_8UC1);
        bitwise_or(out, handMask, finalMask);

        int minArea = (finalMask.cols * finalMask.rows) / 50; // 필요 시 조정 (화면 비율 1/1000보다 작으면 제거) //50으로 수정

        Mat labels, stats, centroids;
        int n = connectedComponentsWithStats(finalMask, labels, stats, centroids, 8, CV_32S);

        for (int i = 1; i < n; ++i) { // 0은 배경
            int area = stats.at<int>(i, CC_STAT_AREA);
            if (area < minArea) {
                finalMask.setTo(0, labels == i);  // 작은 성분 삭제
            }
        }


        imshow("finalMask", finalMask);
        // 2) (작은 성분 제거 후) 남은 성분들에 사각형 그리기
        Mat Result = img.clone();

        //-----사각형 그리기-----//
        Mat labels2, stats2, centroids2;
        //n2=라벨 총 개수
        int n2 = connectedComponentsWithStats(finalMask, labels2, stats2, centroids2, 8, CV_32S);

        for (int i = 1; i < n2; i++) {
            int left = stats2.at<int>(i, CC_STAT_LEFT);
            int top = stats2.at<int>(i, CC_STAT_TOP);
            int width = stats2.at<int>(i, CC_STAT_WIDTH);
            int height = stats2.at<int>(i, CC_STAT_HEIGHT);
            int area = stats2.at<int>(i, CC_STAT_AREA);

            if (area < minArea) continue;

            //손 영역 사각형 그리기
            Rect box(left, top, width, height);
            rectangle(Result, box, Scalar(0, 255, 0), 2);

        }


        imshow("Hand detection", Result);

        if (waitKey(1) == 27)
            break;
    }
    return 0;
}

void makeSkinMask(Mat& Image,
    Scalar lower, Scalar upper,
    int ksize, int openIter, int closeIter)
{
    GaussianBlur(Image, Image, Size(5, 5), 0);
    cvtColor(Image, Image, COLOR_BGR2YCrCb);

    Mat mask(Image.size(), CV_8UC1, Scalar(0));
    /*inRange(Image, lower, upper, Image); //for문으로 수정*/
    for (int j = 0; j < Image.rows; j++) {
        for (int i = 0; i < Image.cols; i++) {
            uchar Y = Image.at<Vec3b>(j, i)[0];
            uchar Cr = Image.at<Vec3b>(j, i)[1];
            uchar Cb = Image.at<Vec3b>(j, i)[2];

            if (Y >= lower[0] && Y <= upper[0]
                && Cr >= lower[1] && Cr <= upper[1]
                && Cb >= lower[2] && Cb <= upper[2]) {
                mask.at<uchar>(j, i) = 255;
            }
            else {
                mask.at<uchar>(j, i) = 0;
            }
        }
    }
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(ksize, ksize)); //원형 커널 생성
    morphologyEx(mask, mask, MORPH_OPEN, kernel, Point(-1, -1), openIter);  // 열기
    morphologyEx(mask, mask, MORPH_CLOSE, kernel, Point(-1, -1), closeIter); // 닫기
    Image = mask;
}

void detect3FrameMotion(Mat& Image, Mat& out, int minNeighbors)
{
    static Mat prev1, prev2;

    // 초기화: 히스토리가 없으면 그냥 통과
    if (prev1.empty()) {
        out = Image.clone();
        prev1 = Image.clone();
        prev2 = Image.clone();
        return;
    }

    // 1) 프레임 차이 기반 motion 맵 (변화한 곳 = 255)
    Mat d1, d2, motion;
    bitwise_xor(Image, prev1, d1);
    bitwise_xor(Image, prev2, d2);
    bitwise_or(d1, d2, motion); // 최근 2프레임 중 하나라도 달라졌으면 motion

    //0/1로 정규화
    Mat m01, sum;
    threshold(motion, m01, 0, 1, THRESH_BINARY);
    boxFilter(m01, sum, CV_8U, Size(3, 3), Point(-1, -1), false, BORDER_REPLICATE);
    //가장자리 픽셀을 복제해서 채움

    Mat keepMask;
    threshold(sum, keepMask, minNeighbors - 1, 255, THRESH_BINARY);

    bitwise_and(Image, keepMask, out);
    //keepMask가 참(255)+이미지도 하얀부분 위치만 최종 out에 남김.

    prev2 = prev1.clone();
    prev1 = Image.clone();
}

void thresHand(Mat& Image, Mat& handMask, double margin = 0.05)
{
    Mat bin = Image.clone();
    const int w = bin.cols;
    const int h = bin.rows;
    const double r = 1.0 / 27.0;
    const double A_min = w * h * r; // 너무 작은 성분 제거 임계값

    Mat labels, stats, centroids;
    int num_labels = connectedComponentsWithStats(Image, labels, stats, centroids, 8, CV_32S);
    if (num_labels <= 1) {
        Mat::zeros(Image.size(), CV_8UC1);
    }

    //얼굴로 추정되는 가장 큰 영역 찾아냄 
    int faceIdx = -1;
    int maxWidth = -1;
    for (int i = 1; i < num_labels; ++i) {
        int w_i = stats.at<int>(i, CC_STAT_WIDTH);
        if (w_i > maxWidth) { maxWidth = w_i; faceIdx = i; }
    }
    if (faceIdx < 0) {
        Mat::zeros(Image.size(), CV_8UC1);
    }

    double faceW = static_cast<double>(maxWidth);
    double thrW = faceW * (1.0 - margin); //얼굴보다 margin 만큼 작은 객체들만 손 후보로 인정

    for (int i = 1; i < num_labels; ++i) {
        if (i == faceIdx) continue; //얼굴로 인식된 라벨은 건너뜀
        int w_i = stats.at<int>(i, CC_STAT_WIDTH);
        int area = stats.at<int>(i, CC_STAT_AREA);

        if (w_i < thrW && area >= A_min) {
            handMask.setTo(255, labels == i);
        }
    }
}

