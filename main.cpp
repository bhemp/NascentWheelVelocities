/*----------------------------------------------
* Nascent Coding Challenge 1
* 
* OpenCV Version: 3.4.5
* April 24, 2019
* 
*--------------------------------------------------*/

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/tracking/tracker.hpp"
#include "opencv2/core/types.hpp"

using namespace cv;
using namespace std;

int main()
{
  // input video
  string inputFile = "cars_passing_input.mp4";

  // open video file
  VideoCapture cap(inputFile);
  if(!cap.isOpened())  // check for success
      return -1;

  // get frame width, height, and FPS from input video
  int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
  int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
  int fps = static_cast<int>(cap.get(CAP_PROP_FPS));
  fps = fps/3; // slow output video framerate
  int lastFrame = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

  Size frame_size(frame_width, frame_height);
  
  // ROI parameters
  double topOffset = frame_height*0.39;
  double ROIheight = frame_height*0.2;
  double bottomOffset = frame_height - ROIheight - topOffset;

  // create Mats for normal and grayscale image frames
  Mat frame;      // direct frame capture
  Mat frameGray;   // grayscale crop from frame
  Mat frameBG;    // background frame to subtract

  // create variables for counts
  int frameCount = 0;
  int wheelCountTotal = 0;     // total number of wheels detected
  int wheelCountCurrFrame = 0; // number of wheels in current frame
  int wheelCountPrevFrame = 0; // number of wheels in previous frame

  // detection parameters
  vector<vector<Point>> contours; // Canny contour return vector
  vector<Vec4i> hierarchy;        // Canny hierarchy return vector
  double minWheelSize = 110;         // minimum wheel size (px)
  double maxWheelSize = 150;         // maximum wheel size (px)
  float eccThresh = 0.5;         // ellipse eccentricity threshold (circle = 0.0)
  size_t minContourPoints = 80;   // minimum number of points to check contour for ellipse
  int medBlur = 15;

  // tracking parameters
  Ptr<MultiTracker> trackers = cv::MultiTracker::create();
  int numTrackers = 0;  // number of trackers in multitracker
  vector<double> vXavg; // average velocity of tracked object
  int lastWheelNum = 0; // right most wheel number on screen
  double ROIpad = 10; // detection rectangle expansion to improve tracking

  int waitTime = 20;   // pause time for frame display
 
  // open output file for writing
  VideoWriter oVideoWriter("output.mp4", 0x00000021, 
                                                              fps, frame_size, true);

  // main program loop
  for(;;)
  {
    // get a new frame from camera
    cap >> frame; 

    // increment frame counter
    frameCount++;
    // break on last frame
    if (frameCount > lastFrame)
      break;

    system("clear");
    cout << "Processing frame: " << frameCount << endl;

    // reinitalze variables
    wheelCountPrevFrame = wheelCountCurrFrame;
    wheelCountCurrFrame = 0;

    // crop the full image to eliminate erroneous detections outside the road and improve speed
    frameGray = frame;

    // convert cropped frame to grayscale
    cvtColor(frame, frameGray, COLOR_BGR2GRAY);

    // remove information outside of ROI
    frameGray(Rect(0, 0, frame_width, topOffset)) = 0;
    frameGray(Rect(0, (topOffset + ROIheight + 1), frame_width, bottomOffset)) = 0;

    // blur to reduce noise
    medianBlur(frameGray, frameGray, medBlur);

    // binarize image
    threshold(frameGray, frameGray, 30, 255, 0); 

    // update trackers if items are present
    if (numTrackers > 0)
      trackers -> update(frameGray);

    // find contours and detect wheels
    findContours( frameGray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE );

    if ( contours.size() > 0 )
    {

      Mat drawing = Mat::zeros( frameGray.size(), CV_8UC3 );

      // wheel detection via contours and ellipse fitting
      for( size_t i = 0; i < contours.size(); i++ )
      {
        if( contours[i].size() > minContourPoints )
        {
          RotatedRect detEllipse = fitEllipse( Mat(contours[i]) );

          // get axes
          double a = detEllipse.size.width  / 2;   // major axis
          double b = detEllipse.size.height / 2;   // minor axis
          double x = detEllipse.center.x;
          double y = detEllipse.center.y;
          if (b > a)   // check that axes are correct
            swap(a,b); // swap if reversed

          // check circularity with eccentricity value
          float ecc = sqrt( 1 - pow(b,2)/pow(a,2) ); // eccentricity (circle = 0)

          // store ellipse if nearly circular and within size range
          if ( ecc<eccThresh && (a*2)>minWheelSize && (a*2)<maxWheelSize )
          {  
            // create rect for each detected ellipse
            // note it does not seem that the OpenCV library currently allows deletion of a tracker instance
            // might lead to memory issues
            wheelCountCurrFrame++;
            if (wheelCountCurrFrame > wheelCountPrevFrame)
            {
              wheelCountTotal++;
              double w = 2*a + 2*ROIpad;
              double h = 2*a + 2*ROIpad;
              // move center point to top left for rect
              x = x - w/2;
              y = y - h/2;
              vXavg.push_back(x); // left rect ref

              trackers -> add(TrackerCSRT::create(), frameGray, Rect(x,y,w,h));
            }
          }
        }
      }
    }

    numTrackers = trackers -> getObjects().size();

    // check if wheel leaves frame
    if (wheelCountCurrFrame < wheelCountPrevFrame)
    {
      lastWheelNum++; // rightmost wheel number left on the screen
    }

    // draw ellipses and determine velocity if objects are on screen and tracked
    if ( numTrackers > 0 && wheelCountCurrFrame > 0)
      {
        for (int i = lastWheelNum; i < numTrackers; i++)
        {     
          // draw ellipses
          Rect2d tempR = trackers -> getObjects()[i];
          float w = tempR.width;
          float h = tempR.height;
          // move top left of rect to center of ellipse
          float x = tempR.x + w/2;
          float y = tempR.y + w/2; // add back top offset

          RotatedRect rRect = RotatedRect( Point2f(x,y), Size2f(w,h), 0 );
          ellipse( frame, rRect, Scalar(0,255,0), 1, 8);

          // calculate average wheel velocity
          // not calculating correctly - may be tracking errors?
          vXavg[i] = ( vXavg[i] + tempR.x ) / 2.0; // left rect ref

          // display wheel velocity and number
          stringstream stream;
          stream << fixed << setprecision(1) << vXavg[i] << "px/frame";
          string velocityText = stream.str();

          Point velocityLoc = Point(x-w, y+h);
          Point numLoc = Point(x+w/2, y-h/2);
          string wheelNum = to_string(i + 1);

          putText(frame, velocityText, velocityLoc, 
              FONT_HERSHEY_DUPLEX, 1.5, cvScalar(0,255,0), 2, CV_AA); // velocity
          putText(frame, wheelNum, numLoc, 
              FONT_HERSHEY_DUPLEX, 3, cvScalar(0,255,0), 2, CV_AA); // number
        } 
      }
    
    // draw vertical line in center of frame
    line(frame, Point(frame_width/2,0), Point(frame_width/2,frame_height), Scalar(0,0,255), 3, CV_AA, 0);

    // draw wheel counter
    Point counterLoc = Point(frame_width-frame_width*.1, frame_height*.15);
    putText(frame, to_string(wheelCountTotal), counterLoc, 
        FONT_HERSHEY_DUPLEX, 5, cvScalar(0,255,0), 2, CV_AA); // number

    // show frames 
    imshow("Processed Frame", frameGray);
    waitKey(waitTime);

    imshow("Output Frame", frame);
    waitKey(waitTime);

    //write the video frame to the file
    oVideoWriter.write(frame);   
  }

  // close video file
  oVideoWriter.release();
    
  return 0;
} // end main loop
