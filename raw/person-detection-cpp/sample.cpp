#include <atomic>
#include <string>
#include <cstring>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdio.h>
#include <csignal>
#include <chrono>
#include <queue>
#include <thread>
#include <condition_variable>

#include <libnlab-ctrl.hpp>
#include <VimbaCPP/Include/VimbaCPP.h>

#include "MJPEGStreamer.hpp"

using namespace std;
using namespace cv;
using namespace std::chrono;
using namespace nlab::ctrl;
using MJPEGStreamer = nadjieb::MJPEGStreamer;

// Maximum size of the inference channel.
#define INFERENCE_CHANNEL_SIZE 3
// Maximum size of the inference result channel.
#define INFERENCE_RESULT_CHANNEL_SIZE 3
// Maximum size of the jpeg encoding channel.
#define JPEG_ENCODING_CHANNEL_SIZE 3
// Maximum size of the video channel.
#define VIDEO_CHANNEL_SIZE 1
// Maximum number of frame buffers used to read frames off the camera.
#define MAX_FRAME_BUFFERS 5
// How many fps the resulting video should have.
#define FPS 30
// How many VPU devices should be used.
#define NUM_VPUS 3
// How many threads should be used to encode frames to jpeg.
#define NUM_JPEG_ENCODERS 3
// The local port for the MJPEG server.
#define MJPEG_PORT 8080

//##################//
//### Interrupts ###//
//##################//

mutex interruptMx;
bool interrupt;

void interruptHandler(int signum) {
   interruptMx.lock();
   interrupt = true;
   cout << "Interrupted! Received signal "+to_string(signum)+". Stopping now..." << endl;
   interruptMx.unlock();
}

bool interrupted() {
    bool intr;
    interruptMx.lock();
    intr = interrupt;
    interruptMx.unlock();
    return intr;
}

//################//
//### Channels ###//
//################//

// The BufferedChannel is a thread-safe queue with a maximum size,
// that offers an atomic read with an optional timeout.
template<typename T>
class BufferedChannel {
public:
    BufferedChannel(int size) : size_(size) { }

    bool read(T& out, milliseconds timeout = milliseconds(0)) {
        unique_lock<mutex> lock(mx_);
        if (queue_.empty() && !cond_.wait_for(lock, timeout, [&]{ return queue_.size() > 0; })) {
            // Nothing available.
            return false;
        }
        
        out = queue_.front();
        queue_.pop();
        return true;
    }

    void write(const T in) {
        mx_.lock();
        if (queue_.size() >= size_) {
            queue_.pop();
        }
        queue_.push(in);
        cond_.notify_one();
        mx_.unlock();
    }

private:
    mutex              mx_;
    queue<T>           queue_;
    condition_variable cond_;
    int                size_;    
};

BufferedChannel<Mat>           jpegEncodeChan(JPEG_ENCODING_CHANNEL_SIZE);
BufferedChannel<vector<uchar>> videoChan(VIDEO_CHANNEL_SIZE);
BufferedChannel<Mat>           infChan(INFERENCE_CHANNEL_SIZE);
BufferedChannel<vector<Rect>>  infResChan(INFERENCE_RESULT_CHANNEL_SIZE);

//################//
//### Counters ###//
//################//

atomic<uint16_t> camFPSCounter{0};
atomic<uint16_t> camFPSCurrent{0};

atomic<uint16_t> jpegFPSCounter{0};
atomic<uint16_t> jpegFPSCurrent{0};

atomic<uint16_t> videoFPSCounter{0};
atomic<uint16_t> videoFPSCurrent{0};

atomic<uint16_t> infFPSCounter{0};
atomic<uint16_t> infFPSCurrent{0};

//###################//
//### FPS Routine ###//
//###################//

// fpsRoutine roughly ticks every second and resets all FPS counters to 0.
void fpsRoutine() {
    while (!interrupted()) {
        // Wait for slightly less than 1s and then calculate the current fps.
        this_thread::sleep_for(999ms);

        // Read the current fps counter and reset them to 0 simultaniously.
        infFPSCurrent = infFPSCounter.exchange(0);
        videoFPSCurrent = videoFPSCounter.exchange(0);
        jpegFPSCurrent = jpegFPSCounter.exchange(0);
        camFPSCurrent = camFPSCounter.exchange(0);
    }
}

//###########################//
//### JPEG Encode Routine ###//
//###########################//

// jpegEncodeRoutine reads in an endless loop Mats off of the jpeg encode channel,
// checks the inference result channel for potential results, and draws them along
// with the FPS counters onto the image. 
// The Mat is then jpeg encoded and pushed to the video channel.
void jpegEncodeRoutine() {
    Mat mat;
    vector<uchar> buf;
    vector<Rect> boxes;
    int i;
    bool ok;
    
    const vector<int> encodeParams = {IMWRITE_JPEG_QUALITY, 90};
    const auto textPos1 = Point(20, 40);
    const auto textPos2 = Point(20, 70);
    const auto textPos3 = Point(20, 100);
    const auto textPos4 = Point(20, 130);
    const int fontType = FONT_HERSHEY_SIMPLEX;
    const double fontScale = 0.9;
    const auto textColor = Scalar(0, 255);
    const int fontThickness = 2;
    const auto boxColor = Scalar(0, 0, 255);

    while(!interrupted()) {
        ok = jpegEncodeChan.read(mat, 100ms);
        if (!ok) {
            // No frame available.
                continue;
        }

        // Draw detection results, if available.
        if (infResChan.read(boxes)) {
            for (i = 0; i < boxes.size(); ++i) {
                rectangle(mat, boxes[i], boxColor);
            }
            boxes.clear();
        }

        // Draw FPS counter.
        putText(mat, to_string(videoFPSCurrent) + " FPS",      textPos1, fontType, fontScale, textColor, fontThickness);
        putText(mat, to_string(camFPSCurrent)   + " FPS Cam",  textPos2, fontType, fontScale, textColor, fontThickness);
        putText(mat, to_string(jpegFPSCurrent)  + " FPS JPEG", textPos3, fontType, fontScale, textColor, fontThickness);
        putText(mat, to_string(infFPSCurrent)   + " FPS Inf",  textPos4, fontType, fontScale, textColor, fontThickness);

        // Encode and send to video routine.
        imencode(".jpg", mat, buf, encodeParams);
        videoChan.write(buf);
        jpegFPSCounter++;
    }
}

//#####################//
//### Video Routine ###//
//#####################//

// videoRoutine reads in an endless loop jpeg encoded frames off of the video channel,
// and publishes them via a MJPEG server.
// The routine is strictly clocked so that frames are published in fixed intervals,
// resulting in a steady video with roughly the defined number of frames per second.
void videoRoutine() {
    vector<uchar> lastBuf;
    const milliseconds interval = (1000ms / FPS) - 1ms;
    milliseconds start, timeout, remaining;
    MJPEGStreamer streamer;
    bool ok;

    // Start MJPEG server.
    streamer.start(MJPEG_PORT, 1);

    cout << "MJPEG server listening on port " << to_string(MJPEG_PORT) << endl;

    while (true) {
        start = duration_cast<milliseconds>(system_clock::now().time_since_epoch());

        if (interrupted()) {
            cout << "Stopping MJPEG server" << endl;
            streamer.stop();
            return;
        }

        timeout = interval - (duration_cast<milliseconds>(system_clock::now().time_since_epoch()) - start);

        // Wait, if the channel is empty, but a maximum of time that is left in our interval.
        // We reuse the last buffer, if no new one got available in time.
        videoChan.read(lastBuf, timeout);

        if (lastBuf.size() > 0) {
            streamer.publish("/stream", string(lastBuf.begin(), lastBuf.end()));
            videoFPSCounter++;
        }

        // Wait remaining time of interval.
        remaining = interval - (duration_cast<milliseconds>(system_clock::now().time_since_epoch()) - start);
        if (remaining > 0ns) {
            this_thread::sleep_for(remaining);
        }
    }
}

//#########################//
//### Inference Routine ###//
//#########################//

// inferenceRoutine reads in an endless loop frames off of the inference channel. 
// It forwards each frame through its detection model and pushes the results to 
// the inferenece result channel.
// Results are probably shown for a different frame than they have been created for.
// However, these slight deviations in timing are not important in this sample.
void inferenceRoutine(string model_path, string config_path) {
    cv::dnn::DetectionModel dm = cv::dnn::DetectionModel(model_path, config_path);
    dm.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    dm.setPreferableTarget(cv::dnn::DNN_TARGET_MYRIAD);

    Mat frame;
    vector<int> classIDs;
    vector<float> confs;
    vector<Rect> boxes;
    bool ok;

    while (true) {
        if (interrupted()) {
            return;
        }

        // Wait, if the channel is empty.
        ok = infChan.read(frame, 100ms);
        if (!ok || frame.total() == 0) {
            // No frame available or empty.
            continue;
        }

        // Do inference.
        dm.detect(frame, classIDs, confs, boxes);
        infFPSCounter++;

        // TODO: show class and confidence as well?
        infResChan.write(boxes);
    }
}

//##############//
//### Camera ###//
//##############//

// vimbaErrorCodeMessage converts the given vimba error type to a human-readable string.
// Copied from VimbaExamples' file "ErrorCodeToMessage.h"
string vimbaErrorCodeMessage(VmbErrorType err) {
    switch(err) {
    case VmbErrorSuccess:        return "Success.";
    case VmbErrorInternalFault:  return "Unexpected fault in VmbApi or driver.";
    case VmbErrorApiNotStarted:  return "API not started.";
    case VmbErrorNotFound:       return "Not found.";
    case VmbErrorBadHandle:      return "Invalid handle ";
    case VmbErrorDeviceNotOpen:  return "Device not open.";
    case VmbErrorInvalidAccess:  return "Invalid access.";
    case VmbErrorBadParameter:   return "Bad parameter.";
    case VmbErrorStructSize:     return "Wrong DLL version.";
    case VmbErrorMoreData:       return "More data  returned than memory provided.";
    case VmbErrorWrongType:      return "Wrong type.";
    case VmbErrorInvalidValue:   return "Invalid value.";
    case VmbErrorTimeout:        return "Timeout.";
    case VmbErrorOther:          return "TL error.";
    case VmbErrorResources:      return "Resource not available.";
    case VmbErrorInvalidCall:    return "Invalid call.";
    case VmbErrorNoTL:           return "TL not loaded.";
    case VmbErrorNotImplemented: return "Not implemented.";
    case VmbErrorNotSupported:   return "Not supported.";
    default:                     return "Unknown";
    }
}

// avtErrorCheck checks, if the given vimba error type indicates a successful operation.
// If not, a runtime_error exception is thrown, with the given msg as prefix and the vimba error code message.
void avtErrorCheck(const VmbErrorType err, const string& msg) {
    if (err != VmbErrorSuccess) {
        throw runtime_error(msg + ": " + vimbaErrorCodeMessage(err));
    }
}

// The FrameObserver class implements the vimba IFrameObserver interface and provides
// a callback to handle new frames read off of the camera.
class FrameObserver : public AVT::VmbAPI::IFrameObserver {
public:
    FrameObserver(AVT::VmbAPI::CameraPtr cam, VmbPixelFormatType pxFmt);

    // FrameReceived is the callback that handles newly read frames.
    // We convert the frame to an OpenCV Mat and distribute it to both the 
    // jpeg encoding routines, as well as the inference routines.
    void FrameReceived(const AVT::VmbAPI::FramePtr frame);

private:
    AVT::VmbAPI::CameraPtr cam_;
    VmbPixelFormatType     pxFmt_;
};

FrameObserver::FrameObserver(AVT::VmbAPI::CameraPtr cam, VmbPixelFormatType pxFmt) : 
    IFrameObserver(cam),
    cam_(cam),
    pxFmt_(pxFmt)
{}

void FrameObserver::FrameReceived(const AVT::VmbAPI::FramePtr frame) {
    if (frame == nullptr) {
        cout << "frameReceived: frame was null" << endl;
        return;
    }

    // Convert frame to a OpenCV matrix.
    // Retrieve size and image.
    VmbUint32_t nImageSize = 0; 
    VmbErrorType err = frame->GetImageSize(nImageSize);
    if (err != VmbErrorSuccess) {
        cout << "frameReceived: get image size " << vimbaErrorCodeMessage(err) << endl;
        return;
    }
    VmbUint32_t nWidth = 0;
    err = frame->GetWidth(nWidth);
    if (err != VmbErrorSuccess) {
        cout << "frameReceived: get width " << vimbaErrorCodeMessage(err) << endl;
        return;
    }
    VmbUint32_t nHeight = 0;
    err = frame->GetHeight(nHeight);
    if (err != VmbErrorSuccess) {
        cout << "frameReceived: get height " << vimbaErrorCodeMessage(err) << endl;
        return;
    }
    VmbUchar_t* pImage = NULL;
    err = frame->GetImage(pImage);
    if (err != VmbErrorSuccess) {
        cout << "frameReceived: get image " << vimbaErrorCodeMessage(err) << endl;
        return;
    }

    // convert image to OpenCV Mat.
    int srcType;
    if (pxFmt_ == VmbPixelFormatMono8 || pxFmt_ == VmbPixelFormatBayerRG8) {
        srcType = CV_8UC1;
    } else {
        srcType = CV_8UC3;
    }
    Mat mat(Size(nWidth, nHeight), srcType, (void*)pImage);

    // Queue frame back to camera for next acquisition.
    cam_->QueueFrame(frame);

    // Resize and convert, if necessary.
    if (pxFmt_ == VmbPixelFormatBayerRG8) {
        cvtColor(mat, mat, COLOR_BayerRG2RGB_EA); // Hint: 2RGB ist required for a valid BGR image. This seems to be an OpenCV bug.
    }
    resize(mat, mat, Size(), 0.5, 0.5, 0);

    // Send the frame to both the encoding and inference routine.
    jpegEncodeChan.write(mat);
    infChan.write(mat);
    camFPSCounter++;
}

// The Camera class is a thin wrapper around a vimba Camera.
// It encapsulates the setup and teardown code and provides easy to use methods
// to quickly get a camera up and running.
// This class is not thread-safe.
class Camera {
public:
    Camera();
    ~Camera();

    // printSystemVersion prints the semver version of the Vimba SDK.
    void printSystemVersion();
    // start opens the first found camera and starts the image acquisition on it.
    bool start();
    // stop stops the image acquisition of the started camera.
    // It is valid to call stop multiple times. 
    // The destructor makes sure to call stop as well.
    void stop();

private:
    AVT::VmbAPI::VimbaSystem& avtSystem_;
    AVT::VmbAPI::CameraPtr    avt_;
    bool                      opened_;
    bool                      grabbing_;
};

Camera::Camera() : 
    avtSystem_(AVT::VmbAPI::VimbaSystem::GetInstance()),
    opened_(false),
    grabbing_(false)
{
    // Start Vimba.
    avtErrorCheck(avtSystem_.Startup(), "vimba system startup");
}

Camera::~Camera() {
    stop();
    avtSystem_.Shutdown();
    cout << "Vimba closed" << endl << flush;
}

void Camera::printSystemVersion() {
    // Print Vimba version.
    VmbVersionInfo_t info;
    avtErrorCheck(avtSystem_.QueryVersion(info), "vimba query version");
    cout << "Vimba C++ API Version " << info.major << "." << info.minor << "." << info.patch << endl;
}

bool Camera::start() {
    // Retrieve a list of found cameras.
    string camID;
    AVT::VmbAPI::CameraPtrVector cams;
    avtErrorCheck(avtSystem_.GetCameras(cams), "vimba get cameras");
    if (cams.size() <= 0) {
        cout << "no camera found" << endl;
        return false;
    }

    // Open the first camera for now.
    avtErrorCheck(cams[0]->GetID(camID), "vimba cam get id");
    avtErrorCheck(avtSystem_.OpenCameraByID(camID.c_str(), VmbAccessModeFull, avt_), "vimba open camera by id");
    opened_ = true;

    // Set pixel format.
    AVT::VmbAPI::FeaturePtr pxFmtFtr;
    avtErrorCheck(avt_->GetFeatureByName("PixelFormat", pxFmtFtr), "vimba get pixel format");

    // Try to set BayerRG8, then BGR, then Mono.
    VmbPixelFormatType pxFmt = VmbPixelFormatBayerRG8;
    VmbErrorType err = pxFmtFtr->SetValue(pxFmt);
    if (err != VmbErrorSuccess) {
        pxFmt = VmbPixelFormatBgr8;
        err = pxFmtFtr->SetValue(pxFmt);
        if (err != VmbErrorSuccess) {
            // Fall back to Mono.
            pxFmt = VmbPixelFormatMono8;
            avtErrorCheck(pxFmtFtr->SetValue(pxFmt), "vimba set pixel format");
        }
    }

    // Set auto exposure.
    AVT::VmbAPI::FeaturePtr expAutoFtr;
    avtErrorCheck(avt_->GetFeatureByName("ExposureAuto", expAutoFtr), "vimba get exposure auto");
    avtErrorCheck(expAutoFtr->SetValue("Continuous"), "vimba set exposure auto");

    // Create FrameObserver and start asynchronous image acquisiton.
    err = avt_->StartContinuousImageAcquisition(MAX_FRAME_BUFFERS, AVT::VmbAPI::IFrameObserverPtr(new FrameObserver(avt_, pxFmt)));
    avtErrorCheck(err, "vimba start continuous image acquisition");
    grabbing_ = true;

    return true;
}

void Camera::stop() {
    if (!opened_) {
        return;
    }

    // Stop image acquisition.
    if (grabbing_) {
        avt_->StopContinuousImageAcquisition();
        grabbing_ = false;
    }

    avt_->Close();
    cout << "Camera closed" << endl;
    opened_ = false;
}

//############//
//### Main ###//
//############//

int main(int argc, char* argv[]) {
    // Register signal handler to detect interrupts (e.g. Ctrl+C, docker stop, ...).  
    signal(SIGINT, interruptHandler);
    signal(SIGTERM, interruptHandler);

    // Create Vimba camera.
    Camera cam = Camera();
    cam.printSystemVersion();

    // Start the camera.
    bool ok = cam.start();
    if (!ok) {
        return 1;
    }

    // Open the Controller and switch on all LEDs to 20% brightness.
    Controller::Ptr ctrl;
    try {
        // Get a the list of available controllers.
        // Simply open the first one.
        vector<Info> infoList = Controller::list();
        if infoList.size() == 0 {
            cout << "no controller found" << endl;
            return 2;
        }

        // Open the controller.
        ctrl = Controller::open(infoList[0].backendID, infoList[0].devPath, {.stateDir = "/tmp/nlab-ctrl-state"});

        vector<LED> leds = ctrl->getLEDs();
        for (const auto& led : leds) {
            ctrl->setLED(led.id, true);
            ctrl->setLEDStrobe(led.id, false);
            ctrl->setLEDBrightness(led.id, 20);
        }
    } catch (const Exception& e) {
        cout << "controller exception! code: " << to_string(e.code()) << ", message: " << e.what() << endl;
        return 3;
    }

    // Spawn all worker threads.
    thread fpsThread(fpsRoutine);
    vector<thread> jpegThreads;
    for (int i = 0; i < NUM_JPEG_ENCODERS; ++i) {
        jpegThreads.push_back(thread(jpegEncodeRoutine));
    }
    thread videoThread(videoRoutine);
    vector<thread> infThreads;
    for (int i = 0; i < NUM_VPUS; ++i) {
        infThreads.push_back(thread(inferenceRoutine, "/person-detection-0200.bin", "/person-detection-0200.xml"));
    }

    // Wait until all threads have exited.
    fpsThread.join();
    for (int i = 0; i < NUM_JPEG_ENCODERS; ++i) {
        jpegThreads[i].join();
    }
    videoThread.join();
    for (int i = 0; i < NUM_VPUS; ++i) {
        infThreads[i].join();
    }
    cout << "All threads gracefully exited" << endl;

    try {
        vector<LED> leds = ctrl->getLEDs();
        for (const auto& led : leds) {
            ctrl->setLED(led.id, false);
        }
    } catch (const Exception& e) {
        cout << "controller exception! code: " << to_string(e.code()) << ", message: " << e.what() << endl;
        return 4;
    }

    return 0;
}