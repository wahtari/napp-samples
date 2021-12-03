/*
 * Wahtari nApp Samples
 * Copyright (c) 2021 Wahtari GmbH
 *
 * All source code in this file is subject to the included LICENSE file.
 */

#include <atomic>
#include <string>
#include <iostream>
#include <stdio.h>
#include <chrono>
#include <queue>
#include <thread>
#include <condition_variable>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>

#include <libnlab-ctrl.hpp>
#include <VimbaCPP/Include/VimbaCPP.h>

#include "MJPEGStreamer.hpp"
#include "BufferedChannel.hpp"
#include "Camera.hpp"

using namespace std::chrono;
using namespace nlab;

using std::string, std::cout, std::cerr, std::endl, std::flush;
using std::atomic, std::mutex, std::unique_lock;
using std::vector, std::thread, std::to_string;
using cv::Mat, cv::Point, cv::Size, cv::Rect, cv::Scalar;
using nadjieb::MJPEGStreamer;

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

std::atomic<bool> interrupt = false;

void interruptHandler(int signum) {
    interrupt = true;
    cerr << "Interrupted! Received signal "+to_string(signum)+". Stopping now..." << endl;
}

bool interrupted() {
    return interrupt;
}

//################//
//### Channels ###//
//################//

samples::BufferedChannel<Mat>           jpegEncodeChan(JPEG_ENCODING_CHANNEL_SIZE);
samples::BufferedChannel<vector<uchar>> videoChan(VIDEO_CHANNEL_SIZE);
samples::BufferedChannel<Mat>           infChan(INFERENCE_CHANNEL_SIZE);
samples::BufferedChannel<vector<Rect>>  infResChan(INFERENCE_RESULT_CHANNEL_SIZE);

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
        std::this_thread::sleep_for(999ms);

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
    
    const vector<int> encodeParams = {cv::IMWRITE_JPEG_QUALITY, 90};
    const auto textPos1 = Point(20, 40);
    const auto textPos2 = Point(20, 70);
    const auto textPos3 = Point(20, 100);
    const auto textPos4 = Point(20, 130);
    const int fontType = cv::FONT_HERSHEY_SIMPLEX;
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
                cv::rectangle(mat, boxes[i], boxColor);
            }
            boxes.clear();
        }

        // Draw FPS counter.
        cv::putText(mat, to_string(videoFPSCurrent) + " FPS",      textPos1, fontType, fontScale, textColor, fontThickness);
        cv::putText(mat, to_string(camFPSCurrent)   + " FPS Cam",  textPos2, fontType, fontScale, textColor, fontThickness);
        cv::putText(mat, to_string(jpegFPSCurrent)  + " FPS JPEG", textPos3, fontType, fontScale, textColor, fontThickness);
        cv::putText(mat, to_string(infFPSCurrent)   + " FPS Inf",  textPos4, fontType, fontScale, textColor, fontThickness);

        // Encode and send to video routine.
        cv::imencode(".jpg", mat, buf, encodeParams);
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
            std::this_thread::sleep_for(remaining);
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

//#############################//
//### Camera Frame Callback ###//
//#############################//

void cameraFrameCallback(const cv::Mat& mat) {
    // Send the frame to both the encoding and inference routine.
    jpegEncodeChan.write(mat);
    infChan.write(mat);
    // Increment the counter.
    camFPSCounter++;
}

//############//
//### Main ###//
//############//

int main(int argc, char* argv[]) {
    // Register signal handler to detect interrupts (e.g. Ctrl+C, docker stop, ...).  
    signal(SIGINT, interruptHandler);
    signal(SIGTERM, interruptHandler);

    // Create camera.
    samples::Camera cam = samples::Camera();
    cam.printSystemVersion();

    // Start the camera.
    bool ok = cam.start(MAX_FRAME_BUFFERS, cameraFrameCallback);
    if (!ok) {
        return 1;
    }

    // Open the Controller and switch on all LEDs to 20% brightness.
    ctrl::Controller::Ptr ctrl;
    try {
        // Get a the list of available controllers.
        // Simply open the first one.
        vector<ctrl::Info> infoList = ctrl::Controller::list();
        if (infoList.size() == 0) {
            cerr << "no controller found" << endl;
            return 2;
        }

        // Open the controller.
        ctrl = ctrl::Controller::open(infoList[0].backendID, infoList[0].devPath, {.stateDir = "/tmp/nlab-ctrl-state"});

        vector<ctrl::LED> leds = ctrl->getLEDs();
        for (const auto& led : leds) {
            ctrl->setLEDStrobe(led.id, false);
            ctrl->setLEDBrightness(led.id, 20);
            ctrl->setLED(led.id, true);
        }
    } catch (const ctrl::Exception& e) {
        cerr << "controller exception! code: " << to_string(e.code()) << ", message: " << e.what() << endl;
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
        infThreads.push_back(thread(inferenceRoutine, "/model.bin", "/model.xml"));
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
        vector<ctrl::LED> leds = ctrl->getLEDs();
        for (const auto& led : leds) {
            ctrl->setLED(led.id, false);
        }
    } catch (const ctrl::Exception& e) {
        cerr << "controller exception! code: " << to_string(e.code()) << ", message: " << e.what() << endl;
        return 4;
    }

    ctrl->close();
    return 0;
}