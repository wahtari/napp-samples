#include "Camera.hpp"

using std::string, std::cout, std::endl, std::flush;

namespace nlab::samples {

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
        throw std::runtime_error(msg + ": " + vimbaErrorCodeMessage(err));
    }
}

//#####################//
//### FrameObserver ###//
//#####################//

FrameObserver::FrameObserver(AVT::VmbAPI::CameraPtr cam, VmbPixelFormatType pxFmt, FrameCallback cb) : 
    IFrameObserver(cam),
    cam_(cam),
    pxFmt_(pxFmt),
    cb_(cb)
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
    cv::Mat mat(cv::Size(nWidth, nHeight), srcType, (void*)pImage);

    // Queue frame back to camera for next acquisition.
    cam_->QueueFrame(frame);

    // Resize and convert, if necessary.
    if (pxFmt_ == VmbPixelFormatBayerRG8) {
        cv::cvtColor(mat, mat, cv::COLOR_BayerRG2RGB_EA); // Hint: 2RGB ist required for a valid BGR image. This seems to be an OpenCV bug.
    }
    cv::resize(mat, mat, cv::Size(), 0.5, 0.5, 0);

    // Execute the callback.
    cb_(mat);
}

//##############//
//### Camera ###//
//##############//

Camera::Camera() : 
    avtSystem_(VimbaSystem::GetInstance()),
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

bool Camera::start(int numFrameBuffers, FrameCallback cb) {
    // Retrieve a list of found cameras.
    string camID;
    CameraPtrVector cams;
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
    FeaturePtr pxFmtFtr;
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
    FeaturePtr expAutoFtr;
    avtErrorCheck(avt_->GetFeatureByName("ExposureAuto", expAutoFtr), "vimba get exposure auto");
    avtErrorCheck(expAutoFtr->SetValue("Continuous"), "vimba set exposure auto");

    // Create FrameObserver and start asynchronous image acquisiton.
    err = avt_->StartContinuousImageAcquisition(numFrameBuffers, IFrameObserverPtr(new FrameObserver(avt_, pxFmt, cb)));
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

} // End of namespace.
