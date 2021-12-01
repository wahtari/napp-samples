# nApp Samples
A collection of sample applications that you can use as a starting point to write your first custom *nApp*.

The samples are classified into two categories:
- **Raw**: Most of the functionality is implemented without libraries/SDKs by Wahtari
- **nGin**: The nGin SDK is used that handles most of the complex functionality for you.

If you have an existing code base that you want to reuse with as few modifications as possible, the **raw** examples give you a good starting point.
But in case you want to pass the heavy lifting on and focus on the important details, the **nGin** samples are the perfect fit.

## Raw
### Person Detection C++
In this [sample](raw/person-detection-cpp) we detect persons using a pre-trained person-detection-model.  

The sample implements the following things:  
- Communication with camera over proprietary driver SDK (in this case the VimbaSDK by AlliedVision)  
- AI inference on Intel(R) Myriad(R) X Chips using OpenCV Communication with onboard controller to control integrated lighting, motorized lens, status LEDs, GPIO Pins using the small Wahtari controller-libs library.  
- Rate limiting video streams to a fixed FPS setting.  
- Opening a video server to display the result stream.

## nGin
- TBA

## Help  
We have a dedicated [Wiki](https://wiki.wahtari.io) that serves as a helpful guide for most questions.  
If you have problems the Wiki does not solve, you can create an issue here on GitHub, or write as an email to `support[at]wahtari.io`
