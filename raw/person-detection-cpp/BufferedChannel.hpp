#pragma once

#include <queue>
#include <condition_variable>

namespace nlab::samples {

// The BufferedChannel is a thread-safe queue with a maximum size,
// that offers an atomic read with an optional timeout.
// If a write would exceed the maximum size, the oldest element is dropped.
template<typename T>
class BufferedChannel {
public:
    BufferedChannel(int size) : size_(size) { }

    bool read(T& out, std::chrono::milliseconds timeout = std::chrono::milliseconds(0)) {
        std::unique_lock<std::mutex> lock(mx_);
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
    std::mutex              mx_;
    std::queue<T>           queue_;
    std::condition_variable cond_;
    int                     size_;    
};

} // End of namespace.