#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>

// Thread-safe FIFO queue with an optional capacity bound (0 = unbounded).
// push() blocks while full, pop() blocks while empty. After close(), push()
// is rejected and pop() drains the remaining items, then returns false.
template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t capacity = 0) : capacity_(capacity) {}

    bool push(T item) {
        std::unique_lock<std::mutex> lk(m_);
        notFull_.wait(lk, [&] { return closed_ || capacity_ == 0 || q_.size() < capacity_; });
        if (closed_) return false;
        q_.push_back(std::move(item));
        notEmpty_.notify_one();
        return true;
    }

    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(m_);
        notEmpty_.wait(lk, [&] { return closed_ || !q_.empty(); });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        notFull_.notify_one();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lk(m_);
            closed_ = true;
        }
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

private:
    std::mutex m_;
    std::condition_variable notEmpty_, notFull_;
    std::deque<T> q_;
    size_t capacity_;
    bool closed_ = false;
};
