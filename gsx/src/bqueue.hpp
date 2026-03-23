#pragma once
#include <deque>
#include <mutex>
#include <condition_variable>
#include <stdexcept>


template<typename T>
class BoundedBlockingQueue {
public:
    explicit BoundedBlockingQueue(size_t capacity) : cap_(capacity) {
        if (cap_ == 0) throw std::invalid_argument("capacity must be > 0");
    }
    BoundedBlockingQueue(const BoundedBlockingQueue&) = delete;
    BoundedBlockingQueue& operator=(const BoundedBlockingQueue&) = delete;
    
    /// @brief Close the queue: reject new pushes; pop returns false when queue is exhausted
    void close() {
        std::scoped_lock lk(m_);
        closed_ = true;
        not_full_.notify_all();
        not_empty_.notify_all();
    }

    void clear() {
        std::scoped_lock lk(m_);
        q_.clear();
        not_full_.notify_all();
    }

    /// @brief Push element to queue, returns false if queue is closed
    bool push(T value) {
        std::unique_lock lk(m_);
        auto can_push = [&] { return closed_ || q_.size() < cap_; };
        not_full_.wait(lk, can_push);
        if (closed_) return false;                           // closed
        q_.push_back(std::move(value));
        lk.unlock();
        not_empty_.notify_one();
        return true;
    }

    /// @brief Pop element from queue, returns false if queue is closed and empty
    bool pop(T& out) {
        std::unique_lock lk(m_);
        auto can_pop = [&] { return closed_ || !q_.empty(); };
        not_empty_.wait(lk, can_pop);
        if (q_.empty()) return false;                        // closed and empty
        out = std::move(q_.front());
        q_.pop_front();
        lk.unlock();
        not_full_.notify_one();
        return true;
    }

    size_t size() const {
        std::scoped_lock lk(m_);
        return q_.size();
    }

    bool closed() const {
        std::scoped_lock lk(m_);
        return closed_;
    }

private:
    mutable std::mutex m_;
    std::condition_variable_any not_full_;
    std::condition_variable_any not_empty_;
    std::deque<T> q_;
    size_t cap_;
    bool closed_ = false;
};
