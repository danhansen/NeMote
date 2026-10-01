#pragma once
#include <atomic>
#include <functional>
#include <thread>

// Exclusive recognition ownership, with recovery after a capture failure.
// Destruction, stdin EOF, and shutdown all stop and join before model release.
class SessionThread {
    std::atomic<bool> stop_{false}, done_{true};
    std::thread thread_;
public:
    ~SessionThread() { stop(); }
    bool active() const { return !done_.load(std::memory_order_acquire); }
    void stop() {
        stop_.store(true);
        if (thread_.joinable()) thread_.join();
        done_.store(true);
    }
    void start(std::function<void(std::atomic<bool>&)> run, std::function<void()> complete = [] {}) {
        if (active()) return;
        stop(); // Reap a completed session before starting another.
        stop_.store(false);
        done_.store(false);
        try {
            thread_ = std::thread([this, run = std::move(run), complete = std::move(complete)] {
                run(stop_);
                done_.store(true, std::memory_order_release);
                complete(); // Publish stopped only after the next start is allowed.
            });
        } catch (...) {
            done_.store(true);
            throw;
        }
    }
};
