#include "session_thread.h"
#include <chrono>
#include <stdexcept>

static void wait_until(const std::function<bool()>& ready) {
    auto end = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!ready()) {
        if (std::chrono::steady_clock::now() > end) throw std::runtime_error("session test timed out");
        std::this_thread::yield();
    }
}
int main() {
    std::atomic<int> runs{0};
    SessionThread session;
    auto blocking = [&](std::atomic<bool>& stop) {
        runs.fetch_add(1);
        while (!stop.load()) std::this_thread::yield();
    };
    session.start(blocking);
    wait_until([&] { return runs.load() == 1; });
    session.start(blocking); // Duplicate start must not create another decoder.
    session.stop();
    if (runs != 1 || session.active()) throw std::runtime_error("duplicate start/stop failed");
    std::atomic<bool> published{false};
    session.start([&](std::atomic<bool>&) { runs.fetch_add(1); }, [&] {
        if (session.active()) throw std::runtime_error("stopped published while still active");
        published.store(true);
    }); // Capture error/early exit.
    wait_until([&] { return !session.active(); });
    session.start(blocking);
    wait_until([&] { return runs.load() == 3; });
    session.stop();
    if (!published) throw std::runtime_error("completion event missing");
    { SessionThread eof; eof.start(blocking); } // Destructor stops and joins.
}
