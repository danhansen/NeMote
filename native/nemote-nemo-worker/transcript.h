#pragma once
#include <string>

// Upstream results are per-utterance snapshots, not append-only token deltas.
class Transcript {
    std::string committed_, current_;
public:
    void update(const std::string& text, bool final, const std::string& punctuation = "") {
        committed_ += punctuation;
        current_ = committed_;
        if (!current_.empty() && !text.empty()) current_ += ' ';
        current_ += text;
        if (final) committed_ = current_;
    }
    const std::string& text() const { return current_; }
};
