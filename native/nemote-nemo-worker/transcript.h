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
    const std::string& finalized_text() const { return committed_; }
};

// SDK-final segments are not necessarily client commits. Keep raw text for
// whole-session normalization and a separately formatted prefix for display.
class PreviewTranscript {
    Transcript raw_;
    std::string formatted_prefix_, display_;
public:
    template<class Normalize>
    void update(const std::string& text, bool final, const std::string& punctuation,
                Normalize normalize) {
        raw_.update(text, final, punctuation);
        if (final || !punctuation.empty())
            formatted_prefix_ = normalize(raw_.finalized_text());
        display_ = formatted_prefix_ + raw_.text().substr(raw_.finalized_text().size());
    }
    const std::string& text() const { return display_; }
    const std::string& raw_text() const { return raw_.text(); }
};
