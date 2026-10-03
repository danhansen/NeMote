#include "transcript.h"
#include <stdexcept>

static void expect(const Transcript& transcript, const std::string& text) {
    if (transcript.text() != text) throw std::runtime_error("transcript mismatch: " + transcript.text());
}
int main() {
    Transcript transcript;
    transcript.update("a wrong partial", false);
    transcript.update("a corrected partial", false);
    expect(transcript, "a corrected partial");
    transcript.update("a corrected final", true);
    transcript.update("next utterance", false, ".");
    expect(transcript, "a corrected final. next utterance");
    transcript.update("next utterance complete", true);
    transcript.update("", true);
    expect(transcript, "a corrected final. next utterance complete");
    Transcript reset;
    reset.update("new session", true);
    expect(reset, "new session");
    Transcript empty;
    empty.update("", true);
    expect(empty, "");

    // Test adapter scheduling and display stability, not NVIDIA's ITN grammar.
    PreviewTranscript preview;
    int calls = 0;
    auto normalize = [&](const std::string& raw) {
        ++calls;
        if (raw == "one dollar") return std::string("$1");
        if (raw == "one dollar. two dollars") return std::string("$1. $2");
        if (raw == "one dollar.") return std::string("$1.");
        if (raw == "one dollar. two dollars today") return std::string("$1. $2 today");
        throw std::runtime_error("normalizer received unexpected or formatted input: " + raw);
    };
    auto display = [&](const std::string& expected) {
        if (preview.text() != expected) throw std::runtime_error("preview mismatch: " + preview.text());
    };
    preview.update("one dollar", false, "", normalize);
    display("one dollar");
    if (calls != 0) throw std::runtime_error("interim normalized prematurely");
    preview.update("one dollar", true, "", normalize);
    display("$1");
    preview.update("two", false, ".", normalize);
    display("$1. two");
    preview.update("two dollars", false, "", normalize);
    display("$1. two dollars");
    if (calls != 2) throw std::runtime_error("interim should retain the normalized prefix");
    preview.update("two dollars", true, "", normalize);
    display("$1. $2");
    if (preview.raw_text() != "one dollar. two dollars") throw std::runtime_error("raw text lost");
    preview.update("today", false, "", normalize);
    display("$1. $2 today");
    preview.update("today", true, "", normalize);
    display("$1. $2 today");
    preview.update("", true, "", normalize);
    display("$1. $2 today");
}
