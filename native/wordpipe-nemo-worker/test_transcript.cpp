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
}
