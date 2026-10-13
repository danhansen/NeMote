#include "pnc_format.h"

static void expect(const std::string& actual, const std::string& expected) {
    if (actual != expected) throw std::runtime_error("PnC formatting mismatch: " + actual);
}
int main() {
    expect(strip_sentence_punctuation("Hello, meet me at 3.14. Email Pat@example.com!"),
           "Hello meet me at 3.14 Email Pat@example.com");
    PncInput input("Pay $1,234.50 to Pat@Example.com via https://Example.com/Case at 3:15 PM.");
    expect(input.text, "pay 123450 to patexamplecom via httpsexamplecomcase at 315 pm");
    expect(input.restore("Pay 123450 to patexamplecom, via httpsexamplecomcase at 315 pm."),
           "Pay $1,234.50 to Pat@Example.com, via https://Example.com/Case at 3:15 PM.");
    PncInput other("It's -3.5% on /Case/File.txt and $ 12.");
    expect(other.restore("It's 35 on casefiletxt and symbol 12."),
           "It's -3.5% on /Case/File.txt and $ 12.");
    expect(PncInput("").restore(""), "");
    expect(PncInput("Use .5 and .config in ./Case/File.").restore("Use 5 and config in casefile."),
           "Use .5 and .config in ./Case/File.");
    expect(PncInput("Take 12 mL, not 12 ML.").restore("Take 12 ml not 12 ml."), "Take 12 mL not 12 ML.");
    int stage = 0;
    expect(postprocess_with_pnc("Pay twelve dollars.", [&](const std::string& raw) {
        if (stage++ != 0) throw std::runtime_error("ITN ran out of order");
        expect(raw, "Pay twelve dollars");
        return std::string("Pay $12");
    }, [&](const std::string& plain) {
        if (stage++ != 1) throw std::runtime_error("PnC ran before ITN");
        expect(plain, "pay 12");
        return std::string("Pay 12.");
    }), "Pay $12.");
    for (const auto& wrong : {"wrong", "pay", "pay 123450 to patexamplecom via httpsexamplecomcase at 315 pm extra"}) {
        bool rejected = false;
        try { input.restore(wrong); } catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("PnC lexical mismatch was accepted");
    }
}
