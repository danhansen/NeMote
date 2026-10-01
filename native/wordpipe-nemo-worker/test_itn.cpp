#include <iostream>
#include "itn.h"
#include "itn_format.h"

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) return 2;
    nemo_speech::asr::postproc::Itn normalizer(argv[1]);
    if (argc == 3) {
        std::cout << format_english_itn(normalizer.normalize(argv[2])) << '\n';
        return normalizer.enabled() ? 0 : 1;
    }
    nemo_speech::asr::postproc::Itn disabled;
    if (!normalizer.enabled() || disabled.enabled()) return 1;
    const std::pair<const char*, const char*> cases[] = {
        {"five hundred two", "502"}, {"third", "3rd"},
        {"Fifteen thousand head of cattle", "15000 head of cattle"},
        {"Alice met Bob", "Alice met Bob"},
        {"one dollar", "$1"}, {"two kilograms", "2 kg"},
        {"I paid one dollar.", "I paid $1 ."},
        {"words already normalized", "words already normalized"}, {"", ""},
    };
    for (const auto& [input, expected] : cases) {
        auto text = normalizer.normalize(input);
        std::cout << input << " -> " << text << '\n';
        if (text != expected || disabled.normalize(input) != input) return 1;
    }
    if (format_english_itn(normalizer.normalize("I paid one dollar.")) != "I paid $1.") return 1;
    if (format_english_itn("Yes , 2.5 kg !") != "Yes, 2.5 kg!") return 1;
    if (format_english_itn("word - word") != "word - word") return 1;
    return 0;
}
