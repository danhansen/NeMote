#pragma once
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// English PnC is a token classifier, not a rewriter. Keep a one-to-one map
// back to written-form tokens so stripping its input never destroys ITN data.
inline std::string pnc_lower(std::string text) {
    for (auto& c : text) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    return text;
}
inline bool pnc_sentence_mark(char c) {
    return std::string(".,!?;:\"()[]{}").find(c) != std::string::npos;
}
inline std::string strip_sentence_punctuation(const std::string& text) {
    std::istringstream stream(text);
    std::string result, word;
    while (stream >> word) {
        // Strip only outer sentence marks, not decimal separators, apostrophes,
        // address syntax or punctuation inside a written-form token.
        auto first = word.begin(), last = word.end();
        while (first != last && pnc_sentence_mark(*first)) {
            // Leading decimals and relative/hidden paths are written-form
            // data, not a sentence mark: .5, ./File, .config.
            if (*first == '.' && last - first > 1 && *(first + 1) != '.') break;
            ++first;
        }
        while (last != first && pnc_sentence_mark(*(last - 1))) --last;
        word = std::string(first, last);
        if (word.empty()) continue;
        if (!result.empty()) result += ' ';
        result += word;
    }
    return result;
}

struct PncInput {
    struct Word { std::string original, input; bool protect; };
    std::vector<Word> words;
    std::string text;
    explicit PncInput(const std::string& formatted) {
        std::istringstream stream(strip_sentence_punctuation(formatted));
        std::string word;
        while (stream >> word) {
            std::string clean;
            bool protect = false;
            // Case can carry meaning in units/acronyms (mL, MHz, USD, PM).
            // Strip it for classification, but don't overwrite that spelling.
            for (size_t i = 1; i < word.size(); ++i)
                if (word[i] >= 'A' && word[i] <= 'Z') protect = true;
            for (unsigned char c : word) {
                if (c >= '0' && c <= '9') protect = true;
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c >= 128 || c == '\'') clean += char(c);
                else protect = true;
            }
            // Preserve standalone written-form symbols too (e.g. '$ 12').
            if (clean.empty()) clean = "symbol";
            clean = pnc_lower(clean);
            words.push_back({word, clean, protect});
            if (!text.empty()) text += ' ';
            text += clean;
        }
    }
    std::string restore(const std::string& prediction) const {
        std::istringstream stream(prediction);
        std::string result, word;
        for (const auto& source : words) {
            if (!(stream >> word) || word.size() < source.input.size() ||
                pnc_lower(word.substr(0, source.input.size())) != source.input)
                throw std::runtime_error("PnC changed transcript words unexpectedly");
            const auto punctuation = word.substr(source.input.size());
            if (!punctuation.empty() && punctuation != "." && punctuation != "," && punctuation != "?")
                throw std::runtime_error("PnC returned unsupported punctuation");
            if (!result.empty()) result += ' ';
            result += source.protect ? source.original + punctuation : word;
        }
        if (stream >> word) throw std::runtime_error("PnC added transcript words unexpectedly");
        return result;
    }
};

template<class Normalize, class Infer>
std::string postprocess_with_pnc(const std::string& raw, Normalize normalize, Infer infer) {
    PncInput input(normalize(strip_sentence_punctuation(raw)));
    return input.restore(infer(input.text));
}
