#pragma once
#include <string>

// The pinned English Sparrowhawk grammar separates sentence punctuation into
// tokens ("I paid $1 ."). Reattach it for dictation without altering other
// languages' typography, decimal digits, hyphens, or the ITN-off transcript.
inline std::string format_english_itn(const std::string& text) {
    std::string output;
    for (char character : text) {
        if (std::string(".,!?;:").find(character) != std::string::npos)
            while (!output.empty() && output.back() == ' ') output.pop_back();
        output += character;
    }
    return output;
}
