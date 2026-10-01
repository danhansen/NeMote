pub const BUS_NAME: &str = "dev.wordpipe.Service";
pub const OBJECT_PATH: &str = "/dev/wordpipe/Service";
pub const INTERFACE_NAME: &str = "dev.wordpipe.Service1";

pub const DEFAULT_BACKEND: &str = "parakeet";
pub const DEFAULT_MODEL_PROFILE: &str = "fast";
pub const DEFAULT_MODEL_FAMILY: &str = "english";
pub const DEFAULT_SHORTCUT: &str = "<Control><Alt>space";
pub const DEFAULT_LANGUAGE: &str = "en-US";
pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;
pub const DEFAULT_NUM_THREADS: u32 = 2;

pub const BACKENDS: &[BackendSpec] = &[BackendSpec {
    id: "parakeet",
    title: "Parakeet",
    description: "Rust parakeet-rs Nemotron streaming backend",
}];

pub const MODEL_PROFILES: &[ModelProfileSpec] = &[
    ModelProfileSpec {
        id: "fast",
        title: "Fast",
        description: "FP32 projected-cache model; larger memory and download footprint.",
        build_profile: "fp32-projected",
        output_name: "nemotron-wordpipe-fast-fp32-projected",
        prebuilt_repo: "fractalyzer/wordpipe-nemotron-fast-fp32-projected",
        english_output_name: "nemotron-wordpipe-en-fast-fp32-projected",
        english_prebuilt_repo: "fractalyzer/wordpipe-nemotron-en-fast-dynamic-fp32-projected",
        ort_format: false,
    },
    ModelProfileSpec {
        id: "compact",
        title: "Compact",
        description: "Quantized projected-cache model; generic ONNX or legacy ORT runtime.",
        build_profile: "compact-fixed-shape",
        output_name: "nemotron-wordpipe-compact-fixed-shape",
        prebuilt_repo: "fractalyzer/wordpipe-nemotron-compact-fixed-shape",
        english_output_name: "nemotron-wordpipe-en-compact-fixed-shape",
        english_prebuilt_repo: "fractalyzer/wordpipe-nemotron-en-compact-dynamic-quint8",
        ort_format: true,
    },
];

pub const MODEL_PRESETS: &[ModelPresetSpec] = &[
    ModelPresetSpec {
        id: "compact",
        title: "compact",
        model_profile: "compact",
        model_family: "multilingual",
    },
    ModelPresetSpec {
        id: "compact-english",
        title: "compact (English only, lower WER)",
        model_profile: "compact",
        model_family: "english",
    },
    ModelPresetSpec {
        id: "fast",
        title: "fast",
        model_profile: "fast",
        model_family: "multilingual",
    },
    ModelPresetSpec {
        id: "fast-english",
        title: "fast (English only, lower WER)",
        model_profile: "fast",
        model_family: "english",
    },
];

pub const LANGUAGE_OPTIONS: &[LanguageSpec] = &[
    LanguageSpec {
        id: "en-US",
        title: "English (US)",
    },
    LanguageSpec {
        id: "en-GB",
        title: "English (UK)",
    },
    LanguageSpec {
        id: "auto",
        title: "Auto",
    },
    LanguageSpec {
        id: "es-US",
        title: "Spanish (US)",
    },
    LanguageSpec {
        id: "es-ES",
        title: "Spanish (Spain)",
    },
    LanguageSpec {
        id: "fr-FR",
        title: "French (France)",
    },
    LanguageSpec {
        id: "fr-CA",
        title: "French (Canada)",
    },
    LanguageSpec {
        id: "de-DE",
        title: "German",
    },
    LanguageSpec {
        id: "it-IT",
        title: "Italian",
    },
    LanguageSpec {
        id: "pt-BR",
        title: "Portuguese (Brazil)",
    },
    LanguageSpec {
        id: "pt-PT",
        title: "Portuguese (Portugal)",
    },
    LanguageSpec {
        id: "nl-NL",
        title: "Dutch",
    },
    LanguageSpec {
        id: "tr-TR",
        title: "Turkish",
    },
    LanguageSpec {
        id: "ru-RU",
        title: "Russian",
    },
    LanguageSpec {
        id: "ar-AR",
        title: "Arabic",
    },
    LanguageSpec {
        id: "hi-IN",
        title: "Hindi",
    },
    LanguageSpec {
        id: "ja-JP",
        title: "Japanese",
    },
    LanguageSpec {
        id: "ko-KR",
        title: "Korean",
    },
    LanguageSpec {
        id: "vi-VN",
        title: "Vietnamese",
    },
    LanguageSpec {
        id: "uk-UA",
        title: "Ukrainian",
    },
    LanguageSpec {
        id: "pl-PL",
        title: "Polish",
    },
    LanguageSpec {
        id: "sv-SE",
        title: "Swedish",
    },
    LanguageSpec {
        id: "cs-CZ",
        title: "Czech",
    },
    LanguageSpec {
        id: "nb-NO",
        title: "Norwegian Bokmal",
    },
    LanguageSpec {
        id: "da-DK",
        title: "Danish",
    },
    LanguageSpec {
        id: "bg-BG",
        title: "Bulgarian",
    },
    LanguageSpec {
        id: "fi-FI",
        title: "Finnish",
    },
    LanguageSpec {
        id: "hr-HR",
        title: "Croatian",
    },
    LanguageSpec {
        id: "sk-SK",
        title: "Slovak",
    },
    LanguageSpec {
        id: "zh-CN",
        title: "Chinese (Simplified)",
    },
    LanguageSpec {
        id: "zh-TW",
        title: "Chinese (Traditional)",
    },
    LanguageSpec {
        id: "hu-HU",
        title: "Hungarian",
    },
    LanguageSpec {
        id: "ro-RO",
        title: "Romanian",
    },
    LanguageSpec {
        id: "et-EE",
        title: "Estonian",
    },
    LanguageSpec {
        id: "el-GR",
        title: "Greek",
    },
    LanguageSpec {
        id: "lt-LT",
        title: "Lithuanian",
    },
    LanguageSpec {
        id: "lv-LV",
        title: "Latvian",
    },
    LanguageSpec {
        id: "bn-IN",
        title: "Bengali",
    },
    LanguageSpec {
        id: "id-ID",
        title: "Indonesian",
    },
    LanguageSpec {
        id: "ms-MY",
        title: "Malay",
    },
    LanguageSpec {
        id: "th-TH",
        title: "Thai",
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendSpec {
    pub id: &'static str,
    pub title: &'static str,
    pub description: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelProfileSpec {
    pub id: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub build_profile: &'static str,
    pub output_name: &'static str,
    pub prebuilt_repo: &'static str,
    pub english_output_name: &'static str,
    pub english_prebuilt_repo: &'static str,
    pub ort_format: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelPresetSpec {
    pub id: &'static str,
    pub title: &'static str,
    pub model_profile: &'static str,
    pub model_family: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LanguageSpec {
    pub id: &'static str,
    pub title: &'static str,
}

pub fn is_backend(value: &str) -> bool {
    BACKENDS.iter().any(|backend| backend.id == value)
}

pub fn is_model_profile(value: &str) -> bool {
    MODEL_PROFILES.iter().any(|profile| profile.id == value)
}

pub fn is_model_family(value: &str) -> bool {
    matches!(value, "multilingual" | "english")
}

pub fn model_preset(value: &str) -> Option<&'static ModelPresetSpec> {
    MODEL_PRESETS.iter().find(|preset| preset.id == value)
}

pub fn model_preset_id(profile: &str, family: &str) -> Option<&'static str> {
    MODEL_PRESETS
        .iter()
        .find(|preset| preset.model_profile == profile && preset.model_family == family)
        .map(|preset| preset.id)
}

pub fn is_language(value: &str) -> bool {
    LANGUAGE_OPTIONS.iter().any(|language| language.id == value)
}

pub fn is_english_language(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase();
    normalized == "en" || normalized.starts_with("en-")
}

pub fn language_available_for_model_family(family: &str, language: &str) -> bool {
    is_language(language) && (family != "english" || is_english_language(language))
}

pub const INTROSPECTION_XML: &str = r#"
<node>
  <interface name="dev.wordpipe.Service1">
    <method name="Start"/>
    <method name="Stop"/>
    <method name="Toggle"/>
    <method name="Shutdown"/>
    <method name="RegisterShellClient"/>
    <method name="GetState">
      <arg name="state" type="a{sv}" direction="out"/>
    </method>
    <method name="GetConfig">
      <arg name="config" type="a{sv}" direction="out"/>
    </method>
    <method name="ListBackends">
      <arg name="backends" type="aa{sv}" direction="out"/>
    </method>
    <method name="ListModelProfiles">
      <arg name="profiles" type="aa{sv}" direction="out"/>
    </method>
    <method name="ListInputDevices">
      <arg name="devices" type="aa{sv}" direction="out"/>
    </method>
    <method name="SetBackend">
      <arg name="backend" type="s" direction="in"/>
    </method>
    <method name="SetModelProfile">
      <arg name="profile" type="s" direction="in"/>
    </method>
    <method name="SetInputDevice">
      <arg name="selector" type="s" direction="in"/>
    </method>
    <method name="SetShortcut">
      <arg name="accelerator" type="s" direction="in"/>
    </method>
    <method name="SetInsertionOptions">
      <arg name="options" type="a{sv}" direction="in"/>
    </method>
    <method name="SetRuntimeOptions">
      <arg name="options" type="a{sv}" direction="in"/>
    </method>
    <method name="InstallModel">
      <arg name="profile" type="s" direction="in"/>
    </method>
    <signal name="StateChanged">
      <arg name="state" type="a{sv}"/>
    </signal>
    <signal name="ConfigChanged">
      <arg name="config" type="a{sv}"/>
    </signal>
    <signal name="SessionStarted">
      <arg name="session_id" type="t"/>
    </signal>
    <signal name="TextDelta">
      <arg name="session_id" type="t"/>
      <arg name="seq" type="t"/>
      <arg name="text" type="s"/>
    </signal>
    <signal name="Partial">
      <arg name="session_id" type="t"/>
      <arg name="seq" type="t"/>
      <arg name="full_text" type="s"/>
    </signal>
    <signal name="Commit">
      <arg name="session_id" type="t"/>
      <arg name="seq" type="t"/>
      <arg name="text" type="s"/>
    </signal>
    <signal name="SessionStopped">
      <arg name="session_id" type="t"/>
    </signal>
    <signal name="InstallProgress">
      <arg name="profile" type="s"/>
      <arg name="progress" type="a{sv}"/>
    </signal>
    <signal name="Metrics">
      <arg name="metrics" type="a{sv}"/>
    </signal>
    <signal name="Error">
      <arg name="message" type="s"/>
    </signal>
  </interface>
</node>
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_locales_select_the_dedicated_family() {
        assert!(is_english_language("en-US"));
        assert!(is_english_language("en-GB"));
        assert!(!is_english_language("auto"));
        assert!(!is_english_language("fr-FR"));
    }

    #[test]
    fn presets_keep_profile_and_family_separate() {
        assert_eq!(
            MODEL_PRESETS
                .iter()
                .map(|preset| preset.title)
                .collect::<Vec<_>>(),
            [
                "compact",
                "compact (English only, lower WER)",
                "fast",
                "fast (English only, lower WER)",
            ]
        );
        let preset = model_preset("compact-english").unwrap();
        assert_eq!(preset.model_profile, "compact");
        assert_eq!(preset.model_family, "english");
        assert_eq!(model_preset_id("fast", "multilingual"), Some("fast"));
        assert_eq!(model_preset_id("fast", "english"), Some("fast-english"));
    }

    #[test]
    fn english_family_only_exposes_english_languages() {
        assert!(language_available_for_model_family("english", "en-US"));
        assert!(language_available_for_model_family("english", "en-GB"));
        assert!(!language_available_for_model_family("english", "auto"));
        assert!(!language_available_for_model_family("english", "fr-FR"));
        assert!(language_available_for_model_family("multilingual", "fr-FR"));
    }
}
