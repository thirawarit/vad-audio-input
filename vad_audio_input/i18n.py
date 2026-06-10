"""Multi-language message catalog for English, Thai and Chinese."""

from typing import (Dict, Final, Literal)

Language = Literal["en", "th", "zh"]

SUPPORTED_LANGUAGES: Final[tuple[Language, ...]] = ("en", "th", "zh")

# Each message key maps to a per-language template. Templates use str.format fields.
_MESSAGES: Final[Dict[str, Dict[Language, str]]] = {
    "scanning_inputs": {
        "en": "Scanning {count} input path(s).",
        "th": "กำลังสแกนพาธอินพุต {count} รายการ",
        "zh": "正在扫描 {count} 个输入路径。",
    },
    "no_inputs": {
        "en": "No supported audio files found.",
        "th": "ไม่พบไฟล์เสียงที่รองรับ",
        "zh": "未找到受支持的音频文件。",
    },
    "processing_file": {
        "en": "Processing '{path}'.",
        "th": "กำลังประมวลผล '{path}'",
        "zh": "正在处理 '{path}'。",
    },
    "loaded_audio": {
        "en": "Loaded '{path}': {duration_ms:.1f} ms @ {sample_rate} Hz, {channels} ch.",
        "th": "โหลด '{path}' แล้ว: {duration_ms:.1f} มิลลิวินาที @ {sample_rate} Hz, {channels} ช่อง",
        "zh": "已加载 '{path}'：{duration_ms:.1f} 毫秒 @ {sample_rate} Hz，{channels} 声道。",
    },
    "detected_spans": {
        "en": "Detected {count} speech span(s).",
        "th": "ตรวจพบช่วงเสียงพูด {count} ช่วง",
        "zh": "检测到 {count} 个语音片段。",
    },
    "wrote_segments": {
        "en": "Wrote {count} segment(s) to '{path}'.",
        "th": "เขียน {count} เซกเมนต์ไปยัง '{path}'",
        "zh": "已将 {count} 个片段写入 '{path}'。",
    },
    "skip_file_error": {
        "en": "Skipping '{path}': {error}",
        "th": "ข้าม '{path}': {error}",
        "zh": "跳过 '{path}'：{error}",
    },
    "done": {
        "en": "Done. {ok} file(s) succeeded, {failed} failed.",
        "th": "เสร็จสิ้น สำเร็จ {ok} ไฟล์ ล้มเหลว {failed} ไฟล์",
        "zh": "完成。{ok} 个文件成功，{failed} 个失败。",
    },
}


def translate(key: str, lang: Language, **kwargs: object) -> str:
    """Return the localized, formatted message for ``key`` in ``lang``.

    Falls back to English, then to the raw key, if a translation is missing, and
    returns the unformatted template if formatting fields are missing.

    Args:
        key: Message catalog key.
        lang: Target language (``"en"``, ``"th"`` or ``"zh"``).
        **kwargs: Values substituted into the message's ``str.format`` fields.

    Returns:
        The localized, formatted message.
    """
    by_lang: Dict[Language, str] = _MESSAGES.get(key, {})
    template: str = by_lang.get(lang) or by_lang.get("en") or key
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        return template
