"""Post-ASR heuristics and fixed-window chunking.

The interjection numbers here come from a real run: a 152-minute file
decoded without VAD produced 1,713 segments, 763 of them short
vocalisations transcribed as words (``ああ`` ×221, ``ごめん`` ×131,
``はい`` ×91) with zero hallucination markers.  These tests pin the rule that
turns that into something watchable without deleting anything a viewer might
want.
"""

from __future__ import annotations

import json

import pytest

from gensrt.heuristics import (
    BUILTIN_HEURISTICS,
    InterjectionRules,
    apply_heuristics,
    collapse_interjections,
    load_heuristics,
    write_default_heuristics,
)
from gensrt.models import SRTSegment


def _seg(i, start, end, text):
    return SRTSegment(index=i, start=start, end=end, text=text)


# Collapse-only rules for the behaviour tests: the ja always_collapse list
# with NO drop list, so "ああ" is collapsed rather than removed first.  The
# per-language and drop behaviour have their own tests further down.
_JA = BUILTIN_HEURISTICS["interjections"]["languages"]["ja"]
DEFAULT_RULES = InterjectionRules.from_dict({
    **BUILTIN_HEURISTICS["interjections"],
    "languages": {},
    "always_collapse": _JA["always_collapse"],
    "drop": [],
})


# ── Collapse behaviour ────────────────────────────────────────────────────

def test_run_of_identical_interjections_becomes_one_cue_spanning_the_run():
    segs = [_seg(1, 0.0, 0.6, "ああ"), _seg(2, 0.7, 1.4, "ああ"),
            _seg(3, 1.5, 2.0, "ああ"), _seg(4, 2.1, 2.7, "ああ")]
    out, st = collapse_interjections(segs, DEFAULT_RULES)
    assert [s.text for s in out] == ["ああ"]
    assert out[0].start == 0.0 and out[0].end == 2.7
    assert st == {"in": 4, "dropped": 0, "collapsed": 3, "out": 1}


def test_a_different_cue_breaks_the_run():
    segs = [_seg(1, 0.0, 0.5, "ああ"), _seg(2, 0.6, 3.0, "ギュッてされるの気持ちです"),
            _seg(3, 3.2, 3.8, "ああ")]
    out, _ = collapse_interjections(segs, DEFAULT_RULES)
    assert [s.text for s in out] == ["ああ", "ギュッてされるの気持ちです", "ああ"]


def test_identical_interjections_outside_the_window_stay_separate():
    segs = [_seg(1, 0.0, 0.5, "ああ"), _seg(2, 10.0, 10.5, "ああ")]   # 9.5 s gap, past the 6 s window
    out, st = collapse_interjections(segs, DEFAULT_RULES)
    assert len(out) == 2 and st["collapsed"] == 0


def test_always_collapse_list_covers_strings_longer_than_max_chars():
    """気持ちいい is 5 chars — beyond max_chars — but listed."""
    segs = [_seg(1, 0.0, 1.5, "気持ちいい"), _seg(2, 1.6, 2.5, "気持ちいい")]
    out, st = collapse_interjections(segs, DEFAULT_RULES)
    assert len(out) == 1 and st["collapsed"] == 1


def test_long_identical_lines_are_left_alone():
    """That is a repetition-loop signal, and a different rule's job."""
    line = "前回の施術の効果があったみたいですね"
    segs = [_seg(1, 0.0, 2.0, line), _seg(2, 2.1, 4.0, line)]
    out, st = collapse_interjections(segs, DEFAULT_RULES)
    assert len(out) == 2 and st["collapsed"] == 0


def test_drop_list_removes_the_string_outright():
    rules = InterjectionRules.from_dict({**BUILTIN_HEURISTICS["interjections"],
                                         "drop": ["ああ"]})
    segs = [_seg(1, 0.0, 0.5, "ああ"), _seg(2, 0.6, 2.0, "はい"), _seg(3, 2.1, 2.5, "ああ")]
    out, st = collapse_interjections(segs, rules)
    assert [s.text for s in out] == ["はい"]
    assert st["dropped"] == 2


def test_output_is_reindexed_from_one():
    segs = [_seg(5, 0.0, 0.5, "ああ"), _seg(9, 0.6, 1.0, "ああ"), _seg(12, 2.0, 4.0, "何ですか?")]
    out, _ = collapse_interjections(segs, DEFAULT_RULES)
    assert [s.index for s in out] == [1, 2]


def test_confidence_fields_survive_from_the_first_cue():
    a = SRTSegment(index=1, start=0.0, end=0.5, text="ああ", avg_logprob=-0.3)
    b = SRTSegment(index=2, start=0.6, end=1.0, text="ああ", avg_logprob=-0.9)
    out, _ = collapse_interjections([a, b], DEFAULT_RULES)
    assert out[0].avg_logprob == -0.3


def test_disabled_rules_pass_everything_through():
    rules = InterjectionRules.from_dict({"enabled": False})
    segs = [_seg(1, 0.0, 0.5, "ああ"), _seg(2, 0.6, 1.0, "ああ")]
    out, st = collapse_interjections(segs, rules)
    assert len(out) == 2 and st["collapsed"] == 0


def test_empty_input():
    out, st = collapse_interjections([], DEFAULT_RULES)
    assert out == [] and st["out"] == 0


# ── Rules file ────────────────────────────────────────────────────────────

def test_missing_file_uses_builtin_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", [str(tmp_path / "gensrt")])
    h = load_heuristics()
    assert h.source_path is None
    assert h.interjections.max_chars == 3


def test_user_file_overrides_defaults(tmp_path):
    p = tmp_path / "gensrt-heuristics.json"
    p.write_text(json.dumps({"interjections": {"max_chars": 2, "drop": ["うっ"]}}),
                 encoding="utf-8")
    h = load_heuristics(p)
    assert h.source_path == p
    assert h.interjections.max_chars == 2
    assert "うっ" in h.interjections.drop
    # keys the user did not set keep their defaults
    assert h.interjections.window_s == 6.0


def test_malformed_file_falls_back_and_does_not_raise(tmp_path, caplog):
    import logging

    p = tmp_path / "gensrt-heuristics.json"
    p.write_text("{ this is not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="gensrt.heuristics"):
        h = load_heuristics(p)
    assert h.source_path is None
    assert h.interjections.max_chars == 3
    assert any("using built-in heuristics" in r.message for r in caplog.records)


def test_write_default_then_load_round_trips(tmp_path):
    p = write_default_heuristics(tmp_path / "gensrt-heuristics.json")
    assert p.is_file()
    h = load_heuristics(p)
    assert h.for_language("ja").always_collapse == DEFAULT_RULES.always_collapse


def test_apply_heuristics_logs_a_summary(caplog):
    import logging

    segs = [_seg(1, 0.0, 0.5, "ああ"), _seg(2, 0.6, 1.0, "ああ")]
    with caplog.at_level(logging.INFO, logger="gensrt.heuristics"):
        out = apply_heuristics(segs, load_heuristics(None) if False else None)
    assert len(out) == 1
    assert any("collapsed" in r.message for r in caplog.records)


# ── Fixed-window chunk mode ───────────────────────────────────────────────

def test_fixed_mode_yields_one_region_for_the_whole_file():
    import numpy as np

    from gensrt.asr.monolingual_whisper import MonolingualWhisperEngine
    from gensrt.models import TranscriptionConfig

    audio = np.zeros(16000 * 30, dtype=np.float32)
    cfg = TranscriptionConfig(chunk_mode="fixed")
    regions = MonolingualWhisperEngine._outer_vad(audio, 16000, cfg)
    assert regions == [(0.0, 30.0)]


def test_no_vad_remains_a_synonym_for_fixed():
    import numpy as np

    from gensrt.asr.monolingual_whisper import MonolingualWhisperEngine
    from gensrt.models import TranscriptionConfig

    audio = np.zeros(16000 * 10, dtype=np.float32)
    cfg = TranscriptionConfig(vad_enabled=False)
    assert MonolingualWhisperEngine._outer_vad(audio, 16000, cfg) == [(0.0, 10.0)]


def test_chunk_mode_is_saveable_and_validated():
    from gensrt.server import _validate_config_patch

    assert not _validate_config_patch({"chunk_mode": "fixed"})[1]
    assert "chunk_mode" in _validate_config_patch({"chunk_mode": "sometimes"})[1]


def test_fixed_mode_constants_are_larger_than_vad_defaults():
    from gensrt.asr._silence_chunking import (
        DEFAULT_MAX_CHUNK_S, DEFAULT_MIN_CHUNK_S, FIXED_MAX_CHUNK_S, FIXED_MIN_CHUNK_S,
    )

    assert FIXED_MIN_CHUNK_S > DEFAULT_MIN_CHUNK_S
    assert FIXED_MAX_CHUNK_S > DEFAULT_MAX_CHUNK_S


def test_window_default_matches_measured_chunk_spacing():
    """Identical interjections from adjacent 5-8 s chunks are 4-6 s apart
    (measured p50 4.0 s, p90 5.7 s on a real file).  A 3 s window caught 35
    of 137; 6 s catches 136."""
    assert DEFAULT_RULES.window_s == 6.0


# ── Per-language rules ────────────────────────────────────────────────────
#
# The rules match SOURCE-language text.  A user who cannot read that script
# cannot write it, so the lists ship per language and are selected by the
# detected language; the user toggles, never types.

def test_ja_rules_include_the_measured_vocalisations():
    h = load_heuristics(None)
    ja = h.for_language("ja")
    assert "ああ" in ja.drop and "あっ" in ja.drop
    assert ja.language == "ja"


def test_words_are_never_on_the_drop_list_by_default():
    """はい / うん are real words: not on any removal list.  ごめん / すごい /
    ごちそう are real words too, but the model produces them over silence
    (confirmed by watching) — they sit on the separate hallucination list,
    never on ``drop``, so a user can move them back without losing the
    distinction."""
    ja = load_heuristics(None).for_language("ja")
    for word in ("ごめん", "はい", "うん", "すごい", "ごちそう"):
        assert word not in ja.drop
    for word in ("ごめん", "すごい", "ごちそう"):
        assert word in ja.hallucination
    assert "気持ちいい" in ja.always_collapse


def test_unmeasured_languages_ship_empty_lists():
    h = load_heuristics(None)
    for code in ("ko", "ml"):
        r = h.for_language(code)
        assert r.drop == frozenset() and r.always_collapse == frozenset()
        assert r.language == code


def test_unknown_language_gets_only_the_global_lists(tmp_path):
    p = tmp_path / "gensrt-heuristics.json"
    p.write_text(json.dumps({"interjections": {
        "drop": ["GLOBAL"],
        "languages": {"ja": {"drop": ["ああ"]}},
    }}), encoding="utf-8")
    r = load_heuristics(p).for_language("xx")
    assert r.drop == {"GLOBAL"} and r.language is None


def test_language_lists_stack_on_global_lists(tmp_path):
    p = tmp_path / "gensrt-heuristics.json"
    p.write_text(json.dumps({"interjections": {
        "drop": ["GLOBAL"],
        "languages": {"ja": {"drop": ["ああ"]}},
    }}), encoding="utf-8")
    r = load_heuristics(p).for_language("ja")
    assert r.drop == {"GLOBAL", "ああ"}


def test_a_language_section_can_be_switched_off(tmp_path):
    p = tmp_path / "gensrt-heuristics.json"
    p.write_text(json.dumps({"interjections": {
        "languages": {"ja": {"enabled": False, "drop": ["ああ"]}},
    }}), encoding="utf-8")
    r = load_heuristics(p).for_language("ja")
    assert "ああ" not in r.drop and r.language is None


def test_first_release_flat_file_still_works(tmp_path):
    """No 'languages' key at all — the shape the first drop shipped."""
    p = tmp_path / "gensrt-heuristics.json"
    p.write_text(json.dumps({"interjections": {"drop": ["ああ"], "always_collapse": ["ごめん"]}}),
                 encoding="utf-8")
    r = load_heuristics(p).for_language("ja")
    assert "ああ" in r.drop and "ごめん" in r.always_collapse


def test_region_codes_fall_back_to_the_base_language():
    assert load_heuristics(None).for_language("ja-JP").language == "ja"


def test_apply_heuristics_uses_the_detected_language():
    segs = [_seg(1, 0.0, 0.5, "ああ"), _seg(2, 1.0, 2.0, "何ですか?")]
    ja = apply_heuristics(segs, load_heuristics(None), "ja")
    ko = apply_heuristics(segs, load_heuristics(None), "ko")
    assert [s.text for s in ja] == ["何ですか?"]         # dropped under ja rules
    assert [s.text for s in ko] == ["ああ", "何ですか?"]   # ko has no drop list


# ── Run report ────────────────────────────────────────────────────────────

def _report_fixture():
    from gensrt.heuristics import build_report

    raw = [_seg(1, 0.0, 0.5, "ああ"), _seg(2, 1.0, 1.5, "ああ"),
           _seg(3, 2.0, 3.0, "はい"), _seg(4, 4.0, 6.0, "何ですか?")]
    rules = load_heuristics(None).for_language("ja")
    final = apply_heuristics(raw, load_heuristics(None), "ja")
    # pretend translation happened to what survived
    translated = [type(s)(index=s.index, start=s.start, end=s.end,
                          text={"はい": "Yes.", "何ですか?": "What is it?"}.get(s.text, s.text))
                  for s in final]
    return raw, translated, rules


def test_report_rows_carry_count_share_and_action():
    from gensrt.heuristics import build_report

    raw, final, rules = _report_fixture()
    rep = build_report(raw, final, rules, translate=lambda ts: ["Oh, yeah."] * len(ts))
    by = {r.source: r for r in rep.rows}
    assert by["ああ"].count == 2 and by["ああ"].dropped
    assert abs(by["ああ"].share - 0.5) < 1e-9
    assert by["はい"].translation == "Yes." and not by["はい"].dropped


def test_dropped_strings_get_their_meaning_from_the_translate_hook():
    """They never reached the translator, so the report asks for them."""
    from gensrt.heuristics import build_report

    raw, final, rules = _report_fixture()
    asked = []
    rep = build_report(raw, final, rules,
                       translate=lambda ts: (asked.extend(ts), ["Oh, yeah."] * len(ts))[1])
    assert "ああ" in asked
    assert {r.source: r for r in rep.rows}["ああ"].translation == "Oh, yeah."


def test_report_survives_a_failing_translate_hook():
    from gensrt.heuristics import build_report

    raw, final, rules = _report_fixture()

    def _boom(_ts):
        raise RuntimeError("no engine")

    rep = build_report(raw, final, rules, translate=_boom)
    assert rep.rows and {r.source for r in rep.rows} >= {"ああ", "はい"}


def test_report_as_text_and_as_dict_render():
    from gensrt.heuristics import build_report

    raw, final, rules = _report_fixture()
    rep = build_report(raw, final, rules)
    text = rep.as_text()
    assert "DROPPED" in text and "ああ" in text
    d = rep.as_dict()
    assert d["raw_cues"] == 4 and d["language"] == "ja" and d["rows"]


def test_report_ignores_long_lines():
    from gensrt.heuristics import build_report

    raw = [_seg(1, 0.0, 2.0, "前回の施術の効果があったみたいですね")]
    rep = build_report(raw, raw, load_heuristics(None).for_language("ja"))
    assert rep.rows == []


def test_transcription_result_carries_the_report_field():
    from gensrt.models import TranscriptionResult

    assert "heuristics_report" in TranscriptionResult.__dataclass_fields__


# ── Rule: hallucination list ──────────────────────────────────────────────

def test_hallucination_strings_are_removed_and_reported_separately():
    from gensrt.heuristics import run_heuristics

    segs = [_seg(1, 0.0, 1.0, "ごめん"), _seg(2, 5.0, 6.0, "何ですか?"),
            _seg(3, 9.0, 10.0, "ああ"), _seg(4, 12.0, 13.0, "ごちそう")]
    out, st = run_heuristics(segs, load_heuristics(None), "ja")
    assert [s.text for s in out] == ["何ですか?"]
    assert st.dropped["hallucination"] == {"ごめん": 1, "ごちそう": 1}
    assert st.dropped["drop"] == {"ああ": 1}
    assert st.reason_for("ごめん") == "hallucination"
    assert st.dropped_total() == 3 and st.cues_out == 1


def test_hallucination_list_is_per_language(tmp_path):
    import json
    from gensrt.heuristics import run_heuristics

    p = tmp_path / "h.json"
    p.write_text(json.dumps({"interjections": {"languages": {
        "ko": {"hallucination": ["감사합니다"]}}}}), encoding="utf-8")
    segs = [_seg(1, 0.0, 1.0, "감사합니다")]
    assert run_heuristics(segs, load_heuristics(p), "ko")[0] == []
    assert len(run_heuristics(segs, load_heuristics(p), "ja")[0]) == 1


# ── Rule: density ─────────────────────────────────────────────────────────

def _density_h(min_count=4, window_s=30.0, hallucination=None):
    from gensrt.heuristics import Heuristics
    ja = {**BUILTIN_HEURISTICS["interjections"]["languages"]["ja"],
          "hallucination": hallucination or [], "drop": []}
    return Heuristics(raw={
        **BUILTIN_HEURISTICS,
        "interjections": {**BUILTIN_HEURISTICS["interjections"], "languages": {"ja": ja}},
        "density": {"enabled": True, "min_count": min_count, "window_s": window_s},
    })


def test_dense_run_of_a_short_cue_is_removed_whole():
    from gensrt.heuristics import run_heuristics

    segs = [_seg(i, 5.0 * i, 5.0 * i + 1, "ごめん") for i in range(1, 6)]   # 5 in 20 s
    out, st = run_heuristics(segs, _density_h(), "ja")
    assert out == []
    assert st.dropped["density"] == {"ごめん": 5}


def test_isolated_occurrences_survive_density():
    from gensrt.heuristics import run_heuristics

    segs = [_seg(i, 60.0 * i, 60.0 * i + 1, "ごめん") for i in range(1, 6)]   # one a minute
    out, st = run_heuristics(segs, _density_h(), "ja")
    assert len(out) == 5 and "density" not in st.dropped


def test_alternating_interjections_still_count_as_a_run():
    """ごめん ああ ごめん ああ ごめん ああ ごめん: other interjections between the
    repeats neither block nor count."""
    from gensrt.heuristics import run_heuristics

    segs = []
    for i in range(8):
        segs.append(_seg(i + 1, 3.0 * i, 3.0 * i + 1, "ごめん" if i % 2 == 0 else "ああ"))
    out, st = run_heuristics(segs, _density_h(), "ja")
    assert st.dropped["density"] == {"ごめん": 4, "ああ": 4}
    assert out == []


def test_dialogue_between_repeats_breaks_the_run():
    from gensrt.heuristics import run_heuristics

    segs = [_seg(1, 0, 1, "ごめん"), _seg(2, 3, 4, "ごめん"),
            _seg(3, 6, 9, "今日はありがとうございました"),
            _seg(4, 10, 11, "ごめん"), _seg(5, 13, 14, "ごめん")]
    out, st = run_heuristics(segs, _density_h(min_count=4), "ja")
    assert "density" not in st.dropped
    assert len(out) == 3           # two runs of two collapse to one cue each


def test_density_runs_before_collapse_so_counts_are_real():
    """Collapse would merge five identical cues into one and hide the run."""
    from gensrt.heuristics import run_heuristics

    segs = [_seg(i, 2.0 * i, 2.0 * i + 1, "はい") for i in range(1, 6)]
    out, st = run_heuristics(segs, _density_h(), "ja")
    assert out == [] and st.collapsed == 0


def test_density_can_be_disabled(tmp_path):
    import json
    from gensrt.heuristics import run_heuristics

    p = tmp_path / "h.json"
    p.write_text(json.dumps({"density": {"enabled": False},
                             "interjections": {"languages": {"ja": {"hallucination": []}}}}),
                 encoding="utf-8")
    segs = [_seg(i, 5.0 * i, 5.0 * i + 1, "ごめん") for i in range(1, 6)]
    out, st = run_heuristics(segs, load_heuristics(p), "ja")
    assert "density" not in st.dropped and len(out) == 1   # collapsed instead


# ── Rule: subject stripping ───────────────────────────────────────────────

def _subj():
    return load_heuristics(None).subject_for("ja", "en")


def test_leading_subject_pronoun_is_stripped_and_recapitalised():
    from gensrt.heuristics import strip_leading_subject

    r = _subj()
    assert strip_leading_subject("I'm sorry.", r) == "Sorry."
    assert strip_leading_subject("It's a huge one.", r) == "A huge one."
    assert strip_leading_subject("I don't know.", r) == "Don't know."
    assert strip_leading_subject("I love you", r) == "Love you"
    assert strip_leading_subject('"I love you."', r) == '"Love you."'
    assert strip_leading_subject("I’m fine", r) == "Fine"            # curly


def test_html_entity_clitics_are_handled_whole():
    """Some engine output arrives as 'I &apos;m going' — the clitic must go
    with the pronoun, never leave '&apos;m going' behind."""
    from gensrt.heuristics import strip_leading_subject

    assert strip_leading_subject("I &apos;m going to focus.", _subj()) == "Going to focus."
    assert strip_leading_subject("It &apos;s great .", _subj()) == "Great ."


def test_subject_is_kept_when_the_remainder_needs_it():
    from gensrt.heuristics import strip_leading_subject

    r = _subj()
    for line in ("I.", "I", "You can rest assured.", "It is characterized by X",
                 "You have chosen", "I did it", "I'll do it", "I've never had",
                 "Are you okay?", "Sorry.",
                 "I'm Lena Kotama, 22 years old.", "I'm Japanese.", "It's 3 o'clock.",
                 "You know, like this.", "I know a little bit.", "You see, it's good.",
                 "I mean it."):
        assert strip_leading_subject(line, r) is None, line
    assert strip_leading_subject("I can't.", r) == "Can't."   # negative is fine


def test_strip_subjects_skips_lines_whose_source_names_a_subject():
    from gensrt.heuristics import strip_subjects

    src = ["ごめん", "私は何も知りません", "気持ちいい", "ごめん"]
    segs = [_seg(1, 0, 1, "I'm sorry."), _seg(2, 2, 3, "I don't know anything."),
            _seg(3, 4, 5, "It feels good."), _seg(4, 6, 7, "ごめん")]   # 4: untranslated
    out, n, examples = strip_subjects(src, segs, _subj())
    assert [s.text for s in out] == ["Sorry.", "I don't know anything.", "Feels good.", "ごめん"]
    assert n == 2 and examples[0] == ("I'm sorry.", "Sorry.")


def test_subject_rule_is_english_target_and_pro_drop_source_only():
    h = load_heuristics(None)
    assert h.subject_for("ja", "en").enabled
    assert h.subject_for("ja-JP", "English").language == "ja"
    assert h.subject_for("ml", "en").enabled and h.subject_for("ko", "en").enabled
    assert not h.subject_for("ja", "de").enabled
    assert not h.subject_for("en", "en").enabled
    assert not h.subject_for("fr", "en").enabled


def test_subject_rule_can_be_disabled_per_language(tmp_path):
    import json

    p = tmp_path / "h.json"
    p.write_text(json.dumps({"subject": {"languages": {"ja": {"enabled": False}}}}),
                 encoding="utf-8")
    assert not load_heuristics(p).subject_for("ja", "en").enabled


def test_strip_subjects_with_misaligned_inputs_is_a_no_op():
    from gensrt.heuristics import strip_subjects

    segs = [_seg(1, 0, 1, "I'm sorry.")]
    out, n, _ = strip_subjects(["a", "b"], segs, _subj())
    assert n == 0 and out[0].text == "I'm sorry."


# ── Report carries the reasons ────────────────────────────────────────────

def test_report_shows_reason_and_partial_counts_from_stats():
    from gensrt.heuristics import HeuristicsStats, build_report, run_heuristics

    raw = [_seg(1, 0, 1, "ああ"), _seg(2, 2, 3, "ごめん"), _seg(3, 4, 5, "はい"),
           _seg(4, 6, 7, "はい"), _seg(5, 60, 61, "はい")]
    h = load_heuristics(None)
    final, st = run_heuristics(raw, h, "ja")
    st.subject_stripped, st.subject_examples = 2, [("I'm sorry.", "Sorry.")]
    rep = build_report(raw, final, h.for_language("ja"), stats=st)
    by = {r.source: r for r in rep.rows}
    assert by["ああ"].reason == "drop" and by["ああ"].dropped == 1
    assert by["ごめん"].reason == "hallucination"
    assert by["はい"].reason is None and by["はい"].collapsed == 1
    text = rep.as_text()
    assert "hallucination" in text and "drop list" in text
    assert "Subject pronouns stripped" in text and "'Sorry.'" in text
    assert rep.as_dict()["subject_stripped"] == 2


def test_report_marks_partial_density_removal():
    from gensrt.heuristics import build_report, run_heuristics

    raw = [_seg(i, 2.0 * i, 2.0 * i + 1, "はい") for i in range(1, 6)]
    raw += [_seg(9, 300, 302, "何ですか?"), _seg(10, 400, 401, "はい")]
    h = _density_h()
    final, st = run_heuristics(raw, h, "ja")
    rep = build_report(raw, final, h.for_language("ja"), stats=st)
    row = {r.source: r for r in rep.rows}["はい"]
    assert row.reason == "density" and row.dropped == 5 and row.count == 6
    assert "DROPPED 5/6 (dense run)" in rep.as_text()


def test_shipped_json_matches_the_builtin_defaults():
    import json
    from pathlib import Path

    shipped = json.loads((Path(__file__).resolve().parents[1] / "gensrt-heuristics.json")
                         .read_text(encoding="utf-8"))
    assert shipped == BUILTIN_HEURISTICS


# ── Collapse span cap and punctuation-insensitive list matching ───────────

def test_collapsed_run_is_shown_for_at_most_max_span_s():
    """Twenty うん over 10 s used to become one 10-second "Yeah." cue."""
    segs = [_seg(i, 0.5 * i, 0.5 * i + 0.4, "うん") for i in range(20)]
    out, st = collapse_interjections(segs, DEFAULT_RULES)
    assert len(out) == 1 and st["collapsed"] == 19
    assert out[0].start == 0.0 and abs(out[0].end - 3.0) < 1e-9


def test_span_cap_does_not_break_the_run():
    """Cues after the cap still merge (the run is tracked by its true end),
    so a long run yields ONE short cue, not a cue every window_s."""
    segs = [_seg(i, 1.0 * i, 1.0 * i + 0.5, "うん") for i in range(12)]   # 12 s run
    out, _ = collapse_interjections(segs, DEFAULT_RULES)
    assert len(out) == 1


def test_span_cap_zero_means_span_the_run(tmp_path):
    import json
    from gensrt.heuristics import InterjectionRules

    rules = InterjectionRules.from_dict({"max_span_s": 0, "languages": {}})
    segs = [_seg(i, 1.0 * i, 1.0 * i + 0.5, "うん") for i in range(5)]
    out, _ = collapse_interjections(segs, rules)
    assert abs(out[0].end - 4.5) < 1e-9


def test_list_matching_ignores_trailing_punctuation():
    from gensrt.heuristics import run_heuristics

    segs = [_seg(1, 0, 1, "ごめん。"), _seg(2, 2, 3, "ああ!"),
            _seg(3, 5, 7, "ご視聴ありがとうございました。"), _seg(4, 9, 10, "何ですか?")]
    out, st = run_heuristics(segs, load_heuristics(None), "ja")
    assert [s.text for s in out] == ["何ですか?"]
    assert st.dropped["hallucination"] == {"ごめん。": 1, "ご視聴ありがとうございました。": 1}


# ── Korean run findings (eleven broadcast files) ──────────────────────────

def test_korean_max_chars_is_two_so_three_syllable_words_are_not_interjections():
    """예산이 ("the budget", 3 hangul blocks) was removed as a dense run of an
    interjection.  A per-language max_chars override fixes the class."""
    from gensrt.heuristics import _is_interjection

    ko = load_heuristics(None).for_language("ko")
    ja = load_heuristics(None).for_language("ja")
    assert ko.max_chars == 2 and ja.max_chars == 3
    assert not _is_interjection("예산이", ko)
    assert _is_interjection("어?", ko) and _is_interjection("진짜", ko)


def test_dense_run_of_a_korean_content_word_survives():
    from gensrt.heuristics import run_heuristics

    segs = [_seg(i, 5.0 * i, 5.0 * i + 1, "예산이") for i in range(1, 6)]
    out, st = run_heuristics(segs, load_heuristics(None), "ko")
    assert len(out) == 5 and "density" not in st.dropped


def test_replacement_character_cues_are_junk():
    """The decoder emitted '1\\ufffdgn' six times, zero duration, same logprob."""
    from gensrt.heuristics import run_heuristics

    segs = [_seg(1, 0, 0.5, "1\ufffdgn"), _seg(2, 0.5, 0.5, "2\ufffdgn"),
            _seg(3, 1, 3, "고마워")]
    out, st = run_heuristics(segs, load_heuristics(None), "ko")
    assert [s.text for s in out] == ["고마워"]
    assert st.dropped["junk"] == {"1\ufffdgn": 1, "2\ufffdgn": 1}
    assert st.reason_for("1\ufffdgn") == "junk"


def test_third_person_subjects_are_kept_by_default():
    """"He's a legend." → "A legend." reads worse than a possibly-wrong
    pronoun; only I / you / we / it are stripped unless the user adds more."""
    from gensrt.heuristics import strip_leading_subject

    r = load_heuristics(None).subject_for("ko", "en")
    assert strip_leading_subject("He's a legend.", r) is None
    assert strip_leading_subject("She's red.", r) is None
    assert strip_leading_subject("They went home.", r) is None
    assert strip_leading_subject("I'm gonna go.", r) == "Gonna go."
    assert strip_leading_subject("It's great.", r) == "Great."


def test_report_writes_txt_and_json(tmp_path):
    from gensrt.heuristics import build_report, run_heuristics

    raw = [_seg(1, 0, 1, "ああ"), _seg(2, 2, 3, "はい")]
    h = load_heuristics(None)
    final, st = run_heuristics(raw, h, "ja")
    rep = build_report(raw, final, h.for_language("ja"), stats=st)
    txt, js = rep.write(tmp_path / "out", "movie")
    assert txt.name == "movie.heuristics.txt" and js.name == "movie.heuristics.json"
    assert "ああ" in txt.read_text(encoding="utf-8")
    import json
    assert json.loads(js.read_text(encoding="utf-8"))["raw_cues"] == 2


def test_cli_report_dir_maps_to_the_config_key():
    from gensrt.cli import _build_parser
    from gensrt.server import _validate_config_patch

    args = _build_parser().parse_args(["--input", "v.mkv", "--heuristics-report-dir", "D:/r"])
    assert args.heuristics_report_out == "D:/r"
    assert _validate_config_patch({"heuristics_report_dir": "D:/r"})[1] == {}


# ── Sign-off hallucinations: substring lists and the repeated-lines table ─

def test_hallucination_contains_matches_variants():
    from gensrt.heuristics import run_heuristics

    segs = [_seg(1, 0, 2, "다음 영상에서 만나요"), _seg(2, 5, 7, "다음 영상에서 만나요!"),
            _seg(3, 9, 11, "시청해주셔서 감사합니다"), _seg(4, 20, 22, "오늘 뭐 먹었어?")]
    out, st = run_heuristics(segs, load_heuristics(None), "ko")
    assert [s.text for s in out] == ["오늘 뭐 먹었어?"]
    assert st.dropped_total("hallucination") == 3


def test_japanese_signoff_variants_are_caught_by_substring():
    from gensrt.heuristics import run_heuristics

    segs = [_seg(1, 0, 2, "ご視聴ありがとうございました！"), _seg(2, 5, 7, "ご視聴ありがとう")]
    out, _ = run_heuristics(segs, load_heuristics(None), "ja")
    assert out == []


def test_report_lists_repeated_long_lines_the_short_table_cannot_see():
    """37 × "See you in the next video." was invisible to a table of strings
    ≤ 4 characters.  Any full line repeated ≥ 3 times is now listed, with
    its translation and what the rules did."""
    from gensrt.heuristics import build_report, run_heuristics

    h = load_heuristics(None)
    raw = [_seg(i, 30.0 * i, 30.0 * i + 2, "다음 영상에서 만나요") for i in range(1, 5)]
    raw += [_seg(9, 200, 203, "정말 재미있는 하루였어요") for _ in range(3)]
    raw += [_seg(20, 400, 402, "안녕")]
    final, st = run_heuristics(raw, h, "ko")
    translated = [type(s)(index=s.index, start=s.start, end=s.end, text="It was a fun day.")
                  for s in final if s.text == "정말 재미있는 하루였어요"]
    rep = build_report(raw, final[:0] + translated, h.for_language("ko"), stats=st,
                       translate=lambda ts: ["See you in the next video."] * len(ts))
    by = {r.source: r for r in rep.repeats}
    assert by["다음 영상에서 만나요"].count == 4
    assert by["다음 영상에서 만나요"].reason == "hallucination" and by["다음 영상에서 만나요"].dropped == 4
    assert by["다음 영상에서 만나요"].translation == "See you in the next video."
    assert by["정말 재미있는 하루였어요"].reason is None and by["정말 재미있는 하루였어요"].translation == "It was a fun day."
    assert "안녕" not in by                          # short strings stay in their own table
    text = rep.as_text()
    assert "Repeated lines, any length" in text and "4x  다음 영상에서 만나요" in text
    assert rep.as_dict()["repeats"][0]["count"] == 4


def test_repeats_are_reported_even_when_there_are_no_short_strings():
    from gensrt.heuristics import build_report, run_heuristics

    h = load_heuristics(None)
    raw = [_seg(i, 30.0 * i, 30.0 * i + 2, "정말 재미있는 하루였어요") for i in range(1, 4)]
    final, st = run_heuristics(raw, h, "ko")
    rep = build_report(raw, final, h.for_language("ko"), stats=st)
    assert rep.rows == [] and len(rep.repeats) == 1
