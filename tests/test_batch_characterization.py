"""Characterization: `speakerscribe.batch` reproduces notebook v5 where it must.

The functions under test are extracted from the frozen v5 notebook
(``notebooks/legacy/speakerscribe_notebook_v5.ipynb``) and executed with the
same inputs as their ports. Deliberate behavior changes are asserted as
such, so any future drift is a visible, reviewed decision.
"""

from __future__ import annotations

import ast
import bisect
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import pytest

from speakerscribe.batch import renderers as r
from speakerscribe.batch.fsio import canonical_hash
from speakerscribe.batch.identity import source_id_for
from speakerscribe.batch.journal import JournalIndex
from speakerscribe.batch.layout import sanitize_name
from speakerscribe.batch.speakers import apply_renames, repair_orphans

LEGACY = (
    Path(__file__).resolve().parents[1] / "notebooks" / "legacy" / "speakerscribe_notebook_v5.ipynb"
)


def load_v5(names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    """Exec the named top-level defs of the v5 notebook in `namespace`."""
    notebook = json.loads(LEGACY.read_text(encoding="utf-8"))
    found: dict[str, ast.stmt] = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
        )
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name in names:
                found[node.name] = node
    missing = names - found.keys()
    assert not missing, f"v5 no define {missing}"
    module = ast.Module(body=list(found.values()), type_ignores=[])
    exec(compile(module, str(LEGACY), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def v5() -> dict[str, Any]:
    if not LEGACY.is_file():
        pytest.skip("notebook v5 de referencia no disponible")
    ns: dict[str, Any] = {
        "Any": Any,
        "bisect": bisect,
        "hashlib": hashlib,
        "json": json,
        "Path": Path,
        "_HABLANTES_NULOS": {None, "", "(no diarization)", "SPEAKER_NO_OVERLAP", "None"},
        "TIMESTAMP_MILISEGUNDOS": False,
        "INCLUIR_ENCABEZADO": False,
        "ACEPTAR_AUDIO_SIN_VOZ": True,
        "MAX_FRACCION_SIN_HABLANTE": 0.35,
        "MODELO": "large-v3",
        "BEAM_SIZE": 5,
        "BATCH_SIZE": 8,
    }
    return load_v5(
        {
            "canonical_hash",
            "format_ts",
            "reparar_hablantes",
            "aplicar_renombres",
            "render_transcript",
            "_ts_srt",
            "render_srt",
            "_turnos",
            "render_full_llm",
            "render_plano",
            "render_splits",
            "build_indices",
            "_diar_profile",
        },
        ns,
    )


def random_segments(seed: int, n: int = 60) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    t = 0.0
    labels = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", None, "SPEAKER_NO_OVERLAP", ""]
    out = []
    for _ in range(n):
        start = round(t + rng.uniform(0, 1.5), 3)
        end = round(start + rng.uniform(0.2, 6.0), 3)
        words = " ".join(
            rng.choice(["hola", "acta", "plan", "IA", "sí", "  ", "Bogotá"])
            for _ in range(rng.randint(0, 9))
        )
        out.append({"start": start, "end": end, "text": words, "speaker": rng.choice(labels)})
        t = end
    return out


def ctx(segments, metadata=None, *, ms=False, header=False, tolerance=0.05) -> r.RenderContext:
    return r.RenderContext(
        segments=segments,
        metadata=metadata or {"diarization_enabled": True, "duration_seconds": 300.0},
        source_rel="a/b.wav",
        options=r.RenderOptions(
            include_header=header, timestamp_ms=ms, monotonic_tolerance_s=tolerance
        ),
    )


class TestIdentityFormulas:
    def test_canonical_hash_is_unchanged(self, v5):
        value = {"root": "x", "relative": "a/ñ *b.wav", "n": [1, 2.5, None]}
        assert canonical_hash(value) == v5["canonical_hash"](value)

    def test_source_id_formula_matches_v5(self, v5):
        root = Path(
            "/content/drive/MyDrive/ProColombia/1B. Resultados/Transcripcion-Diarizacion/data"
        )
        rel = Path("2026-09-30 *Reunión mensual.wav")
        expected = v5["canonical_hash"](
            {"root": v5["canonical_hash"](str(root)), "relative": rel.as_posix()}
        )
        assert source_id_for(root, rel) == expected

    def test_diarization_cache_key_matches_v5_and_the_real_drive_suffix(self, v5):
        from speakerscribe.batch.profiles import diar_cache_name

        class Cfg:
            diarization_model = "pyannote/speaker-diarization-community-1"
            num_speakers = None
            min_speakers = 1
            max_speakers = 20

        name = diar_cache_name(Cfg(), "SIG")
        assert name == f"SIG_{v5['_diar_profile'](Cfg())}.diar.json"
        # Suffix of the 100 cache files on Drive (verified 2026-10-02).
        assert name.endswith(
            "052e0e42a7cb09a0a65eb210e866503b1db870d758474570cd4c319acade0ccf.diar.json"
        )


class TestSpeakers:
    @pytest.mark.parametrize("seed", range(25))
    def test_orphan_repair_identical(self, v5, seed):
        segs = random_segments(seed)
        for tol in (0.0, 0.5, 2.0):
            assert repair_orphans(segs, tol) == tuple(v5["reparar_hablantes"](segs, tol))

    def test_renames_identical(self, v5):
        segs = random_segments(3)
        mapping = {"SPEAKER_00": "SPEAKER_01", "SPEAKER_01": "Ana"}
        assert apply_renames(segs, mapping) == v5["aplicar_renombres"](segs, mapping)


class TestRenderers:
    @pytest.mark.parametrize("seed", range(20))
    @pytest.mark.parametrize("ms", [False, True])
    def test_lenient_transcript_identical(self, v5, seed, ms):
        v5["TIMESTAMP_MILISEGUNDOS"] = ms
        segs = random_segments(seed)
        md = {"diarization_enabled": True, "duration_seconds": 300.0}
        expected = v5["render_transcript"](segs, md, "a/b.wav", "publicado_con_flags_criticos")
        assert r.render_transcript(ctx(segs, md, ms=ms), "publicado_con_flags_criticos") == expected

    def test_strict_transcript_identical_at_v5_tolerance(self, v5):
        v5["TIMESTAMP_MILISEGUNDOS"] = False
        segs = [
            s for s in random_segments(7) if s["speaker"] and s["speaker"] != "SPEAKER_NO_OVERLAP"
        ]
        md = {"diarization_enabled": True}
        assert r.render_transcript(ctx(segs, md), "ok") == v5["render_transcript"](
            segs, md, "a/b.wav", "ok"
        )

    def test_deliberate_change_boundary_drift_tolerated(self, v5):
        """v5 rejected 11 of 94 real files for 0.39-0.85 s boundary overlaps."""
        segs = [
            {"start": 0.0, "end": 5.0, "text": "uno", "speaker": "SPEAKER_00"},
            {"start": 5.2, "end": 5.2, "text": "dos", "speaker": "SPEAKER_01"},
            {"start": 4.6, "end": 8.0, "text": "tres", "speaker": "SPEAKER_00"},  # 0.6 s back
        ]
        md = {"diarization_enabled": True}
        with pytest.raises(ValueError, match="no monotónicos"):
            v5["render_transcript"](segs, md, "a/b.wav", "ok")
        assert "tres" in r.render_transcript(ctx(segs, md, tolerance=1.0), "ok")
        with pytest.raises(r.TranscriptRejectedError, match="no monotónicos"):
            r.render_transcript(ctx(segs, md, tolerance=0.05), "ok")

    @pytest.mark.parametrize("seed", range(15))
    def test_srt_turns_llm_plain_splits_identical(self, v5, seed):
        segs = random_segments(seed)
        md = {"language_detected": "es", "duration_seconds": 4000, "total_words": 321}
        c = ctx(segs, md)
        assert r.render_srt(c) == v5["render_srt"](segs)
        assert r.speaker_turns(segs) == v5["_turnos"](segs)
        full = r.render_full_llm(c)
        assert full == v5["render_full_llm"](segs, md, "a/b.wav")
        assert r.render_plain(c) == v5["render_plano"](segs)
        for words in (5, 40, 1950):
            assert r.render_splits(full, words) == v5["render_splits"](full, words)

    def test_silent_audio_identical(self, v5):
        md = {"diarization_enabled": True, "duration_seconds": 75.4}
        assert r.render_transcript(ctx([], md), "ok") == v5["render_transcript"](
            [], md, "a/b.wav", "ok"
        )

    @pytest.mark.parametrize("seconds", [0, 0.4, 59.5, 3599.999, 7384.25, -3])
    def test_format_ts_identical(self, v5, seconds):
        for ms in (False, True):
            v5["TIMESTAMP_MILISEGUNDOS"] = ms
            assert r.format_ts(seconds, ms) == v5["format_ts"](seconds)


class TestNaming:
    def test_deliberate_change_star_is_kept(self):
        """D1 (2026-10-02): '*' marks 'not yet summarized'; v5 replaced it with '_'."""
        assert sanitize_name("2026-09-24 *Taller de planeación") == (
            "2026-09-24 *Taller de planeación"
        )
        assert sanitize_name("a/b\x00c") == "a_b_c"


class TestJournalIndex:
    def test_indices_match_v5_on_a_synthetic_journal(self, v5):
        def ev(event, sid=None, job=None, sig=None, profile=None, **kw):
            rec = {"event": event, **kw}
            if job:
                rec["job_id"] = job
            if sid:
                rec["source"] = {"id": sid, "content_signature": sig}
            if profile:
                rec["profile"] = {"id": profile}
            return rec

        log = [
            ev("processing", "s1", "j1"),
            ev("failed_retryable", "s1", "j1"),
            ev("quality_rejected", "s1", "j1"),
            ev("completed", "s1", "j2", "sigA", "p1"),
            ev("completed", "s2", "j3", "fast:x", "p1"),
            ev("invalid_media", "s3"),
            ev("content_change_detected", "s2"),
            ev("speakers_renamed", "s1", rename_sha="abc"),
            ev("republished", "s1", "j4", "sigA", "p1"),
        ]
        ours = JournalIndex.build(log)
        theirs = v5["build_indices"](log)
        assert ours.attempts == theirs["intentos"]
        assert ours.ok_by_source == theirs["ok_por_fuente"]
        assert ours.ok_by_content_profile == theirs["ok_por_contenido_perfil"]
        assert ours.invalid == theirs["invalidos"]
        assert ours.dirty == theirs["sucias"]
        assert ours.renames == theirs["renombres_hash"]

    def test_real_journal_if_available(self):
        """Local-only check against the real events.jsonl (never committed)."""
        real = Path(os.environ.get("SPEAKERSCRIBE_REAL_JOURNAL", "/nonexistent"))
        if not real.is_file():
            pytest.skip("journal real no disponible (solo verificación local)")
        events = [json.loads(line) for line in real.read_text().splitlines() if line.strip()]
        index = JournalIndex.build(events)
        assert len(index.ok_by_source) >= 94
        assert all(n == 1 for n in index.attempts.values())
        assert {r["event"] for recs in index.ok_by_source.values() for r in recs} <= {"completed"}
