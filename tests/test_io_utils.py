"""Atomic write semantics: destination is never a torn file."""

from __future__ import annotations

import json

import pytest

from speakerscribe.io_utils import append_jsonl_line, atomic_write_json, atomic_write_text


class TestAtomicWriteText:
    def test_writes_and_replaces(self, tmp_path):
        target = tmp_path / "out.txt"
        atomic_write_text(target, "v1")
        atomic_write_text(target, "v2")
        assert target.read_text() == "v2"
        assert not list(tmp_path.glob("*.tmp")), "temp file must not survive"

    def test_creates_parents(self, tmp_path):
        target = tmp_path / "a" / "b" / "out.txt"
        atomic_write_text(target, "x")
        assert target.read_text() == "x"

    def test_failure_leaves_previous_version_intact(self, tmp_path, monkeypatch):
        """If the replace step fails, the destination keeps the OLD content."""
        target = tmp_path / "out.txt"
        atomic_write_text(target, "previous-complete-version")

        import speakerscribe.io_utils as iu

        def boom(src, dst):
            raise OSError("simulated crash during replace")

        monkeypatch.setattr(iu.os, "replace", boom)
        with pytest.raises(OSError):
            atomic_write_text(target, "new-version-that-never-lands")
        assert target.read_text() == "previous-complete-version"
        assert not list(tmp_path.glob("*.tmp"))


class TestAtomicWriteJson:
    def test_roundtrip(self, tmp_path):
        target = tmp_path / "meta.json"
        atomic_write_json(target, {"a": 1, "ñ": "sí"})
        assert json.loads(target.read_text()) == {"a": 1, "ñ": "sí"}

    def test_non_serializable_fails_before_touching_disk(self, tmp_path):
        target = tmp_path / "meta.json"
        with pytest.raises(TypeError):
            atomic_write_json(target, {"bad": object()})
        assert not target.exists()


class TestAppendJsonl:
    def test_appends_lines(self, tmp_path):
        target = tmp_path / "ledger.jsonl"
        append_jsonl_line(target, {"id": 1})
        append_jsonl_line(target, {"id": 2})
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        assert [r["id"] for r in rows] == [1, 2]
