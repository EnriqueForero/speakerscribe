"""Maintenance helpers: selective deletion, JSON inspection, speaker renaming."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from speakerscribe.config import WorkspacePaths
from speakerscribe.logging_config import logger


def delete_outputs_for(
    paths: WorkspacePaths,
    pattern: str,
    include_diar_cache: bool = False,
) -> int:
    """Delete outputs (txt, srt, json, transcript.md, splits) whose name contains `pattern`.

    Source files in `data/` are never touched. Useful when re-processing a
    single audio without disturbing the others.

    Args:
        paths: WorkspacePaths instance.
        pattern: Substring that must appear in the file name.
        include_diar_cache: When True, also delete the matching pyannote cache.

    Returns:
        Number of files deleted.
    """
    folders = [paths.transcripts, paths.splits]
    if include_diar_cache:
        folders.append(paths.diar_cache)

    deleted = 0
    for folder in folders:
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if f.is_file() and pattern in f.name:
                f.unlink()
                deleted += 1
                logger.debug(f"Deleted: {f.relative_to(paths.base)}")
    logger.info(f"{deleted} file(s) deleted — ready to re-process")
    return deleted


def delete_all_outputs(paths: WorkspacePaths, confirm: str = "") -> int:
    """Delete ALL outputs (txt, srt, json, transcript.md, splits, diar cache).

    Source files in `data/` are NOT touched. Requires explicit confirmation
    to prevent accidents.

    Args:
        paths: WorkspacePaths instance.
        confirm: Must be exactly `"YES DELETE ALL"` for the deletion to proceed.

    Returns:
        Number of files deleted.

    Raises:
        ValueError: If `confirm` is not exactly `"YES DELETE ALL"`.
    """
    if confirm != "YES DELETE ALL":
        raise ValueError(
            'Confirmation required. Pass confirm="YES DELETE ALL" to proceed.'
        )

    deleted = 0
    for folder in [paths.transcripts, paths.splits, paths.diar_cache, paths.audio_tmp]:
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if f.is_file():
                f.unlink()
                deleted += 1
    logger.success(f"{deleted} file(s) deleted — workspace clean")
    return deleted


def inspect_json(json_path: Path) -> dict[str, Any]:
    """Print and return a quick summary of a transcription JSON file.

    Args:
        json_path: Path to the .json file produced by the pipeline.

    Returns:
        Dict with the most useful keys for a glance. Empty dict if the file
        does not exist.
    """
    if not json_path.exists():
        logger.error(f"File not found: {json_path}")
        return {}

    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    info = {
        "audio_file": data.get("audio_file"),
        "model": data.get("model"),
        "diarization_enabled": data.get("diarization_enabled"),
        "speakers_summary": data.get("speakers_summary"),
        "total_segments": data.get("total_segments"),
        "total_words": data.get("total_words"),
        "duration_minutes": data.get("duration_minutes"),
        "real_time_factor": data.get("real_time_factor"),
        "pyannote_version": data.get("pyannote_version"),
        "faster_whisper_version": data.get("faster_whisper_version"),
        "package_version": data.get("package_version"),
        "processed_at": data.get("processed_at"),
    }

    logger.info(f"{json_path.name}")
    for k, v in info.items():
        logger.info(f"   {k:<24}: {v}")
    return info


def rename_speakers_in_outputs(
    paths: WorkspacePaths,
    base_name: str,
    mapping: dict[str, str],
) -> dict[str, int]:
    """Replace SPEAKER_XX labels with real names in ALL outputs of a run.

    Updates `<base_name>.{txt,srt,json,transcript.md}` in `transcripts/` and
    every `<base_name>_*.txt` in `splits/`.

    Args:
        paths: WorkspacePaths instance.
        base_name: Common prefix (e.g. "meeting_2026-04-29_large-v3").
        mapping: `{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}`.

    Returns:
        Dict mapping each filename to the number of replacements made.

    Raises:
        ValueError: If a mapping key does not start with "SPEAKER_".
    """
    if not mapping:
        logger.warning("Empty mapping. Nothing to do.")
        return {}

    for k in mapping:
        if not k.startswith("SPEAKER_"):
            raise ValueError(
                f"Mapping keys must start with 'SPEAKER_'. Got: '{k}'"
            )

    logger.info(f"Renaming speakers for '{base_name}'...")
    for k, v in mapping.items():
        logger.info(f"   {k} -> {v}")

    targets: list[Path] = [
        paths.transcripts / f"{base_name}.txt",
        paths.transcripts / f"{base_name}.srt",
        paths.transcripts / f"{base_name}.transcript.md",
    ]
    targets.extend(paths.splits.glob(f"{base_name}_*.txt"))

    stats: dict[str, int] = {}
    for target in targets:
        if not target.exists():
            continue
        content = target.read_text(encoding="utf-8")
        n_replacements = 0
        for spk_id, name in mapping.items():
            before = content
            content = content.replace(f"[{spk_id}]", f"[{name}]")
            content = content.replace(f"### {spk_id} ·", f"### {name} ·")
            n_replacements += before.count(f"[{spk_id}]") + before.count(f"### {spk_id} ·")
        target.write_text(content, encoding="utf-8")
        stats[target.name] = n_replacements
        logger.info(f"   {target.name}: {n_replacements} replacements")

    json_path = paths.transcripts / f"{base_name}.json"
    if json_path.exists():
        with json_path.open(encoding="utf-8") as f:
            data = json.load(f)
        n_json = 0
        if "segments" in data:
            for seg in data["segments"]:
                if seg.get("speaker") in mapping:
                    seg["speaker"] = mapping[seg["speaker"]]
                    n_json += 1
        if "speakers_summary" in data and data["speakers_summary"]:
            data["speakers_summary"] = {
                mapping.get(k, k): v for k, v in data["speakers_summary"].items()
            }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        stats[json_path.name] = n_json
        logger.info(f"   {json_path.name}: {n_json} segments renamed")

    return stats


__all__ = [
    "delete_all_outputs",
    "delete_outputs_for",
    "inspect_json",
    "rename_speakers_in_outputs",
]
