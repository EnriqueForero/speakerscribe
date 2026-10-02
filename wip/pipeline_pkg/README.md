# WIP — `speakerscribe.pipeline` como paquete (NO integrado)

Borrador de refactor de `speakerscribe/pipeline.py` en etapas (preflight,
idempotencia, WAV, diarización, transcripción, salidas, limpieza). Se
recuperó de `Pruebas/Speakerscribe/z. pipeline_pkg_wip/` en Drive el
2026-10-02 para no perderlo.

Estado: escrito contra la API de 0.3.0, **sin pruebas y sin cablear**. No se
importa desde el paquete ni lo recoge el wheel (vive fuera de
`speakerscribe/`). Antes de integrarlo hay que portar los cambios de 0.3.1
(lectura nativa del WAV, `audio_reader`, clamp de bordes, corte ante errores
de entorno) y cubrir cada etapa con pruebas.
