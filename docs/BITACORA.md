# Bitácora del proyecto

Registro cronológico de qué se hizo, por qué y con qué evidencia. Las entradas nuevas van **arriba**. Fechas en UTC.

Convenciones:
- **HECHO**: verificado, con su fuente.
- **DECISIÓN**: elegido por el dueño del proyecto.
- **PENDIENTE**: lo que falta, quién lo hace y cómo.

---

## 2026-10-02 (tarde) — Primera corrida del dueño y notebook de publicación 0.4

### Contexto
El dueño descargó la rama de trabajo como zip, la subió a `Pruebas/Speakerscribe/` en Drive y ejecutó `notebooks/speakerscribe_lote.ipynb` con *Ejecutar todo*. No se procesó ningún audio.

### Diagnóstico (HECHO, salidas del notebook)
- El censo encontró 12 archivos pendientes en la carpeta `data/` nueva.
- La celda 6 se detuvo en 5,6 s con `WorkspaceBindingError`. El estado estaba registrado con `Pruebas/Speakerscribe/data` y la entrada actual es `…/Transcripcion-Diarizacion/data`.
- Es la protección prevista: no se gastó GPU ni se tocó nada. Falta revincular **una vez** (celda 4, `REVINCULAR = True`).
- La máquina quedó encendida, porque un error del orquestador no apaga.

### Cambios (HECHO)
- **Notebook de publicación adaptado y versionado** en `notebooks/publicacion/`. Antes, sus celdas D podían borrar de GitHub todo lo que faltara en Drive, retroceder la versión y romper `pyproject.toml`. Ahora:
  - La versión se lee de `__init__.py`.
  - Nueva Celda S: Drive ← GitHub, con respaldo y sin borrar nada.
  - Tag y PyPI solo con `PUBLICAR_RELEASE = True`.
  - Las pruebas corren sobre lo que se publica.
  - Los notebooks salen sin salidas de Colab.
  - No se crean commits vacíos.
  - Guardias contra versión menor, borrados, contenido viejo y notebooks con valores locales.
- `tests/test_cli.py` ya no importa `click`: typer 0.27 dejó de depender de él y la suite fallaba en entornos livianos.

### Verificación (HECHO)
Simulación de extremo a extremo del notebook de publicación: git real contra un remoto local y Colab simulado. Todos los escenarios dieron el resultado esperado:
- Drive con 0.3.0: se detiene.
- Celda S y luego publicar: sin borrados, sin commit vacío, `.gitignore` y `pyproject.toml` intactos, sin datos.
- Archivo borrado en Drive: se detiene.
- Mismo número de versión con contenido viejo: se detiene.
- Versión menor: se detiene.
- `PUBLICAR_RELEASE = True`: crea el tag.
- Edición real: se publica.
- Notebook ejecutado en Colab: no se filtran salidas ni el nombre de usuario.
- Notebook viejo ejecutado en Colab: se detiene.
- `PROBAR_SOLO` lleno o bandera en `True`: se detiene.

---

## 2026-10-02 — 0.3.1 y 0.4.0: del fallo de PyAV a un lote reanudable y probado

### Contexto
- El 2026-10-01, entre las 02:06 y las 02:51, el lote de Colab (notebook v5) falló en **6 de 6** archivos con `TypeError: open() got an unexpected keyword argument 'metadata_errors'`.
- Cada fallo ocurrió *después* de diarizar, así que se perdieron unos 45 min de GPU T4.
- Además, el notebook apagó la máquina y con ello se perdió el traceback.

### Diagnóstico (HECHO)
- **PyAV 19.0.0** (publicado el 2026-09-29) eliminó el argumento `metadata_errors` de `av.open()`.
  - **faster-whisper 1.2.1**, la última versión publicada, todavía lo usa.
  - El arreglo upstream (SYSTRAN/faster-whisper#1495) se fusionó el 2026-09-30, pero sigue sin publicarse.
  - Fuentes: https://pypi.org/project/av/#history · https://github.com/SYSTRAN/faster-whisper/pull/1495
- Colab usa Python 3.13, y `pip` instala la última versión de PyAV disponible.
- Estado en Drive según la API (consultado el 2026-10-02):
  - journal con 463 eventos y 94 transcripciones confirmadas;
  - 100 cachés de diarización, de ellas 6 corresponden a los pendientes;
  - 7 audios en `data/`.
- Las 11 alertas «Timestamps no monotónicos» vienen de la deriva del alineamiento por palabra en los bordes de los segmentos (0,39–0,85 s); no indican un fallo de diarización.

### Decisiones del dueño del proyecto
| Id | Decisión |
|---|---|
| D1 | El `.txt` conserva el nombre exacto del audio, incluido el `*` (el `*` significa «aún no resumido con un LLM»). |
| D2 | Revincular el estado a la nueva carpeta de entrada. |
| D3/D4 | `cursos/`, `cursos2/` y las salidas históricas se quedan donde están. |
| D5 | Activar el texto para LLM (`.full_for_llm.txt`). |
| D6 | El dueño configura el *trusted publisher* de PyPI. |
| Q8 | La caché de diarización se conserva 3 meses (90 días). |
| Q9 | Borrar los audios tras una exportación exitosa. Implementado como cuarentena de 30 días en `_procesados/`. |
| Q10 | Una sola raíz de resultados: `ProColombia/1B. Resultados/Transcripcion-Diarizacion`. |
| Q11 | Se acepta el tope `av<19` y la versión 0.3.1. |
| Q13 | GitHub y PyPI deben tener siempre la última versión. |
| Q18 | **No reescribir el historial de git**: el historial de cambios debe conservarse (respuesta del 2026-10-02). |
| Q20 | El borrador del pipeline por etapas va en la rama `wip/pipeline-pkg`. |
| Q21 | Prueba de humo con un audio corto. |
| Q22 | Un PR por fase. |
| — | PyPI se publica **solo después** de una prueba exitosa del dueño en Colab. |
| — | El notebook de publicación de Drive se conserva. Se adaptó a 0.4: ver la entrada siguiente y [GUIA_LOTE.md § 9](GUIA_LOTE.md#9-notebook-de-publicación). |

### Cambios (HECHO, todos en `main`)
| PR | Versión | Qué |
|---|---|---|
| [#5](https://github.com/EnriqueForero/speakerscribe/pull/5) | 0.3.1 | La transcripción ya no usa PyAV: el WAV PCM16 se lee con la librería estándar. Tope `av>=11,<19`. Autoprueba de decodificación. Corte del lote ante errores de entorno. Ajuste de solapes ≤1 s. CI en 3.10–3.13 con integración real. |
| [#6](https://github.com/EnriqueForero/speakerscribe/pull/6) | — | Higiene: `_runs.jsonl` sale del repo sin reescribir el historial. Acciones de GitHub actualizadas a v6/v8. Una sola ruta de publicación. |
| [#7](https://github.com/EnriqueForero/speakerscribe/pull/7) | 0.4.0 | `speakerscribe.batch`: el notebook v5 pasa a ser un paquete con pruebas. Notebook nuevo `notebooks/speakerscribe_lote.ipynb`. |
| [#8](https://github.com/EnriqueForero/speakerscribe/pull/8) | 0.4.0 | Bitácora, guía de operación y `PROBAR_SOLO` en el notebook. |

Otros:
- Rama `wip/pipeline-pkg`: borrador recuperado de Drive; no está integrado.
- PRs de Dependabot #1–#4 cerrados, porque sus cambios ya estaban incluidos.
- `_audio_temp/` (1 WAV temporal de 127 MB) enviado a la papelera de Drive con aprobación del dueño. Se puede recuperar durante 30 días.

### Verificación (HECHO)
- CI 9/9 en verde en cada PR: lint, mypy, lint de notebooks, unitarias 3.10–3.13 e integración con decodificador real.
- 430 pruebas locales con cobertura del 81,6 %.
- Las pruebas de caracterización reproducen el comportamiento del notebook v5 congelado.
- **Gemelo digital** del Drive real (journal real, nombres reales y motor simulado; solo en local, nada se subió):
  1. Pide revincular.
  2. La prueba con un archivo sale bien.
  3. El lote procesa 6/6.
  4. Los 99 `.txt` existentes quedan intactos y los audios pasan a `_procesados/`.
- Las 6 cachés de diarización de los pendientes existen en Drive con la clave esperada y quedan protegidas de la poda.

### PENDIENTE
| Qué | Quién | Cuándo | Cómo |
|---|---|---|---|
| Prueba real en GPU T4 con un audio corto | Dueño | Próxima sesión de Colab | [GUIA_LOTE.md § Primera vez](GUIA_LOTE.md#4-primera-vez-después-de-esta-actualización) |
| Publicar 0.4.0 en PyPI | Dueño + Claude | Después de una prueba exitosa | [GUIA_LOTE.md § Publicar](GUIA_LOTE.md#7-publicar-una-versión-en-pypi) |
| Sincronizar la copia de código en Drive con GitHub | Dueño | Antes de volver a usar el notebook de publicación | Celda S ([GUIA_LOTE.md § 9](GUIA_LOTE.md#9-notebook-de-publicación)) |
| Quitar el tope `av<19` | Mantenimiento | Cuando faster-whisper publique el arreglo #1495 | Subir el mínimo de faster-whisper y retirar el tope |
