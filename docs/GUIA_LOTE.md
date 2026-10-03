# Guía del lote de transcripción (transferencia de conocimiento)

Para quien opera o mantiene el lote de Drive + Colab. Primero cubre lo esencial y al final los detalles. Lo que se hizo y por qué está en [BITACORA.md](BITACORA.md).

## 1. Qué hace, en una frase

Toma cada audio o video de `data/` y deja un `.txt` con quién habló y cuándo, **con el mismo nombre del audio**. Si la sesión de Colab se corta, la siguiente ejecución sigue donde quedó, sin repetir nada.

## 2. Las carpetas

Todo cuelga de una raíz en Drive: `ProColombia/1B. Resultados/Transcripcion-Diarizacion/`.

| Carpeta | Qué hay | ¿La toco? |
|---|---|---|
| `data/` | Los audios por transcribir | ✅ Aquí deja los audios nuevos |
| `entregables/` | Un `.txt` por grabación y el resumen `_resumen.md` | ✅ Puede renombrarlos o editarlos; el lote lo respeta |
| `entregables/.speakerscribe_state/` | La memoria del lote: registro, cachés y copias maestras | ❌ No borrar ni editar |
| `transcripts/` | Formatos opcionales: `.srt`, `.md`, `.json` | Opcional |
| `splits/` | Texto corrido para pegar en un LLM (`.full_for_llm.txt`) | ✅ |
| `_procesados/AAAA-MM-DD/` | Audios ya transcritos con éxito; se borran solos a los 30 días | Puede recuperar un audio desde aquí |

## 3. Uso normal (cada vez)

1. Ponga los audios en `data/`.
2. Abra `notebooks/speakerscribe_lote.ipynb` en Colab con GPU T4 y use *Entorno de ejecución → Ejecutar todo*.
3. Al terminar, la máquina se apaga sola. Los resultados quedan en `entregables/` y `splits/`.

Solo se edita la **celda 3** (configuración).

## 4. Primera vez después de esta actualización

1. Abra el notebook nuevo: https://colab.research.google.com/github/EnriqueForero/speakerscribe/blob/main/notebooks/speakerscribe_lote.ipynb y guarde una copia en su Drive (*Archivo → Guardar una copia en Drive*). Use **esa** copia de aquí en adelante. No ejecute la de `Pruebas/Speakerscribe/notebooks/`: es la copia de código, y la Celda S la reemplaza.
2. Confirme que el secreto `HF_TOKEN` (🔑) tiene acceso al notebook.
3. Ejecute las celdas **1 y 2**. Si la 2 anuncia un «REINICIO PLANIFICADO», vuelva a ejecutar 1 y 2: es normal y ocurre una sola vez.
4. **Prueba con un solo audio.** En la celda 3 escriba parte del nombre de un audio corto en `PROBAR_SOLO`. Por ejemplo, para `2026-09-28 *Tema X - Cifras.wav` basta con `PROBAR_SOLO = 'Tema X'`. Ejecute la celda 3 y luego la 6.
   - **La primera vez la celda 6 se detiene** con «El estado está vinculado a OTRA carpeta de entrada». Es esperado: `data/` cambió de lugar. En la celda 4 ponga `REVINCULAR = True`, ejecútela **una vez** y vuelva a dejarla en `False`. Después ejecute otra vez la celda 6.
5. Revise el `.txt` en `entregables/`: hablantes, tiempos y texto.
6. **Lote completo.** Deje `PROBAR_SOLO = ''` y use *Ejecutar todo*.

### Qué es `PROBAR_SOLO`
Es un filtro por nombre:
- `''` (vacío) procesa **todo** `data/`.
- Un texto procesa **solo** los audios cuyo nombre **contiene** ese texto, sin distinguir mayúsculas.

En modo prueba la máquina **no** se apaga al final, para que pueda revisar. Los demás audios siguen en `data/` para la corrida completa.

## 5. Qué esperar y qué hacer si algo pasa

| Situación | Qué hace el lote | Qué hace usted |
|---|---|---|
| Colab corta la sesión | Lo confirmado está a salvo; el archivo en curso se rehace | Vuelva a ejecutar todo |
| Error de entorno (librerías, CUDA, token) | Se detiene sin gastar intentos; la máquina **no** se apaga | Ejecute la celda 8 (autopsia) y comparta la salida |
| Colab sin cuBLAS de CUDA 12 (imágenes con CUDA 13) | Antes de cargar modelos instala `nvidia-cublas-cu12`, una vez por máquina (~600 MB, menos de un minuto) | Nada. Si aun así falla: *Cambiar tipo de entorno → Versión del entorno* anterior (p. ej. 2026.07) |
| La diarización falla en un archivo | Lo reintenta en la siguiente sesión; en el último intento lo publica **marcado** | Revise `pendientes_revision.json` |
| Usted renombró o editó un `.txt` (p. ej. quitó el `*`) | Lo respeta y lo informa; no lo regenera | Nada |
| Ya existe un archivo con ese nombre en `entregables/` que el lote no creó | El nuevo sale como `nombre~<id>.txt`; el suyo queda intacto | Nada |
| RAM alta | Al 70 % recicla modelos; al 88 % cierra limpio | Reinicie la sesión y ejecute de nuevo |
| El mismo audio en dos lugares | Se transcribe una vez y la otra copia se reutiliza sin GPU | Nada |
| Quiere nombres reales en vez de SPEAKER_00 | — | Celda 7 (renombrar hablantes, sin GPU) |

## 6. Cómo funciona por dentro (lo mínimo para mantenerlo)

- **Registro (journal):** `events.jsonl` anota cada paso. Un resultado solo cuenta como hecho cuando queda escrito `completed`. Por eso una sesión cortada nunca deja un `.txt` a medias.
- **Copia maestra:** `intermedios/<id>.json.gz` guarda el resultado crudo. Cambiar formatos o nombres de hablantes se resuelve **sin GPU** desde esa copia.
- **Caché de diarización:** si el ASR falla después de diarizar, la diarización queda guardada y el reintento solo paga el ASR.
- **Perfiles:**
  - El de **motor** (modelo, idioma, beam, hablantes, glosario) cuesta GPU si cambia.
  - El de **presentación** (formatos, encabezado) se aplica sin GPU.
- **Código:** `speakerscribe/batch/` en GitHub. Mapa de módulos en [ARCHITECTURE.md](ARCHITECTURE.md#batch-package).

## 7. Publicar una versión en PyPI

Regla del proyecto: **se publica solo después de una prueba exitosa en Colab**.

**Una sola vez (dueño del proyecto):**
1. En https://pypi.org/manage/project/speakerscribe/settings/publishing/ agregue un *trusted publisher* de GitHub con estos datos:
   - owner: `EnriqueForero`
   - repository: `speakerscribe`
   - workflow: `release.yml`
   - environment: `pypi`
2. En GitHub, en *Settings → Environments → New environment*, cree el entorno `pypi`.

Opcional: en *Settings → Environments → pypi*, si GitHub le ofrece *Required reviewers*, agréguese. Así el paso de PyPI espera su clic de aprobación.

**Cada versión:**
1. La versión vive **solo** en `speakerscribe/__init__.py` (`__version__`). El CHANGELOG debe tener su sección, con la fecha.
2. Con `main` en verde, dispare `release.yml` de **una** de estas formas. Las tres son equivalentes:

   | Vía | Cómo |
   |---|---|
   | **A. github.com** | *Releases → Draft a new release → Choose a tag*: escriba `vX.Y.Z` → *Create new tag on publish* (destino `main`) → *Publish release* |
   | **B. Notebook de publicación** | Celdas A → B → **S** → en la Celda A `PUBLICAR_RELEASE = True` → **D.GitHub**. Crea y empuja el tag. |
   | **C. git** | `git tag vX.Y.Z origin/main && git push origin vX.Y.Z` |

3. `release.yml` hace el resto:
   - verifica que el tag coincida con la versión;
   - construye el paquete con el `pyproject.toml` del repositorio;
   - lo publica en PyPI sin tokens;
   - crea el *Release* de GitHub, o le adjunta los archivos si se creó en la web.
4. Verifique:
   - el flujo en verde en *Actions*;
   - la versión en https://pypi.org/project/speakerscribe/;
   - `pip install speakerscribe==X.Y.Z` en un Colab nuevo.

**Si falla el paso de PyPI** (p. ej. el *trusted publisher* mal configurado): corríjalo y use *Re-run failed jobs* en *Actions*. **No** cree otro tag.

**Es irreversible:** PyPI no permite volver a subir un mismo archivo, ni aunque se borre ([PyPI, file name reuse](https://pypi.org/help/#file-name-reuse)). Un error se corrige con una versión nueva (p. ej. 0.4.1) y, si hace falta, marcando la mala como *yanked*.

**Emergencia (sin GitHub Actions):** la celda **D.PyPI** del notebook sube con un token (`PYPI_TOKEN`).
- Está apagada por defecto: se activa con `CONFIRMAR_PYPI_DIRECTO = True`.
- Solo publica si su Drive es idéntico a GitHub `main`.
- Usa el mismo `pyproject.toml` del repositorio.

Mientras 0.4.0 no esté en PyPI, el notebook la instala desde GitHub `main`: la celda 2 lo hace sola.

## 8. Sincronizar la copia de código en Drive

`Pruebas/Speakerscribe/` es una **copia de desarrollo**. GitHub `main` es la fuente de verdad. Para ponerla al día, use la **Celda S** del notebook de publicación (sección 9). Si no lo tiene a mano, esta celda hace lo mismo en cualquier Colab con Drive montado:

- No toca `data/` ni resultados.
- Conserva sus notebooks propios.
- Todo lo que reemplaza lo mueve a `z. Backups/sync_<fecha>/`; nada se borra.
- Ejecutarla dos veces no cambia nada la segunda vez.

```python
# 🟡 Sincronizar la copia de desarrollo en Drive con GitHub (main)
import datetime
import filecmp
import shutil
import subprocess
from pathlib import Path

DESTINO = Path('/content/drive/MyDrive/Pruebas/Speakerscribe')   # ⇦ su carpeta de código en Drive
REPO = 'https://github.com/EnriqueForero/speakerscribe'
TMP = Path('/content/ss_repo')
RESPALDO = DESTINO / 'z. Backups' / f'sync_{datetime.datetime.now():%Y%m%d_%H%M%S}'
ESPEJO = ('speakerscribe', 'tests', 'docs', 'scripts', '.github')   # quedan idénticas a GitHub
SOLO_AGREGAR = ('notebooks',)                                         # se conservan sus notebooks propios


def _respaldar(ruta):
    destino = RESPALDO / ruta.relative_to(DESTINO)
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(ruta), str(destino))


def _copiar(origen, destino):
    cambios = 0
    for archivo in origen.rglob('*'):
        if archivo.is_dir() or '__pycache__' in archivo.parts:
            continue
        objetivo = destino / archivo.relative_to(origen)
        if objetivo.exists() and filecmp.cmp(archivo, objetivo, shallow=False):
            continue
        if objetivo.exists():
            _respaldar(objetivo)
        objetivo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archivo, objetivo)
        cambios += 1
    return cambios


shutil.rmtree(TMP, ignore_errors=True)
subprocess.run(['git', 'clone', '--depth', '1', '--quiet', REPO, str(TMP)], check=True)
total = sobrantes = 0
for carpeta in (*ESPEJO, *SOLO_AGREGAR):
    total += _copiar(TMP / carpeta, DESTINO / carpeta)
for carpeta in ESPEJO:  # lo que ya no existe en GitHub va al respaldo (no se borra)
    for archivo in sorted((DESTINO / carpeta).rglob('*'), reverse=True):
        if archivo.is_file() and not (TMP / carpeta / archivo.relative_to(DESTINO / carpeta)).exists():
            _respaldar(archivo)
            sobrantes += 1
for archivo in TMP.iterdir():  # raíz: pyproject.toml, README.md, CHANGELOG.md…
    objetivo = DESTINO / archivo.name
    if archivo.is_file() and not (objetivo.exists() and filecmp.cmp(archivo, objetivo, shallow=False)):
        if objetivo.exists():
            _respaldar(objetivo)
        shutil.copy2(archivo, objetivo)
        total += 1
version = (TMP / 'speakerscribe' / '__init__.py').read_text().split('__version__ = "')[1].split('"')[0]
print(f'✔ Drive = GitHub main (v{version}): {total} actualizado(s), {sobrantes} obsoleto(s) movido(s).')
if RESPALDO.exists():
    print(f'  Lo reemplazado quedó en: {RESPALDO}')
```

## 9. Notebook de publicación

Es el notebook del dueño para publicar desde Drive en GitHub y PyPI. El 2026-10-02 se adaptó a 0.4 y quedó versionado en `notebooks/publicacion/Publicacion_GitHub_PyPI_speakerscribe.ipynb`.

**Cómo obtenerlo y dónde guardarlo**
1. Ábralo en Colab: https://colab.research.google.com/github/EnriqueForero/speakerscribe/blob/main/notebooks/publicacion/Publicacion_GitHub_PyPI_speakerscribe.ipynb
2. Use *Archivo → Guardar una copia en Drive* y mueva la copia a la **raíz** de `Pruebas/Speakerscribe/`. Los `.ipynb` de la raíz nunca se publican, así que ahí puede cambiar banderas sin riesgo.
3. No edite la copia de `notebooks/publicacion/`. Es la referencia que trae la Celda S.

**Flujo**

| Paso | Celda | Qué hace |
|---|---|---|
| 1 | A | Configuración. La versión **no** se escribe: se lee de `speakerscribe/__init__.py`. |
| 2 | B | Carga las funciones. |
| 3 | S | Pone Drive igual a GitHub `main`. Lo que reemplaza va a `z. Backups/sync_<fecha>/`. Úsela **antes** de editar código en Drive. |
| 4 | — | Edite el código en Drive. |
| 5 | D.GitHub | Corre las pruebas sobre lo que se va a publicar, aplica las guardias y sube a `main`. |

**Banderas de la Celda A**

| Bandera | Por defecto (`False`) | Con `True` |
|---|---|---|
| `PUBLICAR_RELEASE` | Solo sube el código | Además crea el tag `vX.Y.Z` y `release.yml` publica en PyPI. Úsela **solo** tras una prueba exitosa. |
| `PERMITIR_BORRADOS` | Se detiene si borraría archivos de GitHub | Permite esos borrados |
| `PERMITIR_RETROCESOS` | Se detiene si un archivo vuelve a una versión anterior | Permite ese retroceso |

**Guardias.** La publicación se detiene **sin tocar GitHub** si:
- Drive tiene una versión menor que GitHub.
- Borraría archivos de GitHub.
- Un archivo de Drive es idéntico a una versión **anterior** del mismo archivo en GitHub. Esto detecta una copia vieja aunque tenga el mismo número de versión. Los notebooks se comparan por el texto de sus celdas.
- Un notebook trae `PROBAR_SOLO` lleno (parte del nombre de una reunión real) o una bandera en `True`.

**Otras protecciones**
- Los notebooks se publican **sin salidas** de Colab. Si solo cambiaron salidas o metadatos, se conserva el de GitHub.
- No se crean commits vacíos.
- Se respetan el `.gitignore` y el `pyproject.toml` del repositorio, cuya versión es dinámica.

Las celdas de consulta (árbol, comparar versiones, estado del repositorio, descargar una versión antigua) no cambiaron.

## 10. Glosario

| Término | Significado |
|---|---|
| Diarización | Saber **quién** habla y cuándo (pyannote) |
| ASR | Pasar voz a texto (Whisper) |
| RTF | Veces más rápido que el tiempo real (RTF 20 = 1 h de audio en 3 min) |
| Revincular | Decirle al lote que la carpeta `data/` cambió de lugar a propósito |
| Marcado / con flags | Se publicó con advertencias de calidad; revise el encabezado `estado:` del `.txt` |
| Trusted publisher | Permiso de PyPI para que GitHub Actions publique sin contraseñas ni tokens |
