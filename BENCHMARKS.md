# Benchmarks

> Estado: **plantilla** — las celdas se llenan con corridas reales en Colab T4
> usando `speakerscribe bench` contra el *golden set* del proyecto.
> Sin estos números, cualquier afirmación de velocidad/calidad es marketing.

## Protocolo

1. **Golden set** (mínimo): 1 audio limpio 2 hablantes (~30 min), 1 reunión
   3-4 hablantes con solapamientos (~45 min), 1 audio difícil (acentos/ruido).
2. **Referencias**: transcripción humana (`ref.txt`) y, si es posible,
   diarización de referencia (`ref.rttm`).
3. **Medición**:
   ```bash
   pip install 'speakerscribe[bench]'
   speakerscribe bench --workspace WS --base-name "<stem>_<modelo>" \
       --ref golden/ref.txt --rttm golden/ref.rttm
   ```
4. El RTF real sale del ledger (`_runs.jsonl`, campo `rtf`). El DER aquí es
   **end-to-end** (etiquetas del transcript final), que es lo que el lector
   experimenta — no el DER aislado del diarizador.

## Resultados (T4, fp16) — PENDIENTE DE CORRIDA

| Config | Audio | RTF (×) | WER | DER e2e | Notas |
|---|---|---:|---:|---:|---|
| 0.2.x secuencial, segment-assign | golden-1 | _ | _ | _ | línea base |
| 0.3.0 batch=8, word-assign | golden-1 | _ | _ | _ | objetivo: tiempo ≤0.4× base; DER −30-50% en turnos rápidos |
| 0.3.0 batch=8, word-assign | golden-2 (solapado) | _ | _ | _ | |
| 0.3.0 + retry anti-loop | golden-3 (difícil) | _ | _ | _ | flags antes/después |

## Criterios de aceptación (del plan de junio 2026)

- Paridad o mejora de WER con batched vs secuencial (|Δ| < 1 pt absoluto).
- Tiempo total del batch ≤ 0.4× la línea base secuencial.
- Cero archivos `"ok"` con diarización fallida (deben ser `ok_degraded`).
- Re-ejecución del batch completo: 100% `skipped` en < 30 s.
