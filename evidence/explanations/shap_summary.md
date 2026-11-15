# Spiegazioni SHAP della candidata v2

Run di riferimento: `474c37426a8b42578626e490ad9192c6` 
Fingerprint artefatto: `0c1f3e2530b46e9094632f103209be352042e5178bf8af9fe7538859192bb799`

## Importanza media assoluta (validation, 200 righe)

| Posizione | Feature | Importanza media (abs SHAP) | Direzione media |
| ---: | --- | ---: | --- |
| 22 | ore_marcia_ultime_24h | 0.0369 | riduce il rischio |
| 19 | carico_medio_pct | 0.0313 | riduce il rischio |
| 23 | avvii_ultime_24h | 0.0307 | riduce il rischio |
| 13 | ore_utilizzo_totali | 0.0257 | riduce il rischio |
| 29 | manutenzioni_programmate_completate_pct | 0.0248 | riduce il rischio |
| 14 | ore_da_cambio_cuscinetto | 0.0221 | riduce il rischio |
| 21 | corrente_media_a | 0.0212 | riduce il rischio |
| 24 | pressione_media_bar | 0.0211 | riduce il rischio |
| 20 | picco_carico_pct | 0.0210 | riduce il rischio |
| 7 | vibrazione_rms_mm_s | 0.0178 | riduce il rischio |

## Richiesta req_02 (probabilita' classe 1 = 0.5296)

| Feature | Contributo SHAP | Segno |
| --- | ---: | --- |
| manutenzioni_programmate_completate_pct | +0.0402 | verso la classe 1 |
| avvii_ultime_24h | -0.0292 | verso la classe 0 |
| asimmetria_vibrazioni_pct | +0.0238 | verso la classe 1 |
| ore_utilizzo_totali | -0.0226 | verso la classe 0 |
| ore_da_cambio_cuscinetto | -0.0223 | verso la classe 0 |
| ore_fermo_precedenti_90g | +0.0214 | verso la classe 1 |

## Limiti

I valori SHAP descrivono come il modello combina le feature: non dimostrano un rapporto causale con il guasto. La spiegazione vale per questa versione dell'artefatto e per la distribuzione dei dati di training.
