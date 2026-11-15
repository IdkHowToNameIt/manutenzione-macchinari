# Riepilogo delle evidenze

Generato il 2026-11-15T14:28:07+00:00 da `scripts/10_riepilogo_evidenze.py`.

## Dati (S1)

- Dataset: `data-input/dataset/dataset.csv`
- Fingerprint SHA-256: `af700d09fac13f09e5b0b8f8713cc0c9d3dca433a8f53c50273fe4647d1c5f3d`
- Righe: 1000, feature: 30
- Schema feature: versione 1.0, fingerprint `f6fc3f70fa153777a141faee4100bba8591e5aef4f71d6a025c9c97998eb177a`
- Split train: target 0 = 501, target 1 = 299
- Split validation: target 0 = 125, target 1 = 75

## Versioni del modello (S2, S4)

| Versione | Modello | Run MLflow | Soglia | Accuracy | Recall (classe 1) | Precision (classe 1) | ROC AUC | FP | FN | Fingerprint artefatto |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v1 | logistic_regression | `8707d4c4859f423c86a0ba9b007a8661` | 0.5 | 0.665 | 0.3466666666666667 | 0.5909090909090909 | 0.7181866666666666 | 18 | 49 | `6eea94e288ab41dbfa515a61efcce53507aecb71c9b56c5f564b2cd964eb9dc0` |
| v2 | random_forest | `474c37426a8b42578626e490ad9192c6` | 0.46 | 0.71 | 0.6533333333333333 | 0.6049382716049383 | 0.75712 | 32 | 26 | `0c1f3e2530b46e9094632f103209be352042e5178bf8af9fe7538859192bb799` |

## Probe del servizio (S4)

| Versione | p50 (ms) | p95 (ms) | Error rate | Probe attribuita | Ambiente |
| --- | ---: | ---: | ---: | --- | --- |
| v1 | 1.741 | 2.856 | 0.0 | True | Python 3.12.16, 8 CPU |
| v2 | 39.58 | 57.87 | 0.0 | True | Python 3.12.16, 8 CPU |

## Gate di rilascio (S4)

Decisione: **accettata** (2026-11-15T14:18:25+00:00)

| ID | Requisito | Misura | Soglia | Esito |
| --- | --- | --- | --- | --- |
| R1 | Accuratezza della candidata non inferiore alla baseline | v2 0.7100 vs v1 0.6650 | v2 >= v1 | superato |
| R2 | Recall sulla classe 1 (guasto entro 7 giorni) | v2 0.6533 | >= 0.65 | superato |
| R3 | Recall sulla classe 1 non inferiore alla baseline | v2 0.6533 vs v1 0.3467 | v2 >= v1 | superato |
| R4 | Latenza p95 della candidata su 30 misure locali | 57.87 ms | <= 120 ms | superato |
| R5 | Error rate della candidata su 30 misure locali | 0.0000 | <= 0.01 | superato |
| R6 | Probe v2 attribuita all'artefatto registrato | fingerprint misurato 0c1f3e2530b46e9094632f103209be352042e5178bf8af9fe7538859192bb799 | uguale al fingerprint dei metadati v2 | superato |
| R7 | Stesso dataset, stessi ID di validation e stesso schema | stesso_dataset=True, stessi_id_validation=True, stesso_schema=True | tutte le uguaglianze vere | superato |

## Attivazione e ripristino (S5)

Modalita': attivazione_documentata_dopo_gate

| Passo | Stato HTTP | Versione servita | Run ID |
| --- | ---: | --- | --- |
| predizione con v1 | 200 | v1 | `8707d4c4859f423c86a0ba9b007a8661` |
| attivazione v2 | 200 | v2 | `474c37426a8b42578626e490ad9192c6` |
| predizione con v2 | 200 | v2 | `474c37426a8b42578626e490ad9192c6` |
| rollback a v1 | 200 | v1 | `8707d4c4859f423c86a0ba9b007a8661` |
| predizione dopo il ripristino | 200 | v1 | `8707d4c4859f423c86a0ba9b007a8661` |
| richiesta incompleta | 422 | - | `-` |

Error rate del drill: 0.25 (1 errore su 4 richieste).
Tutti i controlli superati: si.

## Spiegazioni SHAP della candidata v2

| Feature | Importanza media (abs SHAP) |
| --- | ---: |
| ore_marcia_ultime_24h | 0.0369 |
| carico_medio_pct | 0.03134 |
| avvii_ultime_24h | 0.030731 |
| ore_utilizzo_totali | 0.025666 |
| manutenzioni_programmate_completate_pct | 0.024841 |

## Controllo di drift illustrativo

| Richiesta | Feature oltre soglia | z-score massimo |
| --- | --- | ---: |
| req_01 | nessuna | 2.597 |
| req_02 | nessuna | 2.359 |
| req_03 | nessuna | 2.423 |

## Stato finale

- Versione attiva: v1
- Versione precedente: v2
- Transizioni registrate: 3
