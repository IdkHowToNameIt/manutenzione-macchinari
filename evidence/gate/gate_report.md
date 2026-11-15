# Gate di rilascio v2

Decisione: **accettata** (2026-11-15T14:18:25+00:00)

| ID | Requisito | Misura | Soglia | Esito | Motivazione |
| --- | --- | --- | --- | --- | --- |
| R1 | Accuratezza della candidata non inferiore alla baseline | v2 0.7100 vs v1 0.6650 | v2 >= v1 | superato | Il protocollo comune richiede accuratezza candidata almeno pari alla baseline. |
| R2 | Recall sulla classe 1 (guasto entro 7 giorni) | v2 0.6533 | >= 0.65 | superato | Soglia a priori sulla classe 1: un falso negativo e' un intervento mancato. |
| R3 | Recall sulla classe 1 non inferiore alla baseline | v2 0.6533 vs v1 0.3467 | v2 >= v1 | superato | Un guadagno di accuratezza non deve ridurre la copertura dei guasti. |
| R4 | Latenza p95 della candidata su 30 misure locali | 57.87 ms | <= 120 ms | superato | Limite del protocollo comune per la risposta di predict. |
| R5 | Error rate della candidata su 30 misure locali | 0.0000 | <= 0.01 | superato | Errori HTTP >= 400 sulle richieste valide della probe. |
| R6 | Probe v2 attribuita all'artefatto registrato | fingerprint misurato 0c1f3e2530b46e9094632f103209be352042e5178bf8af9fe7538859192bb799 | uguale al fingerprint dei metadati v2 | superato | Una probe di un altro artefatto non autorizza l'attivazione. |
| R7 | Stesso dataset, stessi ID di validation e stesso schema | stesso_dataset=True, stessi_id_validation=True, stesso_schema=True | tutte le uguaglianze vere | superato | Il confronto tra versioni e' valido solo a parita' di dati e protocollo. |

## Confronto sintetico

| Metrica | v1 | v2 |
| --- | ---: | ---: |
| accuratezza | 0.665 | 0.71 |
| recall_target_1 | 0.3466666666666667 | 0.6533333333333333 |
| precision_target_1 | 0.5909090909090909 | 0.6049382716049383 |
| roc_auc | 0.7181866666666666 | 0.75712 |
| soglia_decisione | 0.5 | 0.46 |
| p95_ms | 2.856 | 57.87 |
| error_rate | 0.0 | 0.0 |
| run_id | 8707d4c4859f423c86a0ba9b007a8661 | 474c37426a8b42578626e490ad9192c6 |
| artifact_fingerprint | 6eea94e288ab41dbfa515a61efcce53507aecb71c9b56c5f564b2cd964eb9dc0 | 0c1f3e2530b46e9094632f103209be352042e5178bf8af9fe7538859192bb799 |

## Nota

La decisione del gate non cambia la versione attiva: l'attivazione e' un'operazione separata e registrata nel registro delle transizioni.
