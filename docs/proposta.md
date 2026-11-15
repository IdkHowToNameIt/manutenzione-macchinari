# Proposta di progetto - Manutenzione predittiva

Documento preparato prima del codice, secondo il checkpoint della consegna.
Risponde alle sei domande richieste e descrive il percorso dati → modello →
servizio → decisione di aggiornamento.

## 1. Evento previsto e uso della previsione

Prevediamo se un macchinario **richiedera' un intervento per guasto entro 7
giorni** dalla fine del turno osservato. La previsione serve a ordinare le
ispezioni e gli interventi programmati della settimana: i macchinari con
probabilita' alta passano davanti nella lista di controllo, i ricambi vengono
riservati prima del fermo.

## 2. Significato di target 0 e target 1

- **target = 1**: guasto con intervento nei 7 giorni successivi al turno.
- **target = 0**: nessun guasto con intervento nei 7 giorni successivi.

Le due classi sono sbilanciate nel train (501 negativi, 299 positivi): la
classe 1 e' quella che interessa operativamente.

## 3. Momento della previsione e informazioni disponibili

La previsione avviene **a fine turno**. Sensori e contatori descrivono
esclusivamente il periodo precedente: temperature, vibrazioni, rumore, usura,
carichi, contatori e storico manutenzioni sono tutti disponibili in quel
momento. Non servono la label ne' dati successivi al turno. Le 30 feature
dell'API sono le stesse del CSV di training, nello stesso ordine; `record_id`,
`split` e `target` restano fuori dagli input.

## 4. Conseguenze di un falso positivo e di un falso negativo

- **Falso positivo** (previsto guasto, nessun guasto): intervento non
  necessario, fermo pianificato, consumo di ricambi e ore del team.
- **Falso negativo** (guasto non previsto): fermo non pianificato, rischio per
  la sicurezza, costo di riparazione maggiore e ritardo sulla produzione.

Il falso negativo e' considerato piu' grave: per questo il gate richiede un
recall minimo sulla classe 1, non solo una buona accuratezza.

## 5. Metriche e limiti operativi

- **Accuracy**: metrica del protocollo comune, confronto candidata/baseline.
- **Recall classe 1** (metrica del soggetto): quota di guasti realmente
  previsti; soglia di gate 0,65 e nessuna regressione rispetto alla baseline.
- **Precision classe 1 e matrice di confusione**: per leggere il costo dei
  falsi positivi e spiegare gli errori.
- **Limiti del servizio**: p95 locale ≤ 120 ms, error rate ≤ 0,01 sulle
  richieste valide, input incompleto → 422 prima della chiamata al modello.

## 6. Attivita' per componente

| Persona | Attivita' principali |
| --- | --- |
| Matteo | Dati e training: lettore CSV, contratto feature, baseline v1, candidata v2 |
| Livia | Servizio e telemetria: FastAPI, validazione, log, metriche, probe |
| Andrea | Registro e rilascio: versioni, fingerprint, gate operativo, attivazione e rollback, Docker |
| Nicolo | Documentazione e verifica: README, relazione, presentazione, revisione delle evidenze |

Il dettaglio di compiti e revisioni incrociate e' in un documento interno del
gruppo, non incluso nel repository.

## Percorso del progetto

```
data-input/dataset/dataset.csv (1000 righe, seed 1302)
        │
        │ lettore + controlli (S1): intestazioni, ID, valori finiti,
        │ classi e split attesi, fingerprint SHA-256 dei byte
        ▼
train 800 righe ── training (S2, S4) ──► MLflow run
validation 200 righe                        │
                                            ▼
                        models/v1 e models/v2: model.pkl + model_metadata.json
                        (run ID, fingerprint dati e artefatto, schema, metriche)
                                            │
                                            │ verifica al caricamento: schema,
                                            │ ordine feature, fingerprint artefatto
                                            ▼
                        servizio FastAPI (S3)
                        models/active_version.json → artefatto realmente servito
                                            │
                                            ▼
                        /predict, /health, /metrics, /admin/activate, /admin/rollback
                                            │
                                            │ log JSONL senza payload + metriche a finestra
                                            ▼
                        osservazioni: richieste assegnate, probe 30 misure, drill, SHAP
                                            │
                                            ▼
                        decisione di aggiornamento: gate (S4)
                        attivazione o ripristino registrati (S5)
```

## Punti di verifica e recupero della versione precedente

| Passaggio | Come viene verificato | Evidenza |
| --- | --- | --- |
| Lettura dati | controllo bloccante su file, colonne, ID, valori finiti, classi, split | `evidence/data/profilo_dati.json` |
| Training | metriche su validation, run MLflow con fingerprint di dati e schema | `evidence/training/v1.json`, `v2.json` |
| Artefatto | fingerprint SHA-256 dei byte e schema al caricamento | `models/*/model_metadata.json` |
| Servizio | `/health` con versione, run ID e fingerprint | `evidence/requests/client_inprocess.json` |
| Probe | 30 misure locali, p95, error rate, attribuzione all'artefatto | `evidence/serving/probe_*.json` |
| Gate | una riga per requisito con misura, soglia ed esito | `evidence/gate/gate_report.md` |
| Transizioni | stato prima/dopo, attore, run ID, fingerprint | `models/active_version.json`, `models/transitions.jsonl` |
| Ripristino | `/admin/rollback` ricarica la versione precedente e la predizione successiva lo conferma | `evidence/transitions/drill_responses.json` |
