# Relazione del progetto - Manutenzione predittiva di macchinari

## 1. Sintesi

Il progetto costruisce un servizio che stima la probabilita' di un intervento
per guasto entro 7 giorni su un macchinario, usando le misure disponibili a fine
turno. Il lavoro segue il percorso dati → training → artefatto e metadati →
servizio → osservazioni → decisione di aggiornamento.

Risultati principali:

- baseline **v1** (StandardScaler + LogisticRegression): accuracy 0,665,
  recall classe 1 0,347, ROC AUC 0,718;
- candidata **v2** (RandomForest, soglia calibrata sul train): accuracy 0,710,
  recall classe 1 0,653, ROC AUC 0,757;
- **gate accettato**: accuratezza e recall non inferiori alla baseline, p95
  della probe 57,87 ms (limite 120 ms), error rate 0,00 (limite 0,01);
- **attivazione e ripristino dimostrati** con la sequenza v1 → v2 → v1, log
  correlati e stato finale sulla baseline;
- servizio provato in-process, via HTTP reale e in un container Docker.

L'esito non e' un via libera produttivo: la validation e' un confronto
didattico sullo stesso split, il dataset e' sintetico e il margine di v2 sul
recall e' stretto. I limiti sono nella sezione 13.

## 2. Soggetto e domanda di previsione

**Domanda**: quali macchinari potrebbero richiedere un intervento per guasto
entro 7 giorni?

**Momento di osservazione**: fine del turno. Sensori e contatori descrivono solo
il periodo precedente.

- **target = 1**: guasto con intervento nei 7 giorni successivi.
- **target = 0**: nessun guasto nei 7 giorni successivi.

La previsione serve a ordinare le ispezioni e gli interventi programmati e a
riservare i ricambi. Un **falso negativo** (guasto non previsto) provoca un
fermo non pianificato ed e' il costo principale; un **falso positivo** consuma
ore del team e ricambi su un intervento non necessario. Per questo il gate non
guarda solo l'accuratezza ma anche il recall sulla classe 1.

## 3. Dati e contratto (S1)

### 3.1 File assegnati

| File | Uso | Righe |
| --- | --- | --- |
| `data-input/dataset/dataset.csv` | training e validation, con `record_id`, `split`, 30 feature, `target` | 1000 (800 train, 200 validation) |
| `data-input/requests/richieste_monitoraggio.csv` | input operativi per il servizio; **non** e' un dataset di training | 4 (3 valide, 1 incompleta) |

Fingerprint SHA-256 del dataset:
`af700d09fac13f09e5b0b8f8713cc0c9d3dca433a8f53c50273fe4647d1c5f3d`.

Split e classi (non rigenerati):

| Split | target 0 | target 1 | Totale |
| --- | ---: | ---: | ---: |
| train | 501 | 299 | 800 |
| validation | 125 | 75 | 200 |

### 3.2 Contratto delle feature

Le 30 feature sono descritte in `config/feature_schema.json` con posizione,
nome, unita', intervallo della simulazione e tipo. Lo stesso file e' usato da
lettore, training, registro e servizio: l'ordine della lista `features` nell'API
e' l'ordine del file e non puo' divergere tra le fasi. Il fingerprint dello
schema (`f6fc3f70...`) viene salvato nei metadati di ogni versione e controllato
al caricamento.

Un solo file descrive gli input: questo evita il caso classico in cui il
training usa un ordine di colonne e il servizio un altro, producendo predizioni
plausibili ma sbagliate.

### 3.3 Controlli del lettore

`src/manutenzione/data_loader.py` rifiuta, con messaggi espliciti:

- file mancante o vuoto, intestazione assente;
- colonne duplicate, colonne obbligatorie mancanti;
- `record_id` mancante o ripetuto;
- valori non numerici o non finiti (`nan`, `inf`);
- `target` diverso da 0/1 e `split` diverso da train/validation;
- target o `record_id` presenti tra le feature (leakage).

Il lettore restituisce anche fingerprint dei byte, conteggio righe e classi per
split: sono le informazioni che servono a training e valutazione per legare ogni
risultato ai dati usati. Il profilo prodotto e' in
`evidence/data/profilo_dati.json`.

### 3.4 Perche' gli input delle richieste non sono training

Il CSV delle richieste contiene solo `features` (o un elenco incompleto), non
`target`, `record_id` o `split`. Non e' una fonte di label: serve a provare il
contratto del servizio. Le 30 feature sono tutte misure del periodo precedente
al turno: nessuna diventa disponibile dopo l'evento da prevedere.

## 4. Baseline v1 (S2)

### 4.1 Scelta del modello

La baseline e' una pipeline `StandardScaler` + `LogisticRegression`
(`C=1.0`, `max_iter=1000`). E' semplice da spiegare, veloce e adatta a feature
numeriche con scale diverse. Lo scaler sta **dentro** la pipeline: si adatta sul
solo train e viene salvato con il modello, quindi il servizio applica la stessa
preparazione usata in training.

### 4.2 Protocollo

- fit sul solo train (800 righe), valutazione sulla validation (200 righe);
- nessuna trasformazione addestrata sulla validation;
- metrica principale del protocollo: accuracy; metrica del soggetto: recall
  classe 1; lettura degli errori con la matrice di confusione;
- run MLflow con parametri, metriche, tag (fingerprint dataset e schema) e
  artefatto; metadati in `models/v1/model_metadata.json`.

### 4.3 Risultati

| Metrica | Valore |
| --- | ---: |
| Accuracy | 0,665 |
| Recall classe 1 | 0,347 |
| Precision classe 1 | 0,591 |
| F1 classe 1 | 0,437 |
| ROC AUC | 0,718 |
| Matrice (TN, FP; FN, TP) | (107, 18; 49, 26) |

La matrice spiega il problema: su 75 guasti reali, **49 non vengono previsti**.
La baseline e' prudente: 18 falsi positivi su 125 macchinari sani. Per un uso
operativo la copertura dei guasti e' troppo bassa, anche se l'accuratezza non e'
pessima: e' l'argomento dell'ipotesi per la candidata.

Run MLflow: `8707d4c4859f423c86a0ba9b007a8661`.
Fingerprint artefatto:
`6eea94e288ab41dbfa515a61efcce53507aecb71c9b56c5f564b2cd964eb9dc0`.

## 5. Servizio e telemetria (S3)

### 5.1 Architettura

Il servizio e' una applicazione FastAPI (`src/manutenzione/serving/app.py`).

- All'avvio legge `models/active_version.json`, verifica l'artefatto e carica in
  memoria pipeline, metadati e soglia di decisione.
- `POST /predict` valida il payload e restituisce `prediction`,
  `probability_target_1`, `decision_threshold`, `model_version`, `run_id` e
  `request_id`.
- `GET /health` restituisce versione, run ID, fingerprint dell'artefatto, numero
  di feature, soglia e origine della selezione (`active_version` o versione
  forzata per una prova).
- `GET /metrics` restituisce conteggio, errori, error rate, p50 e p95 della
  finestra, con separazione per versione.
- `GET /versions` e `GET /schema` espongono registro e contratto.
- `POST /admin/activate` e `POST /admin/rollback` cambiano la versione servita
  solo dopo verifica dell'artefatto, con token amministrativo.

### 5.2 Contratto e validazione

La richiesta e' `{"features": [30 numeri]}`. Il modello Pydantic e' costruito
dall'ordine dello schema:

- lunghezza esatta 30: una lista di 29 produce **422** (caso `req_04`);
- valori non finiti → 422;
- campi extra → 422 (`extra="forbid"`);
- feature di conteggio non intere → 422.

Il 422 viene prodotto prima della chiamata al modello. Il gestore degli errori
di validazione restituisce solo posizione, messaggio e tipo dell'errore: non
rimanda indietro i valori ricevuti, che potrebbero essere `NaN` o dati sensibili.

### 5.3 Log e metriche

Ogni richiesta genera una riga JSON in `logs/serving.jsonl` con timestamp,
request ID, metodo, path, stato HTTP, durata, versione, run ID ed esito. **Il
payload non viene registrato.** Lo stesso request ID compare nella risposta e
nel log: e' la correlazione usata anche nel drill.

Le metriche hanno una finestra esplicita (15 minuti da configurazione) e
riguardano solo `POST /predict`: `/health`, `/metrics`, `/versions` e gli
endpoint amministrativi non entrano nel conteggio. Errori = stato ≥ 400. Il p95
e' calcolato con `numpy.percentile` (interpolazione lineare) sulle durate della
finestra. La definizione e' riportata nella risposta stessa di `/metrics`.

### 5.4 Prove eseguite

| Prova | Cosa dimostra | Evidenza |
| --- | --- | --- |
| Client in-process | contratto, 422, identita' versione, request ID nei log | `evidence/requests/client_inprocess.json` |
| HTTP reale (probe) | latenza misurata da fuori processo su 3 warm-up + 30 richieste | `evidence/serving/probe_v1.json`, `probe_v2.json` |
| Container Docker | stessa sequenza dentro l'immagine, con log del container | `evidence/serving/container_prova.json` |

Nella prova in-process le quattro richieste assegnate producono 200, 200, 200 e
422; l'error rate della finestra e' 0,25 e tutti i request ID risultano nei log.
La prova in-process non dimostra nulla su rete e packaging: per questo HTTP e
container sono prove separate.

## 6. Candidata v2 (S4)

### 6.1 Ipotesi

La baseline lineare non rappresenta le combinazioni tra temperatura, vibrazione,
carico e usura: un macchinario puo' essere a rischio per l'interazione tra piu'
segnali, non per uno solo. La candidata e' una `RandomForestClassifier` con 300
alberi, profondita' massima 8, foglie minime di 4 e pesi di classe bilanciati,
allenata sullo stesso train.

La modifica include la **soglia di decisione**: invece di usare 0,5, la soglia
viene scelta con validazione incrociata a 5 fold sul solo train, prendendo il
valore con accuratezza out-of-fold massima tra quelli con recall ≥ 0,65. La
validation non partecipa a questa scelta. La soglia selezionata e' **0,46**
(accuratezza out-of-fold 0,680, recall out-of-fold 0,669).

### 6.2 Confronto sulla stessa validation

| Metrica | v1 | v2 |
| --- | ---: | ---: |
| Soglia | 0,50 | 0,46 |
| Accuracy | 0,665 | **0,710** |
| Recall classe 1 | 0,347 | **0,653** |
| Precision classe 1 | 0,591 | 0,605 |
| ROC AUC | 0,718 | **0,757** |
| Falsi positivi | 18 | 32 |
| Falsi negativi | 49 | **26** |

La candidata riduce i guasti mancati da 49 a 26, al prezzo di 14 falsi positivi
in piu'. Per il problema descritto e' lo scambio desiderato, ma il margine sul
recall e' stretto: 0,653 contro la soglia di 0,65. Un secondo campione potrebbe
ribaltare l'esito del gate.

### 6.3 Spiegazioni SHAP

Su 200 righe di validation, le feature con importanza media assoluta piu' alta
sono `ore_marcia_ultime_24h`, `carico_medio_pct`, `avvii_ultime_24h`,
`ore_utilizzo_totali` e `manutenzioni_programmate_completate_pct`: il modello
usa soprattutto intensita' e storia d'uso, non solo la temperatura. La
spiegazione della richiesta `req_02` e' in
`evidence/explanations/shap_summary.md`.

SHAP descrive come il modello combina le feature, non un rapporto causale. Con
un dataset sintetico le direzioni medie non vanno lette come fisica del
macchinario.

## 7. Gate di rilascio (S4)

Il gate e' in `evidence/gate/gate_report.md`; ogni requisito ha misura, soglia,
esito e motivazione.

| ID | Requisito | Misura | Esito |
| --- | --- | --- | --- |
| R1 | Accuracy v2 ≥ v1 | 0,710 vs 0,665 | superato |
| R2 | Recall classe 1 ≥ 0,65 | 0,653 | superato |
| R3 | Recall v2 ≥ v1 | 0,653 vs 0,347 | superato |
| R4 | p95 probe v2 ≤ 120 ms | 57,87 ms | superato |
| R5 | Error rate probe v2 ≤ 0,01 | 0,000 | superato |
| R6 | Probe attribuita all'artefatto v2 | fingerprint coincidente | superato |
| R7 | Stesso dataset, stessi ID di validation, stesso schema | tutte vere | superato |

**Decisione: accettata.** Un miglioramento di accuratezza non avrebbe compensato
una violazione della latenza o un peggioramento del recall; qui tutti i
requisiti sono soddisfatti. La decisione del gate **non** cambia la versione
attiva: l'attivazione e' un'operazione separata e registrata.

## 8. Attivazione e ripristino (S5)

### 8.1 Sequenza osservabile

Il drill parte con v1 attiva e usa l'API reale su un processo separato:

| Passo | Esito | Versione servita | Run ID |
| --- | ---: | --- | --- |
| predizione con v1 su `req_01` | 200 | v1 | `8707d4c4...` |
| attivazione v2 via `/admin/activate` | 200 | v2 | `474c3742...` |
| predizione con v2 su `req_02` | 200 | v2 | `474c3742...` |
| rollback via `/admin/rollback` | 200 | v1 | `8707d4c4...` |
| predizione con v1 su `req_03` | 200 | v1 | `8707d4c4...` |
| richiesta incompleta `req_04` | 422 | - | - |

Ogni transizione registra azione, attore, stato prima/dopo, run ID, fingerprint
dell'artefatto, fingerprint del dataset e motivo, sia in
`models/active_version.json` (stato e storico) sia in
`models/transitions.jsonl` (registro append-only).

### 8.2 Come si dimostra che il modello e' cambiato

Non basta stampare "rollback". Le prove raccolte sono:

- il body della risposta di `activate` riporta `served_version = v2` e il run ID
  di v2;
- la predizione successiva riporta `model_version = v2` e lo stesso run ID;
- dopo il rollback la predizione riporta v1 e il run ID di v1;
- `/health` finale riporta versione v1 e fingerprint
  `6eea94e288ab41dbfa515a61efcce53507aecb71c9b56c5f564b2cd964eb9dc0`, uguale
  ai metadati di v1;
- il puntatore finale ha `active = v1`, `previous = v2`.

Le quattro predizioni del drill hanno request ID presenti nel log e un error
rate di **0,25** (1 su 4). E' una misura del drill e resta separata dalle
misure della probe, dove l'error rate e' 0,00.

### 8.3 Gate e drill sono cose diverse

- Il **gate** decide se la candidata puo' essere usata; non tocca il servizio.
- L'**attivazione** e' un'operazione esplicita, con token, che verifica
  l'artefatto e ricarica il modello.
- Il **ripristino** e' un'altra operazione registrata: il drill lo esegue per
  lasciare lo stato finale sulla baseline richiesta dalla consegna, dopo aver
  documentato l'attivazione di v2.

## 9. MLflow, versioning e riproducibilita'

- MLflow 3 usa un backend SQLite locale (`mlflow.db`) e la cartella `mlruns/`
  per gli artefatti; ogni versione ha una run con parametri, metriche, tag e
  copia dell'artefatto.
- I metadati (`model_metadata.json`) contengono versione, run ID, fingerprint di
  dataset e artefatto, ordine delle feature, fingerprint dello schema, metriche,
  soglia, configurazione e versioni software.
- Il caricatore ricalcola il fingerprint dei byte prima di usare l'artefatto e
  rifiuta copie non corrispondenti o schemi diversi. I test verificano anche il
  caso di artefatto manomesso.
- `models/v1` e `models/v2` non vengono sovrascritti: una nuova versione usa una
  cartella nuova.
- Le evidenze essenziali restano in `evidence/` anche se `mlruns/` viene
  rigenerato.

Tutto il progetto e' in Git con branch separati per macro-area (dati/training,
servizio, valutazione/rilascio, documentazione), mantenuti dopo il merge. La CI
GitHub Actions installa le dipendenze ed esegue i 25 test a ogni push.

## 10. Monitoraggio, logging e drift

Disponibile subito, senza label:

- conteggio richieste, errori con stato ≥ 400, error rate, p50/p95 della
  finestra, separati per versione;
- log JSONL per correlare risposta e evento tramite request ID;
- health con identita' dell'artefatto servito;
- controllo di plausibilita' delle richieste rispetto al train con z-score
  robusto (`scripts/09_controllo_drift.py`).

Richiede invece le label, disponibili solo dopo 7 giorni:

- accuracy, recall, precision e matrice di confusione reali;
- confronto tra probabilita' previste e frequenza osservata.

Il controllo di drift eseguito sulle tre richieste valide non segnala valori
oltre soglia (z-score massimo 2,60). Con tre payload e' solo una verifica di
plausibilita': in produzione servirebbe una finestra aggregata di statistiche,
non i payload grezzi, che per scelta non vengono salvati.

## 11. Etica, sicurezza e audit

- **Minimizzazione**: i log non contengono payload; le risposte di errore non
  rimandano i valori ricevuti.
- **Permessi**: gli endpoint amministrativi richiedono un token confrontato in
  tempo costante; il default locale va sostituito in ambienti condivisi.
- **Audit**: ogni transizione ha attore, motivo, timestamp, stato prima/dopo, run
  ID e fingerprint.
- **Integrita'**: un artefatto o uno schema non coerenti bloccano il
  caricamento; un'attivazione di versione inesistente restituisce 409.
- **Trasparenza**: metriche e spiegazioni SHAP sono conservate con la versione
  che le ha prodotte.

Limiti di sicurezza dichiarati: `/predict` non richiede autenticazione, non c'e'
TLS e il servizio non e' esposto oltre localhost nelle prove. Sono scelte
accettabili per un esercizio locale, non per la produzione.

## 12. Test, CI/CD e container

- 25 test pytest coprono: controlli dati (file vuoto, ID duplicati, colonne
  duplicate, valori non finiti, split e target inattesi), registro (roundtrip,
  artefatto manomesso, schema cambiato, attivazione/rollback, versione esistente),
  servizio (health, predict, 422, 401, 409, metriche, attivazione, rollback,
  versione forzata).
- CI: `.github/workflows/ci.yml`, Python 3.12, installazione da
  `requirements.txt`, `pip install --no-deps -e .`, `pytest`.
- Docker: immagine `manutenzione-predittiva-api:1.0.0`, avvio con
  `docker run` o `docker compose`; prova reale eseguita con health, predizione
  v1, attivazione v2, predizione v2, rollback, predizione v1 e 422.
- Il container di prova e i processi Uvicorn delle prove sono stati rimossi o
  arrestati.

## 13. Limiti e sviluppi

1. **Validation non indipendente**: e' lo stesso split per entrambe le versioni,
   utile al confronto ma non una prova su dati futuri. Servirebbe una
   validazione temporale su periodi successivi.
2. **Dati sintetici**: i numeri non stimano le prestazioni su un impianto reale.
3. **Margine del recall**: 0,653 contro 0,65 e' fragile; con piu' dati la
   decisione del gate potrebbe cambiare.
4. **Soglia scelta sul train**: corretta rispetto al leakage, ma con 800 righe
   la stima out-of-fold ha incertezza.
5. **Probe non rappresentativa del carico**: 30 richieste sequenziali su una
   istanza locale, non un test di carico concorrente.
6. **Drift illustrativo**: tre payload non bastano per un monitoraggio
   statistico; manca una pipeline di aggregazione delle statistiche.
7. **Nessuna notifica automatica**: il servizio espone metriche e log, ma non
   invia avvisi quando error rate o p95 superano le soglie.
8. **Sicurezza**: token amministrativo semplice, nessun TLS, nessuna
   autenticazione su predict.
9. **Spiegazioni non causali**: SHAP aiuta a leggere il modello, non a
   dimostrare la causa del guasto.

Sviluppi naturali: validazione temporale, monitoraggio del drift su finestre
aggregate, notifiche su soglie, autenticazione del client, test di carico,
retraining periodico con lo stesso protocollo di gate.

## 14. Divisione del lavoro e revisioni

| Persona | Area | Revisione assegnata |
| --- | --- | --- |
| Matteo | Dati, training, baseline e candidata | riletta da Livia |
| Livia | Servizio, telemetria, prove HTTP | riletta da Matteo |
| Andrea | Registro, gate, attivazione/rollback, Docker | riletto da Nicolo |
| Nicolo | README, relazione, presentazione, riepilogo evidenze | riletto da Andrea |

Il dettaglio delle attivita' e delle checklist di revisione e' in un documento
interno del gruppo, non pubblicato. La presentazione segue il percorso
problema → dati → baseline → servizio → candidata → decisione → ripristino →
limiti.

## 15. Evidenze di riferimento

| Domanda | File |
| --- | --- |
| Profilo dati e fingerprint | `evidence/data/profilo_dati.json` |
| Metriche v1 e v2, run MLflow | `evidence/training/v1.json`, `v2.json` |
| Metadati e artefatti | `models/v1/`, `models/v2/` |
| Prova in-process | `evidence/requests/client_inprocess.json` |
| Probe 30 misure | `evidence/serving/probe_v1.json`, `probe_v2.json` |
| Prova container | `evidence/serving/container_prova.json` |
| Gate | `evidence/gate/gate_report.md` |
| Attivazione e rollback | `evidence/transitions/drill_responses.json` |
| Transizioni e stato attivo | `models/transitions.jsonl`, `models/active_version.json` |
| Spiegazioni SHAP | `evidence/explanations/shap_summary.md` |
| Drift illustrativo | `evidence/monitoring/drift_report.md` |
| Tabella unica dei numeri | `evidence/riepilogo_metriche.md` |
