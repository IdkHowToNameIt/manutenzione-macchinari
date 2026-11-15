# Manutenzione predittiva di macchinari

Progetto ModelOps: prevedere quali macchinari richiederanno un intervento per
guasto **entro 7 giorni**, a partire dalle misure di sensori e contatori
disponibili **a fine turno**.

- **target = 1**: guasto con intervento nei 7 giorni successivi.
- **target = 0**: nessun guasto nei 7 giorni successivi.
- **Input del servizio**: 30 feature numeriche, nell'ordine di
  `config/feature_schema.json`. `record_id`, `split` e `target` non entrano mai
  negli input.
- **Dati assegnati**: `data-input/dataset/dataset.csv` (1000 righe: 800 train,
  200 validation) e `data-input/requests/richieste_monitoraggio.csv` (3 richieste
  valide + 1 incompleta).

La proposta iniziale e le risposte alle sei domande della consegna sono in
[docs/proposta.md](docs/proposta.md); la relazione completa in
[docs/relazione.md](docs/relazione.md).

## Struttura del progetto

```
Manutenzione-macchinari/
├── README.md                  questo file
├── requirements.txt           dipendenze con versioni usate
├── pyproject.toml             pacchetto src/ installabile in editable
├── config/                    project.yaml, feature_schema.json
├── data-input/                CSV assegnati (dataset e richieste)
├── consegna/                  consegna e istruzioni originali
├── src/manutenzione/          codice del progetto
│   ├── config.py              caricamento configurazione
│   ├── contracts.py           schema feature, fingerprint, controlli
│   ├── data_loader.py         lettore CSV con validazione
│   ├── registry.py            versioni, metadati, attivazione, rollback
│   ├── training.py            pipeline, metriche, run MLflow, soglia
│   └── serving/               API FastAPI, schemi, telemetria
├── scripts/                   esecuzione passo per passo (01 → 11)
├── tests/                     25 test con pytest
├── models/                    v1/, v2/, active_version.json, transitions.jsonl
├── logs/                      log JSONL del servizio e delle istanze di prova
├── evidence/                  evidenze di dati, training, serving, gate, transizioni
├── docs/                      proposta, relazione, presentazione, divisione del lavoro
└── docker/                    Dockerfile e docker-compose.yml
```

## Ambiente

Versioni effettivamente usate (Python 3.12.16, Windows 11, venv creata con `uv`):

| Pacchetto | Versione |
| --- | --- |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| mlflow | 3.18.0 |
| shap | 0.54.1 |
| fastapi | 0.163.1 |
| pydantic | 2.14.3 |
| uvicorn | 0.54.0 |
| httpx | 0.28.1 |
| pyyaml | 6.0.3 |
| pytest | 9.2.0 |

Python 3.12 e' richiesto: con Python 3.14 alcune dipendenze scientifiche non
hanno ancora wheel compatibili in questo ambiente. Setup da PowerShell nella
cartella del progetto:

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv pip install --python .venv\Scripts\python.exe --no-deps -e .
```

In alternativa, con `venv` e `pip`: `python -m venv .venv`, attivazione, poi
`pip install -r requirements.txt` e `pip install --no-deps -e .`.

## Riproduzione completa

Eseguire gli script in ordine. Tutti leggono i percorsi da
`config/project.yaml` e scrivono le evidenze in `evidence/`.

```powershell
# S1 - controlli e profilo dei dati
.venv\Scripts\python.exe scripts\01_verifica_dati.py

# S2 - baseline v1, run MLflow, prima attivazione
.venv\Scripts\python.exe scripts\02_training_baseline_v1.py

# S3 - prova del contratto con client in-process
.venv\Scripts\python.exe scripts\03_client_inprocess.py

# S4 - candidata v2, probe locale, gate di rilascio
.venv\Scripts\python.exe scripts\04_training_candidata_v2.py
.venv\Scripts\python.exe scripts\05_probe_servizio.py
.venv\Scripts\python.exe scripts\06_gate_rilascio.py

# S5 - attivazione v2, rollback a v1, evidenze della sequenza
.venv\Scripts\python.exe scripts\07_attivazione_rollback.py

# Analisi e sintesi
.venv\Scripts\python.exe scripts\08_spiegazioni_shap.py
.venv\Scripts\python.exe scripts\09_controllo_drift.py
.venv\Scripts\python.exe scripts\10_riepilogo_evidenze.py

# Test automatici
.venv\Scripts\python.exe -m pytest
```

Note di riproducibilita':

- `models/v1` e `models/v2` non vengono sovrascritti: se le cartelle esistono,
  gli script di training si fermano con un messaggio esplicito. Per rifare tutto
  da zero spostare (non cancellare senza motivo) `models/`, `mlflow.db` e
  `mlruns/`, poi rilanciare dalla 02.
- MLflow usa un backend SQLite locale (`mlflow.db`) e la cartella `mlruns/` per
  gli artefatti delle run; `evidence/training/*.json` conserva run ID, metriche
  e fingerprint anche senza la UI.
- I percorsi dei CSV sono solo in `config/project.yaml`: chi riceve il README
  trova gli stessi file in `data-input/`.
- Le prove via HTTP e via container sono separate dalla prova in-process:
  `evidence/requests/client_inprocess.json`, `evidence/serving/probe_v*.json`,
  `evidence/serving/container_prova.json`.

## Servizio

Avvio locale (porta 8000, host da configurazione):

```powershell
.venv\Scripts\python.exe -m manutenzione.serving.app
```

Documentazione interattiva su `http://127.0.0.1:8000/docs`.

| Metodo | Percorso | Funzione |
| --- | --- | --- |
| GET | `/health` | stato, versione servita, run ID, fingerprint artefatto |
| POST | `/predict` | predizione con `{"features": [30 valori]}` |
| GET | `/metrics` | conteggio, errori, error rate e p95 della finestra |
| GET | `/versions` | versioni registrate, attiva e precedente |
| GET | `/schema` | schema atteso (nomi e ordine delle 30 feature) |
| POST | `/admin/activate` | attiva una versione verificata (token richiesto) |
| POST | `/admin/rollback` | ripristina la versione precedente (token richiesto) |

La risposta di `predict` contiene `prediction`, `probability_target_1`,
`decision_threshold`, `model_version`, `run_id` e `request_id`; lo stesso
`request_id` compare nell'header `X-Request-ID` e nella riga di log.

Esempio con la prima richiesta del CSV:

```powershell
.venv\Scripts\python.exe -c "import csv,json,httpx; row=next(csv.DictReader(open('data-input/requests/richieste_monitoraggio.csv',encoding='utf-8'))); print(httpx.post('http://127.0.0.1:8000/predict', json=json.loads(row['payload_json']), timeout=10).json())"
```

Amministrazione e sicurezza:

- gli endpoint `/admin/*` richiedono l'header `X-Admin-Token`, confrontato in
  tempo costante con `MANUTENZIONE_ADMIN_TOKEN` (default locale
  `local-admin-token`, da cambiare in ambienti condivisi);
- i log contengono timestamp, request ID, path, stato, durata, versione e run ID,
  **mai il payload grezzo**;
- una lista `features` con 29 valori, un valore non finito o un campo extra
  producono **422** prima di chiamare il modello;
- una versione con fingerprint o schema non coerenti non viene caricata e
  l'attivazione viene rifiutata con un motivo esplicito.

## Risultati

Validation (200 righe, stesse per entrambe le versioni):

| Versione | Modello | Soglia | Accuracy | Recall classe 1 | Precision classe 1 | ROC AUC | FP / FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| v1 | LogisticRegression + StandardScaler | 0.50 | 0.665 | 0.347 | 0.591 | 0.718 | 18 / 49 |
| v2 | RandomForest (300 alberi) | 0.46 | 0.710 | 0.653 | 0.605 | 0.757 | 32 / 26 |

Probe locale su istanza HTTP separata (3 warm-up + 30 misure, stesso payload
valido per versione):

| Versione | p50 | p95 | Error rate | Attribuzione all'artefatto |
| --- | ---: | ---: | ---: | --- |
| v1 | 1.74 ms | 2.86 ms | 0.00 | si |
| v2 | 39.58 ms | 57.87 ms | 0.00 | si |

Gate: **accettata** (R1-R7 superati, dettaglio in
`evidence/gate/gate_report.md`). Il gate non cambia la versione attiva.

Sequenza dimostrata (stato finale sulla baseline, come richiesto):

```
v1 → 200 (req_01, prob 0.530) → attivazione v2 → 200 (req_02, prob 0.530)
   → rollback v1 → 200 (req_03, prob 0.174) → req_04 (29 feature) → 422
```

Error rate del drill: **0,25** (una richiesta respinta su quattro), separato
dalle misure della probe. Transizioni in `models/transitions.jsonl`, evidenze in
`evidence/transitions/`.

Riepilogo sempre aggiornato: `evidence/riepilogo_metriche.md`.

## Evidenze

| Cartella | Contenuto |
| --- | --- |
| `evidence/data/` | profilo dati, fingerprint, classi per split, controllo richieste |
| `evidence/training/` | metriche e run ID di v1 e v2 |
| `evidence/requests/` | prova del contratto con client in-process |
| `evidence/serving/` | probe v1/v2, prova container, log delle istanze |
| `evidence/gate/` | tabella requisiti, esito e confronto |
| `evidence/transitions/` | passi del drill, log correlati, transizioni, stato finale |
| `evidence/explanations/` | importanza SHAP e spiegazione della richiesta v2 |
| `evidence/monitoring/` | controllo di drift illustrativo sulle richieste |
| `models/` | artefatti e metadati per versione, stato attivo, registro transizioni |

## Test e CI

```powershell
.venv\Scripts\python.exe -m pytest
```

25 test coprono lettore e controlli dati, registro e fingerprint, contratto
HTTP (200, 422, 401, 409), metriche, attivazione e rollback. La pipeline
GitHub Actions in `.github/workflows/ci.yml` ripete installazione e test a ogni
push.

## Docker

```powershell
docker build -f docker/Dockerfile -t manutenzione-predittiva-api:1.0.0 .
docker run --rm -p 8000:8000 manutenzione-predittiva-api:1.0.0
# oppure
docker compose -f docker/docker-compose.yml up --build
```

Prova realmente eseguita il 2026-11-15 con Docker 29.5.3 (container Linux):
health, predizione v1, attivazione v2, predizione v2, rollback, predizione v1 e
richiesta incompleta 422. Evidenza: `evidence/serving/container_prova.json` e
`logs/container_prova.log`. Il container di prova viene rimosso dallo script
`scripts/11_verifica_container.py`.

## Stato finale e cleanup

- Versione attiva: **v1**; versione precedente: **v2**; transizioni registrate:
  attivazione iniziale, attivazione v2, rollback a v1.
- Processi e container avviati durante le prove sono stati arrestati; nessun
  servizio resta in ascolto.
- Restano nel repository: CSV assegnati, sorgenti, dipendenze con versioni,
  modelli con metadati, log, evidenze e documentazione.
- `mlruns/` e `mlflow.db` sono esclusi dal versionamento ma possono essere
  rigenerati con gli script; le evidenze essenziali delle run sono in
  `evidence/training/`.

## Limiti

- La validation non e' un test indipendente: e' un confronto didattico tra v1 e
  v2 sullo stesso split, non una validazione temporale su dati futuri reali.
- Il dataset e' sintetico (seed 1302, generato dal docente): i risultati non
  stimano le prestazioni su un impianto reale.
- Il margine di v2 sul recall e' stretto (0.653 contro soglia 0.65) e va
  ricontrollato con piu' dati prima di un uso operativo.
- La probe misura 30 richieste sequenziali su una sola istanza locale: non
  rappresenta il carico reale del servizio.
- Il controllo di drift sulle tre richieste valide e' illustrativo; in
  produzione serve una finestra aggregata di traffico.
- SHAP descrive il modello, non la causalita' del guasto.

## Gruppo

Matteo (dati e training), Livia (servizio e telemetria), Andrea (registro e
rilascio), Nicolo (documentazione e verifica). La divisione dei compiti e le
revisioni incrociate sono descritte nella documentazione interna del gruppo,
non inclusa nel repository.
