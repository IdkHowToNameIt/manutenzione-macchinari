# Controllo di drift sulle richieste assegnate

Riferimento: train, 800 righe, fingerprint `af700d09fac13f09e5b0b8f8713cc0c9d3dca433a8f53c50273fe4647d1c5f3d`.

Soglia: |z-score robusto| > 3.0. Solo tre payload validi: il controllo segnala valori lontani dalla distribuzione di training, non un drift statistico consolidato.

| Richiesta | Versione | Feature oltre soglia |
| --- | --- | --- |
| req_01 | v1 | nessuna |
| req_02 | v2 | nessuna |
| req_03 | v1 | nessuna |

## Come usarlo in produzione

Il servizio non salva i payload nei log: un controllo di drift reale deve leggere una finestra aggregata di statistiche (per esempio medie orarie per feature) prodotta a monte, oppure un campione autorizzato e anonimizzato. Questo script verifica soltanto che le richieste assegnate siano plausibili rispetto al train.
