# Bitcoin AI Analyst

Sistem Multi-Agent AI untuk analisis Bitcoin dengan output keputusan
**Buy / Hold / Sell + Confidence Score**, berbasis data Technical, On-chain,
Macro, Sentiment, dan Liquidity.

## Arsitektur

```
Data Collection (10+ API) ──► TimescaleDB Warehouse ──► Feature Engineering
        │                                                      │
        ▼                                                      ▼
  ccxt / fredapi /                                   Sub-Agents (Qwen AI):
  pandas-datareader /                                Technical · On-chain ·
  httpx                                              Macro · News · Sentiment · Risk
                                                               │
                                                               ▼
                                                   Master Decision Agent
                                                   (Buy/Hold/Sell + Confidence)
```

## Quickstart (Phase 1)

Prasyarat: Python 3.11+, dan salah satu dari:
- **PostgreSQL + TimescaleDB native** (setup aktif saat ini: PostgreSQL 18 + TimescaleDB 2.28.1), atau
- **Docker Desktop** (alternatif; gunakan `docker compose up -d`).

```powershell
# 1) Salin konfigurasi environment lalu isi kredensial Anda
copy .env.example .env

# 2a) [Native] Bootstrap role + database + ekstensi (sekali, sebagai superuser)
#     Ganti dulu password 'change_me' di scripts/setup_database.sql agar sama dengan .env
psql -U postgres -h localhost -f scripts\setup_database.sql

# 2b) [Alternatif Docker] docker compose up -d

# 3) Buat virtual environment & install dependencies
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# 4) Inisialisasi skema database (tabel + hypertables) — idempotent
.venv\Scripts\python scripts\init_db.py

# 5) Verifikasi kesehatan database (hypertables + ORM round-trip)
.venv\Scripts\python scripts\check_db.py
```

## ⭐ PENGGUNAAN HARIAN — Satu Perintah

```powershell
.venv\Scripts\python scripts\run_daily.py
```

Perintah utama proyek ini: menjalankan **seluruh pipeline data** (6 fetcher +
Feature Engineering) lalu **inference AI** (5 sub-agent + Master Agent),
menyimpan keputusan ke tabel `decisions`, dan mencetak laporan lengkap +
ringkasan gabungan. Aturan gate: bila pipeline gagal total (mis. database
mati) inference dibatalkan; bila hanya sebagian fetcher gagal, inference
tetap berjalan dengan fitur terbaru yang ada di database.

Melihat riwayat keputusan (psql / pgAdmin, database `bitcoin_ai`):

```sql
-- 5 keputusan terakhir
SELECT timestamp::date AS tanggal, action, confidence_score,
       executive_summary
FROM decisions ORDER BY timestamp DESC LIMIT 5;

-- Detail panel sub-agent untuk satu hari
SELECT jsonb_pretty(sub_agents_json) FROM decisions
WHERE timestamp::date = '2026-07-10';
```

Untuk otomatisasi penuh, jadwalkan `run_daily.py` (bukan `run_collectors.py`)
di Windows Task Scheduler — lihat bagian penjadwalan di bawah.

## Troubleshooting: `permission denied for table ...`

Bila `run_daily.py` berhenti dengan **HAK AKSES DATABASE BERMASALAH**:

```powershell
psql -U postgres -h localhost -d bitcoin_ai -f scripts\fix_permissions.sql
.venv\Scripts\python scripts\check_db.py
```

Penyebabnya: `setup_database.sql` hanya memberi GRANT pada **skema**, yang
mengizinkan `btc_analyst` membuat objek baru tapi tidak memberi hak atas
tabel yang sudah dibuat role lain. Kalau ada tabel yang dimiliki `postgres`,
`btc_analyst` tidak bisa mengaksesnya sama sekali.

`fix_permissions.sql` memberi GRANT eksplisit, **memindahkan kepemilikan** ke
`btc_analyst` supaya tidak terulang, dan menyetel `ALTER DEFAULT PRIVILEGES`.
Perlu password superuser `postgres` (yang dipakai saat setup awal).

## Menjalankan Pipeline Data (Phase 2)

Satu perintah menjalankan SEMUA fetcher dengan error isolation (satu fetcher
gagal → sisanya tetap jalan) dan ringkasan eksekusi di akhir:

```powershell
# Run harian (default): jendela lookback kecil, cepat & ramah rate-limit
.venv\Scripts\python scripts\run_collectors.py

# Backfill historis penuh (run pertama / pemulihan setelah lama mati)
.venv\Scripts\python scripts\run_collectors.py --mode full
```

Model konkurensi: Binance Spot, FRED, Alternative.me, Binance Vision, dan
Stooq/FRED berjalan **konkuren** (`asyncio.gather`); Blockchain.com berjalan
**sekuensial setelahnya, sendirian** (rate limit ketat). Semua fetcher
melakukan **upsert idempoten** — dijalankan berulang tidak pernah membuat
duplikat. Exit code: `0` bila tidak ada FAILED (SKIPPED tidak fatal), `1`
bila ada — cocok untuk monitoring scheduler.

### Penjadwalan Otomatis (Windows Task Scheduler, setiap hari 01:00)

Opsi cepat via PowerShell (jalankan sekali, sesuaikan path bila proyek pindah):

```powershell
schtasks /Create /TN "BitcoinAI-Daily" /SC DAILY /ST 01:00 /TR "\"C:\Users\syafi\OneDrive\Documents\my project\2.0\Equity and Crypto Research\Bitcoin\Bitcoin Paper\bitcoin-ai-analyst\.venv\Scripts\python.exe\" \"C:\Users\syafi\OneDrive\Documents\my project\2.0\Equity and Crypto Research\Bitcoin\Bitcoin Paper\bitcoin-ai-analyst\scripts\run_daily.py\" --mode incremental"
```

Atau via GUI: buka **Task Scheduler** → *Create Basic Task* →
Trigger **Daily 01:00** → Action *Start a program*:
- **Program/script**: `...\bitcoin-ai-analyst\.venv\Scripts\python.exe`
- **Add arguments**: `"...\bitcoin-ai-analyst\scripts\run_daily.py" --mode incremental`

(`run_daily.py` = pipeline data + keputusan AI tersimpan ke `decisions`.
Gunakan `run_collectors.py` bila hanya ingin refresh data tanpa inference.)

Catatan teknis: file `.env` dimuat dengan **path absolut** relatif ke root
proyek (lihat `src/config.py`), jadi task tetap bekerja meskipun Task
Scheduler memakai working directory yang berbeda. Pastikan service
`postgresql-x64-18` berjalan (default: otomatis saat boot).

## Struktur Proyek

```
bitcoin-ai-analyst/
├── docker-compose.yml      # TimescaleDB (PostgreSQL 16) lokal
├── requirements.txt
├── .env.example            # Template environment variables
├── scripts/
│   └── init_db.py          # Buat tabel + konversi ke hypertable (idempotent)
└── src/
    ├── config.py           # Settings via pydantic-settings (.env)
    ├── logging_config.py   # Logging terpusat
    ├── db/
    │   ├── engine.py       # Async engine + session_scope
    │   └── models.py       # SQLAlchemy 2.0 models (5 tabel time-series)
    └── agents/
        └── base_agent.py   # Base Qwen client (strict JSON + Pydantic + tenacity)
```

## Tabel Data Warehouse

| Tabel                 | Isi                              | Sumber aktif                                  |
|-----------------------|----------------------------------|-----------------------------------------------|
| `crypto_prices`       | OHLCV candle harian **multi-aset** | ccxt — Binance (via data-api.binance.vision)  |
| `macro_indicators`    | DXY (DTWEXBGS), 10Y yield (DGS10)| fredapi (FRED)                                |
| `on_chain_metrics`    | hash-rate, n-transactions, supply| Blockchain.com charts API (httpx)             |
| `sentiment_data`      | Fear & Greed Index harian        | Alternative.me (httpx)                        |
| `derivatives_metrics` | funding rate, open interest      | Arsip Binance Vision (Deribit ISP-blocked)    |
| `market_indices`      | S&P 500 & Nasdaq 100 close       | Stooq (fallback otomatis: FRED SP500/NASDAQ100)|
| `daily_features`      | 49 fitur harian per (hari, simbol) | `FeatureEngine` (pandas/numpy/`ta` — bukan LLM) |
| `decisions`           | Keputusan AI per (hari, simbol)  | `run_inference.py` / `run_daily.py` (upsert per hari+aset) |
| `coins_registry`      | "Coin DNA": sektor, chain, base_asset, coingecko_id | seed `migrate_phase8.py` + `seed_sectors_phase8.py` |
| `market_regimes`      | Rezim pasar harian + bukti lengkapnya | `RegimeDetector` (rule-based, bukan LLM) |
| `sector_performance`  | Rotasi modal per sektor (rs vs BTC) | `SectorTracker` (rule-based, bukan LLM) |
| `supply_metrics`      | Circulating & total supply harian | `SupplyFetcher` (CoinGecko, 1 request) |
| `tokenomics_events`   | Unlock/burn/halving terkurasi    | `seed_tokenomics.py` (manual, tervalidasi) |

Tabel time-series adalah **TimescaleDB hypertable** dengan `timestamp` sebagai
kolom partisi (bagian dari composite primary key). `coins_registry` dan
`tokenomics_events` sengaja **tabel biasa** — keduanya data dimensi, bukan
deret waktu. Detail kendala jaringan (host yang diblokir ISP dan solusinya)
ada di `CLAUDE.md`.

## Feature Engineering (Phase 3)

```powershell
# Hitung ulang semua fitur dari warehouse -> tabel daily_features (idempoten)
.venv\Scripts\python scripts\test_features_phase3.py
```

`src/features/engine.py` membaca 6 tabel sumber, menyelaraskan ke spine
harian (tanggal candle aset ybs.), melakukan **forward-fill terbatas** (default 7
hari; `funding_rate` 35 hari karena arsip bulanan, `dxy_value` 14 hari karena
lag rilis mingguan FRED H.10), lalu menghitung 37 fitur: Technical (RSI,
MACD, Bollinger, ATR, realized vol via library `ta` — `pandas-ta` tidak
kompatibel Python 3.14), Macro RoC, MA on-chain, RoC sentimen, rasio
OI/Volume + perubahan funding, RoC indeks saham, dan **relative strength vs
BTC** (`price_btc_ratio`, `rs_7d`, `rs_30d`). Agent Phase 4 cukup:
`SELECT * FROM daily_features WHERE timestamp = :day AND symbol = 'BTC/USDT'`.

> **Wajib sejak Phase 8:** setiap query ke `daily_features` HARUS memfilter
> `symbol`. Tabel kini berisi satu baris per (hari, aset), sehingga
> `ORDER BY timestamp DESC LIMIT 1` tanpa filter akan mengembalikan altcoin
> sembarang.

## Multi-Asset / Altcoin (Phase 8 Part 1)

```powershell
# Migrasi skema + seed coins_registry (idempoten; --dry-run untuk pratinjau)
.venv\Scripts\python scripts\migrate_phase8.py

# Uji fondasi multi-aset (7 gate: registry, hierarki, keying, RS, no-gap)
.venv\Scripts\python scripts\test_altcoin_phase8.py
```

`coins_registry` adalah sumber kebenaran untuk simbol mana yang di-ingest.
`BinanceMultiFetcher` membacanya dan menerapkan **hierarchical ingestion**:
`BTC/USDT` → `ETH/USDT` → altcoin (alfabetis), berurutan. BTC selalu
di-ingest walau dinonaktifkan di registry — ia penyebut setiap fitur relatif
dan Phase 1-7 bergantung padanya.

**Kebijakan kegagalan: strict + fallback terbatas.** Simbol *load-bearing*
(yang membagi fitur simbol lain — `BTC/USDT`, plus `base_asset` mana pun yang
dideklarasikan koin aktif) tidak boleh gagal diam-diam:

| Kondisi | Perilaku |
|---|---|
| Fetch live BTC sukses | normal |
| Fetch live BTC gagal, candle DB ≤ 1 hari | WARNING + lanjut pakai candle tersimpan (`STALE FALLBACK` di ringkasan) |
| Fetch live BTC gagal, candle DB > 1 hari / tidak ada | **`FetcherError` — abort total** |
| Fetch ETH/altcoin gagal | WARNING + lanjut (tidak ada yang dibagi olehnya) |

Atur via `abort_if_head_fails` dan `base_fallback_max_age_days` di
`BinanceMultiFetcher`.

Kolom baru di `daily_features`: `symbol` (bagian dari PK), `asset_close` /
`asset_volume` (harga aset baris itu sendiri), serta `price_btc_ratio`,
`rs_7d`, `rs_30d`. `btc_close`/`btc_volume` tetap **selalu harga Bitcoin** di
setiap baris — itulah sebabnya baris BTC lama tidak berubah sama sekali.
Menambah koin: `INSERT` ke `coins_registry`, lalu jalankan
`run_collectors.py`.

## Regime Detection & Dynamic Weighting (Phase 8 Part 2)

```powershell
# Uji deteksi rezim + bobot dinamis (7 gate, termasuk Bull 2021 / Bear 2022)
.venv\Scripts\python scripts\test_regime_phase8.py

# Perlu sekali saja: backfill riwayat dalam agar 2021-2022 tersedia
.venv\Scripts\python scripts\backfill_history.py --since 2020-01-01
```

`src/features/regime_detector.py` mengklasifikasikan pasar setiap hari ke
salah satu dari lima rezim (plus `UNDETERMINED` bila tidak ada yang jelas)
dan menyimpannya ke hypertable `market_regimes` **beserta seluruh bukti**
(SMA50/200, rasio volume, F&G, MACD, volatilitas vs baseline 1 tahun,
drawdown 90 hari, RSI). Rule-based, bukan hasil belajar — label yang
mengubah bobot keputusan nyata harus bisa dijelaskan dan direproduksi.

`src/agents/weighting_engine.py` memetakan rezim → bobot pilar:

| Rezim | technical | macro | onchain | sentiment | liquidity_risk |
|---|---:|---:|---:|---:|---:|
| BULL_TRENDING | **35%** | 20% | 15% | 15% | 15% |
| BEAR_TRENDING | 25% | **30%** | 15% | 10% | 20% |
| SIDEWAYS_LOW_VOL | 20% | **25%** | **25%** | 15% | 15% |
| SIDEWAYS_HIGH_VOL | 20% | 15% | 10% | 25% | **30%** |
| CRASH_RECOVERY | 15% | 10% | 15% | **30%** | **30%** |
| UNDETERMINED / fallback | 20% | 20% | 20% | 20% | 20% |

Fallback ke bobot setara terjadi bila rezim tidak terdeteksi, tidak dikenal,
`UNDETERMINED`, atau confidence < 45. Panel difusikan **secara deterministik
di Python** (`MasterAgent.fuse`), bukan diminta ke LLM — LLM menerima skor
terbobot sebagai anchor lalu tetap memutuskan aksi dan narasinya, dan
menyebut rezim + bobot di ringkasan eksekutif.

### Transition damping

Rezim yang **baru muncul hari ini** confidence-nya didiskon 30%
(`TRANSITION_CONFIDENCE_FACTOR = 0.7`) sebelum diuji ke ambang. Efeknya:

| Situasi | Ambang confidence mentah |
|---|---|
| Rezim sudah bertahan (sama dengan kemarin) | **45** |
| Rezim baru berubah dari kemarin | **65** |

Batas rezim adalah bagian paling tidak andal dari classifier mana pun —
membalik seluruh bobot panel karena bacaan satu hari mengundang whipsaw.

Tiga kondisi yang **sengaja bukan** transisi:

- **Tidak ada catatan sebelumnya** (run pertama) — tidak ada perpindahan
  yang terjadi, jadi cold start tidak dihukum.
- **Catatan sebelumnya sudah basi** (> `TRANSITION_LOOKBACK_DAYS` = 3 hari,
  mis. job harian sempat mati) — perubahan lintas jurang itu bukan
  transisi segar; rezim barunya sudah punya waktu untuk mapan.
- **Rezim sebelumnya `UNDETERMINED`** — perpindahan dari "tidak yakin" ke
  "yakin" adalah *emergence of clarity*, bukan whipsaw: tidak ada keyakinan
  lama yang dibalik. Hanya transisi antar rezim tegas (BULL/BEAR/SIDEWAYS/
  CRASH) yang didiskon.

```python
engine.get_weights_with_transition(current_regime, previous_regime, confidence)
engine.evaluate(current, previous, confidence)   # + alasan lengkap
```

> Angka bobot di atas diturunkan dari prinsip Krause 2026 yang sudah dipakai
> di system prompt Master Agent. Silakan disesuaikan — semuanya satu baris
> di `REGIME_WEIGHTS`.

## Sector Rotation Tracker (Phase 8 Part 3)

```powershell
# Perluas coins_registry (idempoten; --verify cek simbol ke Binance dulu)
.venv\Scripts\python scripts\seed_sectors_phase8.py --verify

# Uji tracker (7 gate, termasuk paritas pandas vs SQL)
.venv\Scripts\python scripts\test_sector_phase8.py
```

`src/features/sector_tracker.py` mengelompokkan setiap altcoin aktif menurut
`coins_registry.sector`, merata-ratakan `rs_7d`/`rs_30d` per sektor, lalu
menyimpannya ke hypertable `sector_performance`.

Status adalah **2x2** atas kekuatan mingguan vs bulanan:

| Status | `rs_7d` | `rs_30d` | Bacaan |
|---|:---:|:---:|---|
| `IN_FLOW` | > 0 | > 0 | memimpin dan bertahan |
| `ACCUMULATING` | > 0 | < 0 | bulanan lemah, mingguan mulai inflow — **bottoming/reversal** |
| `PULLBACK` | < 0 | > 0 | bulanan kuat, sedang koreksi mingguan |
| `OUT_OF_FLOW` | < 0 | < 0 | tertinggal secara menyeluruh |
| `NEUTRAL` | salah satu tepat nol, atau datanya tidak lengkap | | |

Kuadrannya **saling lepas**, jadi tidak ada urutan pengecekan yang bisa
membuat satu status tak tercapai — beda dengan versi tiga-status awal di
mana `ACCUMULATING` adalah subset ketat `OUT_OF_FLOW`.

**BTC dikecualikan** dari agregasi: `rs_7d`-nya 0.0 secara definisi, jadi
memasukkannya akan menarik semua rata-rata sektor ke nol dan membuat bucket
L1 lemah secara struktural.

**Data hilang ditangani eksplisit.** Rata-rata mengabaikan NULL (seperti SQL
`AVG`), jadi koin yang baru listing punya `rs_7d` tapi belum punya `rs_30d`.
Karena itu `coin_count` dan `coin_count_30d` disimpan **terpisah** — "AI +18%"
terbaca sama saja apakah berasal dari 3 koin atau 1, dan hanya kedua angka
itu yang membedakannya.

Komposisi registry saat ini (20 altcoin + BTC):

| Sektor | Koin |
|---|---|
| L1 | ETH, SOL, BNB, AVAX, ADA |
| L2 | ARB, OP, POL |
| DeFi | UNI, AAVE |
| AI | TAO, FET, RENDER |
| Meme | DOGE, PEPE, WIF |
| RWA | ONDO, POLYX, PENDLE |
| Infrastructure | LINK *(1 koin — ditandai `[THIN]` di mana pun muncul)* |

> `MATIC` dan `RNDR` sudah delisted (menjadi `POL` dan `RENDER`). Jalankan
> `--verify` sebelum seeding agar migrasi ticker berikutnya tidak diam-diam
> membuat sektor kosong.

Seeding memakai `ON CONFLICT DO NOTHING` sehingga **tidak pernah** mengubah
baris yang sudah ada — itu melindungi kurasi manual. Untuk mengubah sektor
koin yang sudah terdaftar, tambahkan ke `SECTOR_CORRECTIONS` lalu jalankan
`--reclassify` (opt-in, eksplisit).

## Tokenomics & Coin DNA (Phase 8 Part 4)

```powershell
# Muat jadwal unlock/burn dari file (validasi all-or-nothing)
.venv\Scripts\python scripts\seed_tokenomics.py --file data\unlocks.json --dry-run
.venv\Scripts\python scripts\seed_tokenomics.py --file data\unlocks.json
.venv\Scripts\python scripts\seed_tokenomics.py --show

# Uji Part 4 (6 gate)
.venv\Scripts\python scripts\test_tokenomics_phase8.py
```

`daily_features` kini punya **49 fitur** (47 numerik + 2 kategorikal):

| Kelompok | Fitur |
|---|---|
| Supply & tokenomics | `days_to_next_unlock`, `supply_shock_score_30d`, `inflation_rate_7d`, `supply_utilization` |
| Volatilitas | `vol_percentile_30d`, `vol_regime` (LOW/NORMAL/HIGH) |
| Likuiditas & struktur | `liquidity_score`, `market_cap_tier` (LARGE/MID/SMALL), `btc_beta_30d`, `btc_corr_30d` |
| Momentum | `momentum_score`, `trend_strength` |

**Beta ≠ korelasi**, jadi keduanya disimpan. Beta = `cov(r_alt, r_btc)/var(r_btc)`
(seberapa *keras* koin bergerak), korelasi = seberapa *seiring* ia bergerak.
Contoh nyata: LINK korelasi 0,87 tapi beta 1,31 — mengikuti BTC hampir
sempurna, tapi 31% lebih kencang.

### Kontrak data yang belum tersedia

| Kondisi | Hasil |
|---|---|
| Tidak ada event tokenomics | `days_to_next_unlock` = **NULL**, `supply_shock_score_30d` = **0** |
| `supply_metrics` belum ada | 4 fitur supply = **NULL** |
| Riwayat < 30 hari | `btc_beta_30d` = **NULL** |

`0` dan `NULL` di baris pertama berbeda dengan sengaja: nol berarti "tidak
ada yang terjadwal" (jawaban nyata), NULL berarti "tidak diketahui".

`supply_metrics` hanya tumbuh **ke depan** — CoinGecko gratis memberi
snapshot, bukan deret historis, jadi fitur turunannya NULL untuk baris lama.
Menginterpolasi mundur berarti mengarang justru besaran yang diukur.

> Jadwal unlock dikurasi manual lewat `seed_tokenomics.py`; tidak ada API
> gratis yang cukup andal. Tabel kosong adalah keadaan yang sah — tabel
> yang diisi tebakan tidak.

## Coin DNA → Agent (Phase 8 Part 5)

```powershell
.venv\Scripts\python scripts\test_agent_integration_phase8.py
```

Fitur DNA kini disuntikkan ke prompt tiap pilar sesuai relevansinya:

| Agent | Konteks DNA yang diterima |
|---|---|
| Technical | `vol_regime`, `vol_percentile_30d`, `btc_beta_30d`, `btc_corr_30d`, `trend_strength` |
| Liquidity Risk | `market_cap_tier`, `liquidity_score`, `supply_shock_score_30d`, `days_to_next_unlock` |
| On-Chain | `supply_utilization`, `inflation_rate_7d` |
| Macro & Sentiment | `market_cap_tier` + sektor saja |
| Master | ringkasan naratif via `format_dna_summary()` |

Pemisahan pilar Phase 4 tetap dijaga: Technical **tidak** menerima konteks
unlock, Liquidity **tidak** menerima beta.

### NULL punya arti berbeda per kolom

`days_to_next_unlock` NULL berarti **"tidak ada unlock terjadwal"** — itu
kabar baik untuk tekanan suplai. `inflation_rate_7d` NULL berarti **"data
belum tersedia"** — benar-benar tidak diketahui. Menyamakan keduanya akan
membuat agent memperlakukan "tidak ada unlock" sebagai lubang data dan
berjaga-jaga terhadap risiko yang tidak ada. Karena itu setiap kolom punya
teks fallback sendiri di `DNA_FALLBACKS`.

Untuk BTC, prompt menambahkan catatan bahwa Beta 1,00 dan korelasi 1,00
adalah **identitas matematis**, bukan temuan — tanpa itu model melaporkannya
sebagai "keselarasan pasar sempurna".

Tanpa `coin_dna`, prompt kembali persis seperti sebelum Part 5.

## Multi-Asset Inference (Phase 8 Part 6)

```powershell
# Semua aset aktif di coins_registry (BTC dulu, lalu alfabetis)
.venv\Scripts\python scripts
un_inference.py --all

# Subset tertentu
.venv\Scripts\python scripts
un_inference.py --symbols BTC/USDT SOL/USDT

# Uji pipeline (7 gate, semua model call di-stub — tanpa token)
.venv\Scripts\python scripts	est_multi_asset_phase8.py
```

`decisions` kini ber-PK `(timestamp, symbol)` — satu baris per (hari, aset).
Sebelumnya PK hanya `timestamp`, sehingga run 21 koin akan saling menimpa dan
menyisakan satu baris saja.

Setiap aset diproses **berurutan** dengan jeda (`--delay`), dan kegagalan
satu aset **tidak** menghentikan yang lain.

### ⚠️ Batas kuota model (penting)

OpenRouter free tier = **50 request per HARI** (bukan per menit).

| | |
|---|---|
| Call per aset | 6 (5 pilar + Master) |
| Registry saat ini | 21 aset = **126 call** |
| Kapasitas free tier | **~8 aset/hari** |

Pipeline mendeteksi kuota harian yang habis lalu **berhenti** — sisa aset
ditandai SKIPPED, bukan FAILED. Tanpa ini setiap aset menghabiskan retry
backoff untuk gagal dengan cara yang sama (terukur: 3+ menit → 8 detik).

Forecasting tetap **BTC-only**: `forecasts` tidak punya kolom symbol, jadi
altcoin sengaja tidak menerima blok forecast — persentil Monte Carlo Bitcoin
berlabel nama altcoin akan menyesatkan.

## AI Reasoning Engine (Phase 4)

```powershell
# Keputusan harian Buy/Hold/Sell dari 5 sub-agent + Master Agent (CIO)
.venv\Scripts\python scripts\run_inference.py
```

Endpoint: **OpenRouter** (`QWEN_BASE_URL`) dengan reasoning model
`nvidia/nemotron-3-ultra-550b-a55b:free` (`QWEN_MODEL`, alias
`QWEN_MODEL_NAME` juga diterima). Setiap panggilan menyertakan
`extra_body={"reasoning":
{"enabled": true}}`; output diparse oleh extractor JSON tahan-banting
(direct → fence markdown → balanced-brace scan) lalu divalidasi Pydantic
dengan self-correction loop. Lima sub-agent (Technical, Macro, OnChain,
Sentiment, LiquidityRisk) berjalan konkuren dan terisolasi; Master Agent
menimbang konfluensi + regime (grounding: korpus riset di folder induk)
dan menghasilkan `BUY/HOLD/SELL + confidence + ringkasan + peringatan
risiko` dalam Bahasa Indonesia.

Catatan: `run_collectors.py` kini otomatis menjalankan Feature Engineering
sebagai tahap terakhir, jadi urutan harian yang disarankan cukup:
`run_collectors.py` → `run_inference.py`.

## Forecasting & Continuous Learning (Phase 6)

`run_daily.py` kini juga menjalankan dua tahap kuantitatif (terisolasi —
kegagalannya tidak menghalangi keputusan AI):

1. **Monte Carlo Block-Bootstrap** (`src/forecasting/engine.py`, pure numpy):
   log-returns harian ~3 tahun → blok 10-hari (mempertahankan *volatility
   clustering*) → **20.000 jalur** per horizon **7/30/60/90/180 hari** →
   persentil P5/P50/P95 + `prob_price_up` → upsert ke tabel `forecasts`
   (RNG di-seed dari tanggal → hasil harian reproducible; parameter `as_of`
   tersedia untuk backtest tanpa look-ahead).
2. **Evaluasi berkelanjutan** (`src/forecasting/evaluator.py`): forecast
   yang jatuh tempo dinilai vs harga aktual → `price_error_pct` +
   `directional_correct` → tabel `forecast_evaluations` + scoreboard
   (MAPE & directional hit-rate) di log.

Master Agent menerima ringkasan forecast ini sebagai *statistical base
rate* dalam konteks keputusannya.

## Continuous Learning & Calibration (Phase 7)

Setelah evaluasi, `LearningEvaluator` (`src/forecasting/learning.py`)
menghitung metrik kalibrasi per horizon dari 30 evaluasi terakhir dan
menyimpannya ke tabel `learning_metrics` (skip anggun bila < 10 evaluasi):

- **magnitude_bias** = rata-rata `(actual - p50)/p50` (fraksi). >0 = model
  memprediksi terlalu **rendah** (terlalu pesimis) → koreksi menggeser naik.
- **coverage_rate** = fraksi aktual di dalam [P5, P95] (~0.90 ideal;
  <0.85 overconfident/band terlalu sempit; >0.95 underconfident).
- **rolling_mape** (persen) & **brier_score** (0..1, akurasi probabilitas arah).

**Feedback loop:** `ForecastEngine` membaca metrik terbaru dan mengoreksi
simulasi berikutnya — **bias correction** (drift `ln(1+bias)/horizon` per hari,
di-clamp ±25%) dan **volatility scaling** (coverage <0.85 → ×1.1 memperlebar
band; >0.95 → ×0.95). Di `run_daily.py`, urutannya **Evaluasi → Learning →
Forecast** sehingga forecast hari ini memakai koreksi yang baru dihitung.

> ⚠️ Catatan definisi: spesifikasi awal menyebut "bias positif = terlalu
> optimis", tetapi rumus `(actual - p50)/p50` berarti bias positif = prediksi
> terlalu **rendah** (pesimis). Implementasi mengikuti rumus (Task 2), yang
> konsisten secara matematis.

Query kalibrasi & akurasi sistem:

```sql
-- Metrik kalibrasi terbaru per horizon
SELECT DISTINCT ON (horizon_days) horizon_days, sample_size,
       round(rolling_mape::numeric,2) AS mape_pct,
       round(magnitude_bias::numeric*100,2) AS bias_pct,
       round(coverage_rate::numeric*100,1) AS coverage_pct,
       round(brier_score::numeric,3) AS brier
FROM learning_metrics ORDER BY horizon_days, evaluated_at DESC;

-- Scoreboard akurasi keseluruhan
SELECT count(*) AS n, round(avg(abs(price_error_pct)),2) AS mape_pct,
       round(avg(directional_correct::int)*100,1) AS hit_rate_pct
FROM forecast_evaluations;
```

Uji layer ini: `python scripts/test_learning_phase7.py` (menyuntik bias +10%
sintetis, memverifikasi deteksi & auto-koreksi, lalu membersihkan diri).

## Status Fase

- [x] **Phase 1** — Fondasi (environment, schema, base models, Qwen client)
- [x] **Phase 2** — Data Collection (6 fetcher + orkestrasi + penjadwalan)
- [x] **Phase 3** — Feature Engineering (fitur harian → `daily_features`)
- [x] **Phase 4** — AI Reasoning Engine (5 sub-agent + Master, Laguna M.1)
- [x] **Phase 5** — Persistence & Automation (`decisions` + `run_daily.py`)
- [x] **Phase 6** — Forecasting (Monte Carlo block-bootstrap + evaluasi)
- [x] **Phase 7** — Continuous Learning & Calibration (`learning_metrics` + feedback loop)
- [x] **Phase 8 Part 1** — Altcoin Data Foundation & Coin DNA (`coins_registry`,
      `tokenomics_events`, pipeline multi-aset, relative strength vs BTC)
- [x] **Phase 8 Part 2** — Regime Detection & Dynamic Weighting
      (`market_regimes`, `RegimeDetector`, `DynamicWeightingEngine`)
- [x] **Phase 8 Part 3** — Sector Rotation Tracker (`sector_performance`,
      `SectorTracker`, registry 5 → 21 simbol)
- [x] **Phase 8 Part 4** — Tokenomics & Coin DNA (`supply_metrics`,
      `SupplyFetcher`, `seed_tokenomics.py`, 37 → 49 fitur)
- [x] **Phase 8 Part 5** — Agent Integration & Context Injection
      (`src/agents/utils.py`, DNA per pilar + ringkasan Master)
- [x] **Phase 8 Part 6** — Multi-Asset Inference Pipeline (`decisions` re-key,
      loop 21 aset, isolasi error, deteksi kuota harian)

Penggunaan harian tetap: `python scripts/run_daily.py` (jalur keputusan masih
BTC-only; inference per-altcoin belum dibangun).
