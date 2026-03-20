# mapa-objekty

Detekcja błędnie oznaczonych drzew na drogach w mapach rastrowych (GeoTIFF).

Problem: oznaczenia drzew (zielone okręgi) w mapie topograficznej nie powinny nachodić na drogi (żółte/szare pasy). Celem projektu jest automatyczne wykrycie takich błędów na dużym pliku TIFF przy użyciu klasycznego CV do auto-labelowania i sieci neuronowej (EfficientNet-B0) do detekcji.

---

## Struktura projektu

```
mapa-objekty/
├── dane-mapa/
│   ├── N-34-137-B-d-3.tif          # Mapa rastrowa (38 MB)
│   ├── img/
│   │   ├── ok/                      # Ręczne przykłady poprawnych oznaczeń
│   │   └── not-ok/                  # Ręczne przykłady błędnych oznaczeń
│   └── patches/                     # Auto-generated (tworzony przez patch_extractor.py)
│       ├── ok/
│       └── not-ok/
├── src/
│   ├── color_detector.py            # HSV detekcja + obliczanie IoU
│   ├── patch_extractor.py           # Sliding window + auto-labelowanie z TIFF
│   ├── dataset.py                   # PyTorch Dataset + augmentacja
│   ├── model.py                     # EfficientNet-B0 (PyTorch / torchvision)
│   ├── train.py                     # Pętla treningowa PyTorch
│   ├── inference.py                 # Inferencja PyTorch → heatmapa + CSV
│   ├── model_keras.py               # EfficientNet-B0 (Keras / keras_hub)
│   ├── train_keras.py               # Trening Keras (model.fit)
│   └── inference_keras.py           # Inferencja Keras → heatmapa + CSV
├── models/                          # Zapisane wagi modeli
├── output/                          # Wyniki: heatmapy i pliki CSV
└── pyproject.toml
```

---

## Jak działa pipeline

```
Duży TIFF
    │
    ▼
patch_extractor.py          ← klasyczne CV (color_detector.py)
    │  sliding window 128×128
    │  auto-labelowanie przez IoU zielone okręgi ∩ drogi
    ▼
dane-mapa/patches/ok|not-ok
    │
    ├── + ręczne przykłady (dane-mapa/img/)
    ▼
train.py / train_keras.py   ← EfficientNet-B0, pretrained ImageNet
    │  fine-tuning binarny: OK vs NOT-OK
    ▼
models/best_model.pt        ← PyTorch
models/best_model_keras.keras ← Keras
    │
    ▼
inference.py / inference_keras.py
    │  sliding window po całym TIFF
    │  Gaussian blending nakładających się patchy
    │  NMS grupowanie detekcji
    ▼
output/heatmap.png          ← nakładka termiczna na mapę
output/detections.csv       ← x, y, w, h, confidence każdego błędu
```

---

## Instalacja

Wymagany [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

---

## Użycie

### Krok 1 — Auto-labelowanie patchy z TIFF

Generuje tysiące przykładów treningowych z dużej mapy przez detekcję kolorów HSV.

```bash
uv run python src/patch_extractor.py \
    --tiff dane-mapa/N-34-137-B-d-3.tif \
    --out dane-mapa/patches \
    --patch-size 128 \
    --stride 64 \
    --iou-threshold 0.05 \
    --max-patches 5000
```

Wynik: `dane-mapa/patches/ok/` i `dane-mapa/patches/not-ok/`

### Krok 2 — Trening modelu

**PyTorch (torchvision):**
```bash
uv run python src/train.py \
    --manual-dir dane-mapa/img \
    --auto-dir dane-mapa/patches \
    --epochs 30 \
    --batch-size 32 \
    --out models/best_model.pt
```

**Keras (keras_hub):**
```bash
uv run python src/train_keras.py \
    --manual-dir dane-mapa/img \
    --auto-dir dane-mapa/patches \
    --epochs 30 \
    --batch-size 32 \
    --out models/best_model_keras.keras
```

### Krok 3 — Inferencja na pełnym TIFF

**PyTorch:**
```bash
uv run python src/inference.py \
    --tiff dane-mapa/N-34-137-B-d-3.tif \
    --model models/best_model.pt \
    --out output \
    --stride 32 \
    --threshold 0.5
```

**Keras:**
```bash
uv run python src/inference_keras.py \
    --tiff dane-mapa/N-34-137-B-d-3.tif \
    --model models/best_model_keras.keras \
    --out output \
    --stride 32 \
    --threshold 0.5
```

---

## Wyniki

| Plik | Opis |
|---|---|
| `output/heatmap.png` | Mapa ciepła nałożona na TIFF (PyTorch) |
| `output/heatmap_keras.png` | Mapa ciepła nałożona na TIFF (Keras) |
| `output/detections.csv` | Lista detekcji: `x, y, w, h, confidence` (PyTorch) |
| `output/detections_keras.csv` | Lista detekcji (Keras) |
| `output/prob_map.npy` | Surowa mapa prawdopodobieństwa numpy (PyTorch) |

Współrzędne w CSV są pikselowe względem oryginalnego TIFF. Jeśli plik jest GeoTIFF, można je przeliczać na współrzędne geograficzne przez `rasterio`.

---

## Moduły

| Moduł | Odpowiedzialność |
|---|---|
| `color_detector.py` | Segmentacja HSV zielonych okręgów i pasów dróg; obliczanie IoU nakładania |
| `patch_extractor.py` | Sliding window po TIFF; auto-labelowanie przez `color_detector` |
| `dataset.py` | `TreeRoadDataset` (PyTorch) + augmentacja + `WeightedRandomSampler` |
| `model.py` | `build_model()` — EfficientNet-B0 (torchvision) z wymienioną głową |
| `train.py` | Ręczna pętla treningowa PyTorch z early stopping i `CosineAnnealingLR` |
| `inference.py` | Sliding window inferencja; Gaussian blending; NMS; zapis heatmapy i CSV |
| `model_keras.py` | `build_model_keras()` — EfficientNet-B0 z `keras_hub` |
| `train_keras.py` | Trening przez `model.fit()` z callbackami Keras |
| `inference_keras.py` | Inferencja Keras; reużywa helperów z `inference.py` |

---

## Parametry kluczowe

| Parametr | Domyślnie | Opis |
|---|---|---|
| `--patch-size` | 128 | Rozmiar okna w pikselach |
| `--stride` | 64 (ekstrakcja) / 32 (inferencja) | Krok sliding window |
| `--iou-threshold` | 0.05 | Minimalne nakładanie okrąg∩droga → NOT-OK |
| `--threshold` | 0.5 | Próg pewności modelu do detekcji |
| `--max-patches` | 5000 | Limit auto-generated patchy z TIFF |

---

## Zależności

- Python ≥ 3.11
- `torch` + `torchvision` — model PyTorch
- `keras` + `keras-hub` — model Keras (backend: TensorFlow)
- `opencv-python` — przetwarzanie obrazu, HSV, kontury
- `rasterio` — wczytywanie GeoTIFF z metadanymi
- `scikit-learn` — podział train/val, stratified split
- `scipy` — NMS przez `label` z `ndimage`
- `matplotlib` — generowanie heatmap

---

## API (FastAPI)

### Uruchomienie lokalnie

```bash
uv run uvicorn app.main:app --reload --port 8080
```

### Endpoint `POST /detect`

Przyjmuje plik TIFF, zwraca archiwum ZIP z wynikami.

```bash
curl -X POST http://localhost:8080/detect \
  -F "file=@dane-mapa/N-34-137-B-d-3.tif" \
  -F "threshold=0.5" \
  -F "stride=32" \
  -o results.zip
```

| Parametr | Typ | Domyślnie | Opis |
|---|---|---|---|
| `file` | TIFF (multipart) | — | Mapa rastrowa do analizy |
| `threshold` | float 0–1 | `0.5` | Próg pewności modelu |
| `stride` | int 8–128 | `32` | Krok sliding window (px) |

**Odpowiedź:** `application/zip` zawierający:
- `heatmap.png` — nakładka termiczna na mapę
- `detections.csv` — `x, y, w, h, confidence` każdej detekcji

### Inne endpointy

| Endpoint | Opis |
|---|---|
| `GET /` | Info o serwisie |
| `GET /health` | Health check (używany przez Cloud Run) |
| `GET /docs` | Swagger UI |

---

## Deploy — Google Cloud Run

### Wymagania wstępne

#### 1. Zaloguj się i ustaw projekt

```bash
gcloud auth login
gcloud auth configure-docker europe-central2-docker.pkg.dev
gcloud config set project TWOJ_PROJEKT
```

#### 2. Włącz wymagane API (raz na projekt)

```bash
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  --project=TWOJ_PROJEKT
```

#### 3. Utwórz bucket na modele (raz)

```bash
gsutil mb -l europe-central2 gs://TWOJ_PROJEKT-models
```

Wgraj wytrenowany model:
```bash
gsutil cp models/best_model.pt gs://TWOJ_PROJEKT-models/map-annotation/best_model.pt
```

Sprawdź że plik jest dostępny:
```bash
gsutil ls gs://TWOJ_PROJEKT-models/map-annotation/
```

#### 4. Utwórz repozytorium w Artifact Registry (raz)

```bash
gcloud artifacts repositories create docker-repo \
  --repository-format=docker \
  --location=europe-central2 \
  --project=TWOJ_PROJEKT
```

#### 5. Uprawnienia Service Account dla Cloud Run

Cloud Run potrzebuje dostępu do GCS żeby pobrać model przy starcie:

```bash
# Pobierz domyślny service account Cloud Run
SA=$(gcloud iam service-accounts list \
  --filter="displayName:Compute Engine default" \
  --format="value(email)" \
  --project=TWOJ_PROJEKT)

# Nadaj dostęp do odczytu z GCS
gcloud projects add-iam-policy-binding TWOJ_PROJEKT \
  --member="serviceAccount:${SA}" \
  --role="roles/storage.objectViewer"
```

### Deploy

```bash
GCP_PROJECT=twoj-projekt-gcp ./deploy.sh
```

Skrypt automatycznie:
1. Buduje obraz Docker (`linux/amd64`)
2. Pushuje do Artifact Registry
3. Deployuje na Cloud Run z `--no-allow-unauthenticated`

### Wywołanie produkcyjnego API

```bash
# Pobierz token
TOKEN=$(gcloud auth print-identity-token)

curl -X POST https://CLOUD_RUN_URL/detect \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "file=@mapa.tif" \
  -o results.zip
```

### Sprawdzanie logów Cloud Run

```bash
# Ostatnie 50 logów
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=map-annotation-validator" \
  --project=$GCP_PROJECT \
  --limit=50 \
  --format="value(textPayload)"

# Tylko błędy
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=map-annotation-validator AND severity>=ERROR" \
  --project=$GCP_PROJECT \
  --limit=20 \
  --format="value(textPayload)"

# Stream logów na żywo (podczas deployu)
gcloud beta run services logs tail map-annotation-validator \
  --project=$GCP_PROJECT \
  --region=europe-central2
```

---

### Zasoby Cloud Run

| Parametr | Wartość |
|---|---|
| Pamięć | 2 GB |
| CPU | 2 vCPU |
| Timeout | 300 s |
| Max instancji | 3 |
| Port | 8080 |

### Zmienne środowiskowe

| Zmienna | Opis |
|---|---|
| `MODEL_GCS_URI` | `gs://bucket/path/best_model.pt` — model ładowany z GCS |
| `MODEL_PATH` | Lokalny fallback gdy brak GCS URI |
