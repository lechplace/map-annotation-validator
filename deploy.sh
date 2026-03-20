#!/usr/bin/env bash
# deploy.sh
# Buduje obraz Docker, pushuje do Artifact Registry i deployuje na Cloud Run.
#
# Wymagania:
#   gcloud CLI zalogowany: gcloud auth login
#   Docker działający lokalnie
#
# Użycie:
#   chmod +x deploy.sh
#   ./deploy.sh

set -euo pipefail

# ── Załaduj .env jeśli istnieje ───────────────────────────────────────────────
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# ── Konfiguracja — dostosuj do swojego projektu GCP ──────────────────────────
GCP_PROJECT="${GCP_PROJECT:-twoj-projekt-gcp}"
GCP_REGION="${GCP_REGION:-europe-central2}"          # Warszawa
SERVICE_NAME="map-annotation-validator"
IMAGE_NAME="map-annotation-validator"

# Artifact Registry — repozytorium musi istnieć
AR_REPO="docker-repo"
IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO}/${IMAGE_NAME}:latest"

# Model w GCS — wgraj trained model przed deployem
MODEL_GCS_URI="${MODEL_GCS_URI:-gs://${GCP_PROJECT}-models/map-annotation/best_model.pt}"

# Cloud Run — zasoby
MEMORY="2Gi"
CPU="2"
TIMEOUT="300"   # sekund (max czas przetwarzania dużego TIFF)
MAX_INSTANCES="3"
# ─────────────────────────────────────────────────────────────────────────────

echo "==> Projekt GCP:  ${GCP_PROJECT}"
echo "==> Region:       ${GCP_REGION}"
echo "==> Obraz:        ${IMAGE_URI}"
echo "==> Model GCS:    ${MODEL_GCS_URI}"
echo ""

# 1. Uwierzytelnij Docker do Artifact Registry
echo "==> Konfiguracja Docker auth..."
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

# 2. Zbuduj obraz
echo "==> Budowanie obrazu Docker..."
docker build --platform linux/amd64 -t "${IMAGE_URI}" .

# 3. Push do Artifact Registry
echo "==> Push obrazu..."
docker push "${IMAGE_URI}"

# 4. Deploy na Cloud Run
echo "==> Deploy na Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_URI}" \
  --region "${GCP_REGION}" \
  --platform managed \
  --memory "${MEMORY}" \
  --cpu "${CPU}" \
  --timeout "${TIMEOUT}" \
  --max-instances "${MAX_INSTANCES}" \
  --set-env-vars "MODEL_GCS_URI=${MODEL_GCS_URI}" \
  --no-allow-unauthenticated \
  --port 8080 \
  --project "${GCP_PROJECT}"

echo ""
echo "==> Deploy zakończony!"
echo "==> URL serwisu:"
gcloud run services describe "${SERVICE_NAME}" \
  --region "${GCP_REGION}" \
  --project "${GCP_PROJECT}" \
  --format "value(status.url)"
