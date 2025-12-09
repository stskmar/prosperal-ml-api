# ML Batch Job Worker + B2 Storage + GCP Deployment
[TODO] Project desc

---
## Project Structure
````
prosperal-ml/
├── dataset/
│	└── bank-full.csv
├── notebooks/
│	├── bank_lead_scoring.ipynb
│	└── requirements.txt
├── service/
│	├── inference_service.py
│	├── main.py
│	└── requirements.txt
│	└── artifacts/
│		├── CatBoost_calibrated_model.pkl
│		├── preprocessor.pkl
│		├── label_encoders.pkl
│		├── feature_names.pkl
│		└── shap_explainer.pkl
└── temp/
````

## Requirements
- Python 3.10+
- Docker
- Google Cloud Console
- Backblaze B2

---
## Clone Repository
````
git clone https://github.com/stskmar/prosperal-ml-api.git
````

---
## Backblaze B2 Setup
1. Buat akun di Backblaze
2. Lalu buat bucket bernama <bucket-name>
3. Klik bagian **App Keys** →  Create Application Key
4. Pilih permission **Read/Write**

---
## GCP Setup
### Export variables
````
export PROJECT_ID=<project_id>
export REGION=<region>
export ARTIFACT_REGISTRY=<artifact_registry_name>
export REPO_NAME=<>
````

### Create project & activate billing
````
gcloud projects create <project-id>
gcloud beta billing projects link <project-id> --billing-account=<billing-id>
gcloud config set project <project-id>
```` 

### Enable required APIs
````
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com \
  cloudkms.googleapis.com
````

### Create service accounts
1. **Cloud Build SA**
  ````
  gcloud iam service-accounts create cloud-build-sa --display-name "Cloud Build SA"

  gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:cloud-build-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"
  ````
2. **Cloud Run Job SA**
  ````
  gcloud iam service-accounts create job-runner-sa --display-name "Job Runner SA"

  gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:job-runner-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.invoker"
  
  gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:job-runner-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
  ````
3. **Cloud Build**
  ````
  gcloud iam service-accounts create batch-worker-sa --display-name="Batch Worker Service Account"
  
  gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:batch-worker-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
  ````
4. **Scheduler Invoker**
  ````
  gcloud iam service-accounts create scheduler-invoker \
    --display-name="Scheduler Invoker" \
    --project=PROJECT_ID

  gcloud run jobs add-iam-policy-binding JOB_NAME \
    --member="serviceAccount:scheduler-invoker@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.invoker" \
    --region=REGION

  gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:scheduler-invoker@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
  ````

### Save secrets in Secret Manager
````
echo -n "B2_KEY_ID_VALUE" | gcloud secrets create B2_KEY_ID --data-file=-
echo -n "B2_BUCKET_NAME" | gcloud secrets create B2_BUCKET --data-file=-
echo -n "B2_APP_KEY_VALUE" | gcloud secrets create B2_APP_KEY --data-file=-
echo -n "B2_S3_ENDPOINT_VALUE" | gcloud secrets create B2_S3_ENDPOINT --data-file=-
echo -n "INPUT_PATH_VALUE" | gcloud secrets create INPUT_PATH --data-file=-
````

### Artifact Registry
````
gcloud artifacts repositories create <registry_name> \
  --repository-format=docker \
  --location=REGION
````

### Trigger
````
gcloud beta builds triggers create github \
  --repo-name=prosperal-ml --repo-owner=<github_owner> \
  --branch-pattern="^main$" \
  --build-config=cloudbuild.yaml
````

### Build & Push image worker via Cloud Build
````
gcloud builds submit --tag \
    REGION-docker.pkg.dev/PROJECT_ID/<registry_name>/lead-worker:latest
````

### Deploy to Cloud Run Jobs
````
gcloud run jobs create lead-scoring-job \
  --image=REGION-docker.pkg.dev/PROJECT_ID/ml-images/lead-scoring:TAG \
  --region=REGION \
  --tasks=1 \
  --max-retries=0 \
  --service-account=job-runner-sa@PROJECT_ID.iam.gserviceaccount.com
````
Run job:
````
gcloud run jobs execute lead-scoring-job \
  --region=REGION \
  --set-env-vars=INPUT_PATH=projects/PROJECT_ID/secrets/INPUT_PATH:latest,OUTPUT_PREFIX=outputs/2025-12-08/job-1234 \
  --set-secrets=B2_KEY_ID=projects/PROJECT_ID/secrets/B2_KEY_ID:latest,B2_APP_KEY=projects/PROJECT_ID/secrets/B2_APP_KEY:latest
````

### Create Scheduler
````
PROJECT_NUMBER=$(gcloud projects describe PROJECT_ID --format='value(projectNumber)')

gcloud iam service-accounts add-iam-policy-binding scheduler-invoker@PROJECT_ID.iam.gserviceaccount.com \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudscheduler.gserviceaccount.com" \
  --role="roles/iam.serviceAccountTokenCreator" \
  --project=PROJECT_ID

gcloud scheduler jobs create http run-lead-daily \
  --schedule="0 2 * * *" \
  --uri="https://REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/PROJECT_ID/jobs/JOB_NAME:run" \
  --http-method=POST \
  --oidc-service-account-email=scheduler-invoker@PROJECT_ID.iam.gserviceaccount.com \
  --location=REGION \
  --message-body='{}'
````