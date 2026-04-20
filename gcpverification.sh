echo "========================================"
echo "=== 1. CLOUD RUN SERVICES ==="
echo "========================================"
gcloud run services list --region=asia-south1 --project=project-7b98dc63-2de4-4f60-8b8

echo ""
echo "========================================"
echo "=== 2. BACKEND HEALTH CHECK ==="
echo "========================================"
curl -s https://sentiment-api-565865319827.asia-south1.run.app/health

echo ""
echo ""
echo "========================================"
echo "=== 3. FRONTEND PING ==="
echo "========================================"
curl -sI https://sentiment-dashboard-565865319827.asia-south1.run.app | head -3

echo ""
echo "========================================"
echo "=== 4. ARTIFACT REGISTRY (DOCKER IMAGES) ==="
echo "========================================"
gcloud artifacts docker images list asia-south1-docker.pkg.dev/project-7b98dc63-2de4-4f60-8b8/cloud-run-source-deploy --project=project-7b98dc63-2de4-4f60-8b8 --limit=10 2>&1 | head -20

echo ""
echo "========================================"
echo "=== 5. RECENT CLOUD BUILDS ==="
echo "========================================"
gcloud builds list --region=asia-south1 --project=project-7b98dc63-2de4-4f60-8b8 --limit=5 2>&1 | head -10

echo ""
echo "========================================"
echo "=== 6. CLOUD STORAGE BUCKETS ==="
echo "========================================"
gcloud storage buckets list --project=project-7b98dc63-2de4-4f60-8b8 --format="value(name)" 2>&1 | head -10

echo ""
echo "========================================"
echo "=== VERIFICATION COMPLETE ==="
echo "========================================"