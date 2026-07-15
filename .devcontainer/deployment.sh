SECRET_SERVER="http://52.186.168.229:3800"

fetch_secret() {
  curl -s -X POST "$SECRET_SERVER/$1" \
    -H "Content-Type: application/json" \
    -d "{\"password\": \"$WORKSHOP_PASSWORD\"}"
}

# Fetch Azure and OTel values from secret server
AZURE_ENDPOINT=$(fetch_secret azure_endpoint)
AZURE_DEPLOYMENT=$(fetch_secret azure_deployment)
AZURE_SUBSCRIPTION_KEY=$(fetch_secret azure_subscription_key)
AZURE_API_VERSION=$(fetch_secret azure_api_version)
OTEL_SERVICE_NAME=$(fetch_secret otel_service_name)

# Replace placeholders in setEnv.sh
sed -i "s,DYNATRACE_EXPORTER_OTLP_ENDPOINT_TOREPLACE,$DYNATRACE_EXPORTER_OTLP_ENDPOINT," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,DYNATRACE_API_TOKEN_TOREPLACE,$DYNATRACE_API_TOKEN," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,AZURE_ENDPOINT_TOREPLACE,$AZURE_ENDPOINT," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,AZURE_DEPLOYMENT_TOREPLACE,$AZURE_DEPLOYMENT," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,AZURE_SUBSCRIPTION_KEY_TOREPLACE,$AZURE_SUBSCRIPTION_KEY," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,AZURE_API_VERSION_TOREPLACE,$AZURE_API_VERSION," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,OTEL_SERVICE_NAME_TOREPLACE,$OTEL_SERVICE_NAME," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,TAVILY_API_KEY_TOREPLACE,$TAVILY_API_KEY," /workspaces/$RepositoryName/setEnv.sh

source /workspaces/$RepositoryName/setEnv.sh
chmod +x /workspaces/$RepositoryName/run.sh
