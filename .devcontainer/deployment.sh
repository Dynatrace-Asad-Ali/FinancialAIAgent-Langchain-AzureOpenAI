sed -i "s,DYNATRACE_EXPORTER_OTLP_ENDPOINT_TOREPLACE,$DYNATRACE_EXPORTER_OTLP_ENDPOINT," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,DYNATRACE_API_TOKEN_TOREPLACE,$DYNATRACE_API_TOKEN," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,AZURE_ENDPOINT_TOREPLACE,$AZURE_ENDPOINT," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,AZURE_MODEL_NAME_TOREPLACE,$AZURE_MODEL_NAME," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,AZURE_DEPLOYMENT_TOREPLACE,$AZURE_DEPLOYMENT," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,AZURE_SUBSCRIPTION_KEY_TOREPLACE,$AZURE_SUBSCRIPTION_KEY," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,AZURE_API_VERSION_TOREPLACE,$AZURE_API_VERSION," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,AZURE_EMBEDDINGS_MODEL_DEPLOYMENT_TOREPLACE,$AZURE_EMBEDDINGS_MODEL_DEPLOYMENT," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,AZURE_EMBEDDINGS_MODEL_NAME_TOREPLACE,$AZURE_EMBEDDINGS_MODEL_NAME," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,AZURE_EMBEDDINGS_API_VERSION_TOREPLACE,$AZURE_EMBEDDINGS_API_VERSION," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,AZURE_OPENAI_API_KEY_TOREPLACE,$AZURE_OPENAI_API_KEY," /workspaces/$RepositoryName/setEnv.sh
sed -i "s,TAVILY_API_KEY_TOREPLACE,$TAVILY_API_KEY," /workspaces/$RepositoryName/setEnv.sh

source /workspaces/$RepositoryName/setEnv.sh
chmod +x /workspaces/$RepositoryName/run.sh
chmod +x /workspaces/$RepositoryName/mcp/run-mcp.sh
