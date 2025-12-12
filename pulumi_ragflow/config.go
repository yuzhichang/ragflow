package main

import "os"

// RAGFlowConfig holds RAGFlow-specific configuration
type RAGFlowConfig struct {
	Image   string
	Service ServiceConfig
	API     APIConfig
}

// ServiceConfig holds service configuration
type ServiceConfig struct {
	Type    string
	Enabled bool
}

// APIConfig holds API service configuration
type APIConfig struct {
	Service ServiceConfig
}

// GatewayConfig holds Gateway configuration
type GatewayConfig struct {
	ClassName   string
	Namespace   string
	Annotations map[string]string
	Hosts       []GatewayHost
	TLS         []GatewayTLS
	Host        string
}

// GatewayHost holds Gateway host configuration
type GatewayHost struct {
	Host  string
	Paths []GatewayPath
}

// GatewayPath holds Gateway path configuration
type GatewayPath struct {
	Path     string
	PathType string
}

// GatewayTLS holds Gateway TLS configuration
type GatewayTLS struct {
	Hosts      []string
	SecretName string
}

// Simple typed config for Pulumi stack values used in the PoC.
type StackConfig struct {
	Name      string
	Namespace string
	Env       map[string]string
	RAGFlow   RAGFlowConfig
	Gateway   GatewayConfig
}

// LoadConfig reads minimal config values from environment variables
// for the PoC migration to avoid depending on Pulumi config.
func LoadConfig() (StackConfig, error) {
	name := os.Getenv("PULUMI_NAME")
	if name == "" {
		name = "ragflow"
	}
	namespace := os.Getenv("PULUMI_NAMESPACE")
	if namespace == "" {
		namespace = "ragflow"
	}

	// Gateway configuration (always enabled). Gateway class is detected automatically, host can be overridden.
	gatewayHost := os.Getenv("RAGFLOW_GATEWAY") // If empty, a hostless Gateway/HTTPRoute will be created
	if gatewayHost == "" {
		gatewayHost = "ragflow.local"
	}

	env := map[string]string{
		"DOC_ENGINE":          "elasticsearch",
		"RAGFLOW_IMAGE":       "infiniflow/ragflow:nightly",
		"STACK_VERSION":       "8.11.3",
		"MYSQL_HOST":          "mysql",
		"MYSQL_PORT":          "3306",
		"MYSQL_DBNAME":        "ragflow",
		"MYSQL_USER":          "root",
		"MYSQL_PASSWORD":      "root",
		"REDIS_HOST":          "redis",
		"REDIS_PASSWORD":      "infini_rag_flow",
		"ES_HOST":             "elasticsearch",
		"ELASTIC_PASSWORD":    "infini_rag_flow",
		"MINIO_HOST":          "minio",
		"MINIO_PORT":          "9000",
		"MINIO_USER":          "minioadmin",
		"MINIO_PASSWORD":      "minioadmin",
		"MINIO_BUCKET":        "ragflow",
		"PYTHONPATH":          "/ragflow",
		"TEI_HOST":            "tei",
		"TEI_MODEL":           "BAAI/bge-small-en-v1.5",
		"SVR_WEB_HTTP_PORT":   "80",
		"SVR_WEB_HTTPS_PORT":  "443",
		"SVR_HTTP_PORT":       "9380",
		"ADMIN_SVR_HTTP_PORT": "9381",
		"SVR_MCP_PORT":        "9382",
		"COMPOSE_PROFILES":    "elasticsearch,cpu,tei-cpu",
	}

	ragflow := RAGFlowConfig{
		Image: env["RAGFLOW_IMAGE"],
		Service: ServiceConfig{
			Type: "ClusterIP",
		},
		API: APIConfig{
			Service: ServiceConfig{
				Type:    "ClusterIP",
				Enabled: true,
			},
		},
	}

	gateway := GatewayConfig{
		ClassName:   "",
		Namespace:   namespace,           // Use the same namespace as other resources
		Annotations: map[string]string{}, // Can be extended later
		Hosts:       []GatewayHost{},     // Can be extended later
		TLS:         []GatewayTLS{},      // Can be extended later
		Host:        gatewayHost,
	}

	return StackConfig{
		Name:      name,
		Namespace: namespace,
		Env:       env,
		RAGFlow:   ragflow,
		Gateway:   gateway,
	}, nil
}
