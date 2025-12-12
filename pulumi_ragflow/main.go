package main

// https://www.pulumi.com/docs/iac/guides/building-extending/components/build-a-component/

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"

	k8smetav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/clientcmd"

	"github.com/pulumi/pulumi-kubernetes/sdk/v4/go/kubernetes"
	apiextensions "github.com/pulumi/pulumi-kubernetes/sdk/v4/go/kubernetes/apiextensions"
	v1 "github.com/pulumi/pulumi-kubernetes/sdk/v4/go/kubernetes/apps/v1"
	corev1 "github.com/pulumi/pulumi-kubernetes/sdk/v4/go/kubernetes/core/v1"
	metav1 "github.com/pulumi/pulumi-kubernetes/sdk/v4/go/kubernetes/meta/v1"
	"github.com/pulumi/pulumi/sdk/v3/go/pulumi"
)

// Helper function to convert string port to int
func parsePort(portStr string) int {
	port, err := strconv.Atoi(portStr)
	if err != nil {
		// Default to 80 if parsing fails
		return 80
	}
	return port
}

func main() {
	pulumi.Run(func(ctx *pulumi.Context) error {
		config, err := LoadConfig()
		if err != nil {
			return err
		}

		// Create Kubernetes provider
		k8sProvider, err := kubernetes.NewProvider(ctx, "k8s-provider", &kubernetes.ProviderArgs{})
		if err != nil {
			return err
		}

		// Create namespace
		namespace, err := corev1.NewNamespace(ctx, "ragflow-namespace", &corev1.NamespaceArgs{
			Metadata: &metav1.ObjectMetaArgs{
				Name: pulumi.String(config.Namespace),
			},
		}, pulumi.Provider(k8sProvider))
		if err != nil {
			return err
		}

		// Create MySQL
		mysqlDeployment, mysqlService, err := createMySQL(ctx, &config, namespace, k8sProvider)
		if err != nil {
			return err
		}

		// Create Redis
		redisDeployment, redisService, err := createRedis(ctx, &config, namespace, k8sProvider)
		if err != nil {
			return err
		}

		// Create Elasticsearch/Infinity
		esDeployment, esService, err := createElasticsearch(ctx, &config, namespace, k8sProvider)
		if err != nil {
			return err
		}

		// Create MinIO
		minioDeployment, minioService, err := createMinIO(ctx, &config, namespace, k8sProvider)
		if err != nil {
			return err
		}

		// Create TEI
		teiDeployment, teiService, err := createTEI(ctx, &config, namespace, k8sProvider)
		if err != nil {
			return err
		}

		// Create RAGFlow deployment
		ragflowDeployment, ragflowService, err := createRAGFlowDeployment(ctx, &config, namespace, k8sProvider)
		if err != nil {
			return err
		}

		// Create Gateway (always enabled)
		gatewayClass, err := detectGatewayType(ctx, k8sProvider)
		if err != nil {
			return fmt.Errorf("failed to detect gateway class: %w", err)
		}

		gateway, err := createGateway(ctx, &config, k8sProvider, ragflowService, gatewayClass)
		if err != nil {
			return err
		}

		// Export outputs
		ctx.Export("namespace", namespace.Metadata.Name())
		ctx.Export("mysqlDeployment", mysqlDeployment.Metadata.Name())
		ctx.Export("mysqlService", mysqlService.Metadata.Name())
		ctx.Export("redisDeployment", redisDeployment.Metadata.Name())
		ctx.Export("redisService", redisService.Metadata.Name())
		ctx.Export("esDeployment", esDeployment.Metadata.Name())
		ctx.Export("esService", esService.Metadata.Name())
		ctx.Export("minioDeployment", minioDeployment.Metadata.Name())
		ctx.Export("minioService", minioService.Metadata.Name())
		ctx.Export("teiDeployment", teiDeployment.Metadata.Name())
		ctx.Export("teiService", teiService.Metadata.Name())
		ctx.Export("ragflowDeployment", ragflowDeployment.Metadata.Name())
		ctx.Export("ragflowService", ragflowService.Metadata.Name())
		ctx.Export("gatewayClass", pulumi.String(gatewayClass))
		ctx.Export("gatewayName", gateway.Metadata.Name())

		return nil
	})
}

func createMySQL(ctx *pulumi.Context, config *StackConfig, namespace *corev1.Namespace, provider pulumi.ProviderResource) (*v1.Deployment, *corev1.Service, error) {
	// MySQL PVC
	mysqlPVC, err := corev1.NewPersistentVolumeClaim(ctx, "mysql-pvc", &corev1.PersistentVolumeClaimArgs{
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String("mysql-pvc"),
			Namespace: namespace.Metadata.Name(),
		},
		Spec: &corev1.PersistentVolumeClaimSpecArgs{
			AccessModes: pulumi.StringArray{
				pulumi.String("ReadWriteOnce"),
			},
			Resources: &corev1.VolumeResourceRequirementsArgs{
				Requests: pulumi.StringMap{
					"storage": pulumi.String("1Gi"),
				},
			},
			StorageClassName: pulumi.String("rook-ceph-block"),
		},
	}, pulumi.Provider(provider))
	if err != nil {
		return nil, nil, err
	}

	// MySQL Deployment
	mysqlDeployment, err := v1.NewDeployment(ctx, "mysql-deployment", &v1.DeploymentArgs{
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String("mysql"),
			Namespace: namespace.Metadata.Name(),
		},
		Spec: &v1.DeploymentSpecArgs{
			Replicas: pulumi.Int(1),
			Selector: &metav1.LabelSelectorArgs{
				MatchLabels: pulumi.StringMap{
					"app": pulumi.String("mysql"),
				},
			},
			Template: &corev1.PodTemplateSpecArgs{
				Metadata: &metav1.ObjectMetaArgs{
					Labels: pulumi.StringMap{
						"app": pulumi.String("mysql"),
					},
				},
				Spec: &corev1.PodSpecArgs{
					Containers: corev1.ContainerArray{
						&corev1.ContainerArgs{
							Name:  pulumi.String("mysql"),
							Image: pulumi.String("mysql:8.0"),
							Ports: corev1.ContainerPortArray{
								&corev1.ContainerPortArgs{
									ContainerPort: pulumi.Int(3306),
								},
							},
							Env: corev1.EnvVarArray{
								&corev1.EnvVarArgs{
									Name:  pulumi.String("MYSQL_ROOT_PASSWORD"),
									Value: pulumi.String("root"),
								},
								&corev1.EnvVarArgs{
									Name:  pulumi.String("MYSQL_DATABASE"),
									Value: pulumi.String("ragflow"),
								},
							},
							VolumeMounts: corev1.VolumeMountArray{
								&corev1.VolumeMountArgs{
									Name:      pulumi.String("mysql-storage"),
									MountPath: pulumi.String("/var/lib/mysql"),
								},
							},
						},
					},
					Volumes: corev1.VolumeArray{
						&corev1.VolumeArgs{
							Name: pulumi.String("mysql-storage"),
							PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSourceArgs{
								ClaimName: mysqlPVC.Metadata.Name().Elem(),
							},
						},
					},
				},
			},
		},
	}, pulumi.Provider(provider))
	if err != nil {
		return nil, nil, err
	}

	// MySQL Service
	mysqlService, err := corev1.NewService(ctx, "mysql-service", &corev1.ServiceArgs{
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String("mysql"),
			Namespace: namespace.Metadata.Name(),
		},
		Spec: &corev1.ServiceSpecArgs{
			Selector: pulumi.StringMap{
				"app": pulumi.String("mysql"),
			},
			Ports: corev1.ServicePortArray{
				&corev1.ServicePortArgs{
					Port:       pulumi.Int(3306),
					TargetPort: pulumi.Int(3306),
				},
			},
		},
	}, pulumi.Provider(provider))
	if err != nil {
		return nil, nil, err
	}

	return mysqlDeployment, mysqlService, nil
}

func createRedis(ctx *pulumi.Context, config *StackConfig, namespace *corev1.Namespace, provider pulumi.ProviderResource) (*v1.Deployment, *corev1.Service, error) {
	// Redis Deployment
	redisDeployment, err := v1.NewDeployment(ctx, "redis-deployment", &v1.DeploymentArgs{
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String("redis"),
			Namespace: namespace.Metadata.Name(),
		},
		Spec: &v1.DeploymentSpecArgs{
			Replicas: pulumi.Int(1),
			Selector: &metav1.LabelSelectorArgs{
				MatchLabels: pulumi.StringMap{
					"app": pulumi.String("redis"),
				},
			},
			Template: &corev1.PodTemplateSpecArgs{
				Metadata: &metav1.ObjectMetaArgs{
					Labels: pulumi.StringMap{
						"app": pulumi.String("redis"),
					},
				},
				Spec: &corev1.PodSpecArgs{
					Containers: corev1.ContainerArray{
						&corev1.ContainerArgs{
							Name:  pulumi.String("redis"),
							Image: pulumi.String("valkey/valkey:8"),
							Ports: corev1.ContainerPortArray{
								&corev1.ContainerPortArgs{
									ContainerPort: pulumi.Int(6379),
								},
							},
							Command: pulumi.StringArray{
								pulumi.String("valkey-server"),
								pulumi.String("--requirepass"),
								pulumi.String("infini_rag_flow"),
							},
						},
					},
				},
			},
		},
	}, pulumi.Provider(provider))
	if err != nil {
		return nil, nil, err
	}

	// Redis Service
	redisService, err := corev1.NewService(ctx, "redis-service", &corev1.ServiceArgs{
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String("redis"),
			Namespace: namespace.Metadata.Name(),
		},
		Spec: &corev1.ServiceSpecArgs{
			Selector: pulumi.StringMap{
				"app": pulumi.String("redis"),
			},
			Ports: corev1.ServicePortArray{
				&corev1.ServicePortArgs{
					Port:       pulumi.Int(6379),
					TargetPort: pulumi.Int(6379),
				},
			},
		},
	}, pulumi.Provider(provider))
	if err != nil {
		return nil, nil, err
	}

	return redisDeployment, redisService, nil
}

func createElasticsearch(ctx *pulumi.Context, config *StackConfig, namespace *corev1.Namespace, provider pulumi.ProviderResource) (*v1.Deployment, *corev1.Service, error) {
	image := "elasticsearch:" + config.Env["STACK_VERSION"]

	// Elasticsearch/Infinity PVC
	esPVC, err := corev1.NewPersistentVolumeClaim(ctx, "es-pvc", &corev1.PersistentVolumeClaimArgs{
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String("es-pvc"),
			Namespace: namespace.Metadata.Name(),
		},
		Spec: &corev1.PersistentVolumeClaimSpecArgs{
			AccessModes: pulumi.StringArray{
				pulumi.String("ReadWriteOnce"),
			},
			Resources: &corev1.VolumeResourceRequirementsArgs{
				Requests: pulumi.StringMap{
					"storage": pulumi.String("2Gi"),
				},
			},
			StorageClassName: pulumi.String("rook-ceph-block"),
		},
	}, pulumi.Provider(provider))
	if err != nil {
		return nil, nil, err
	}

	// Elasticsearch/Infinity Deployment
	esDeployment, err := v1.NewDeployment(ctx, "es-deployment", &v1.DeploymentArgs{
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String("elasticsearch"),
			Namespace: namespace.Metadata.Name(),
		},
		Spec: &v1.DeploymentSpecArgs{
			Replicas: pulumi.Int(1),
			Selector: &metav1.LabelSelectorArgs{
				MatchLabels: pulumi.StringMap{
					"app": pulumi.String("elasticsearch"),
				},
			},
			Template: &corev1.PodTemplateSpecArgs{
				Metadata: &metav1.ObjectMetaArgs{
					Labels: pulumi.StringMap{
						"app": pulumi.String("elasticsearch"),
					},
				},
				Spec: &corev1.PodSpecArgs{
					SecurityContext: &corev1.PodSecurityContextArgs{
						FsGroup:    pulumi.Int(1000),
						RunAsUser:  pulumi.Int(1000),
						RunAsGroup: pulumi.Int(1000),
					},
					InitContainers: corev1.ContainerArray{
						&corev1.ContainerArgs{
							Name:  pulumi.String("init-es-data"),
							Image: pulumi.String("busybox"),
							SecurityContext: &corev1.SecurityContextArgs{
								RunAsUser:  pulumi.Int(0),
								RunAsGroup: pulumi.Int(0),
							},
							Command: pulumi.StringArray{
								pulumi.String("sh"),
								pulumi.String("-c"),
								pulumi.String("mkdir -p /usr/share/elasticsearch/data && chown -R 1000:1000 /usr/share/elasticsearch/data"),
							},
							VolumeMounts: corev1.VolumeMountArray{
								&corev1.VolumeMountArgs{
									Name:      pulumi.String("es-storage"),
									MountPath: pulumi.String("/usr/share/elasticsearch/data"),
								},
							},
						},
					},
					Containers: corev1.ContainerArray{
						&corev1.ContainerArgs{
							Name:  pulumi.String("elasticsearch"),
							Image: pulumi.String(image),
							Ports: corev1.ContainerPortArray{
								&corev1.ContainerPortArgs{
									ContainerPort: pulumi.Int(9200),
								},
							},
							Env: corev1.EnvVarArray{
								&corev1.EnvVarArgs{
									Name:  pulumi.String("discovery.type"),
									Value: pulumi.String("single-node"),
								},
								&corev1.EnvVarArgs{
									Name:  pulumi.String("xpack.security.enabled"),
									Value: pulumi.String("false"),
								},
							},
							VolumeMounts: corev1.VolumeMountArray{
								&corev1.VolumeMountArgs{
									Name:      pulumi.String("es-storage"),
									MountPath: pulumi.String("/usr/share/elasticsearch/data"),
								},
							},
							Resources: &corev1.ResourceRequirementsArgs{
								Requests: pulumi.StringMap{
									"memory": pulumi.String("2Gi"),
									"cpu":    pulumi.String("1000m"),
								},
								Limits: pulumi.StringMap{
									"memory": pulumi.String("4Gi"),
									"cpu":    pulumi.String("2000m"),
								},
							},
						},
					},
					Volumes: corev1.VolumeArray{
						&corev1.VolumeArgs{
							Name: pulumi.String("es-storage"),
							PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSourceArgs{
								ClaimName: esPVC.Metadata.Name().Elem(),
							},
						},
					},
				},
			},
		},
	}, pulumi.Provider(provider))
	if err != nil {
		return nil, nil, err
	}

	// Elasticsearch/Infinity Service
	esService, err := corev1.NewService(ctx, "es-service", &corev1.ServiceArgs{
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String("elasticsearch"),
			Namespace: namespace.Metadata.Name(),
		},
		Spec: &corev1.ServiceSpecArgs{
			Selector: pulumi.StringMap{
				"app": pulumi.String("elasticsearch"),
			},
			Ports: corev1.ServicePortArray{
				&corev1.ServicePortArgs{
					Port:       pulumi.Int(9200),
					TargetPort: pulumi.Int(9200),
				},
			},
		},
	}, pulumi.Provider(provider))
	if err != nil {
		return nil, nil, err
	}

	return esDeployment, esService, nil
}

func createMinIO(ctx *pulumi.Context, config *StackConfig, namespace *corev1.Namespace, provider pulumi.ProviderResource) (*v1.Deployment, *corev1.Service, error) {
	// MinIO PVC
	minioPVC, err := corev1.NewPersistentVolumeClaim(ctx, "minio-pvc", &corev1.PersistentVolumeClaimArgs{
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String("minio-pvc"),
			Namespace: namespace.Metadata.Name(),
		},
		Spec: &corev1.PersistentVolumeClaimSpecArgs{
			AccessModes: pulumi.StringArray{
				pulumi.String("ReadWriteOnce"),
			},
			Resources: &corev1.VolumeResourceRequirementsArgs{
				Requests: pulumi.StringMap{
					"storage": pulumi.String("1Gi"),
				},
			},
			StorageClassName: pulumi.String("rook-ceph-block"),
		},
	}, pulumi.Provider(provider))
	if err != nil {
		return nil, nil, err
	}

	// MinIO Deployment
	minioDeployment, err := v1.NewDeployment(ctx, "minio-deployment", &v1.DeploymentArgs{
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String("minio"),
			Namespace: namespace.Metadata.Name(),
		},
		Spec: &v1.DeploymentSpecArgs{
			Replicas: pulumi.Int(1),
			Selector: &metav1.LabelSelectorArgs{
				MatchLabels: pulumi.StringMap{
					"app": pulumi.String("minio"),
				},
			},
			Template: &corev1.PodTemplateSpecArgs{
				Metadata: &metav1.ObjectMetaArgs{
					Labels: pulumi.StringMap{
						"app": pulumi.String("minio"),
					},
				},
				Spec: &corev1.PodSpecArgs{
					Containers: corev1.ContainerArray{
						&corev1.ContainerArgs{
							Name:  pulumi.String("minio"),
							Image: pulumi.String("minio/minio:latest"),
							Ports: corev1.ContainerPortArray{
								&corev1.ContainerPortArgs{
									ContainerPort: pulumi.Int(9000),
								},
								&corev1.ContainerPortArgs{
									ContainerPort: pulumi.Int(9001),
									Name:          pulumi.String("console"),
								},
							},
							Env: corev1.EnvVarArray{
								&corev1.EnvVarArgs{
									Name:  pulumi.String("MINIO_ROOT_USER"),
									Value: pulumi.String("minioadmin"),
								},
								&corev1.EnvVarArgs{
									Name:  pulumi.String("MINIO_ROOT_PASSWORD"),
									Value: pulumi.String("minioadmin"),
								},
							},
							Command: pulumi.StringArray{
								pulumi.String("minio"),
								pulumi.String("server"),
								pulumi.String("/data"),
								pulumi.String("--console-address"),
								pulumi.String(":9001"),
							},
							VolumeMounts: corev1.VolumeMountArray{
								&corev1.VolumeMountArgs{
									Name:      pulumi.String("minio-storage"),
									MountPath: pulumi.String("/data"),
								},
							},
						},
					},
					Volumes: corev1.VolumeArray{
						&corev1.VolumeArgs{
							Name: pulumi.String("minio-storage"),
							PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSourceArgs{
								ClaimName: minioPVC.Metadata.Name().Elem(),
							},
						},
					},
				},
			},
		},
	}, pulumi.Provider(provider))
	if err != nil {
		return nil, nil, err
	}

	// MinIO Service
	minioService, err := corev1.NewService(ctx, "minio-service", &corev1.ServiceArgs{
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String("minio"),
			Namespace: namespace.Metadata.Name(),
		},
		Spec: &corev1.ServiceSpecArgs{
			Selector: pulumi.StringMap{
				"app": pulumi.String("minio"),
			},
			Ports: corev1.ServicePortArray{
				&corev1.ServicePortArgs{
					Name:       pulumi.String("api"),
					Port:       pulumi.Int(9000),
					TargetPort: pulumi.Int(9000),
				},
				&corev1.ServicePortArgs{
					Name:       pulumi.String("console"),
					Port:       pulumi.Int(9001),
					TargetPort: pulumi.Int(9001),
				},
			},
		},
	}, pulumi.Provider(provider))
	if err != nil {
		return nil, nil, err
	}

	return minioDeployment, minioService, nil
}

func createTEI(ctx *pulumi.Context, config *StackConfig, namespace *corev1.Namespace, provider pulumi.ProviderResource) (*v1.Deployment, *corev1.Service, error) {
	// TEI Deployment
	tei_host := config.Env["TEI_HOST"]
	teiDeployment, err := v1.NewDeployment(ctx, "tei-deployment", &v1.DeploymentArgs{
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String(tei_host),
			Namespace: namespace.Metadata.Name(),
		},
		Spec: &v1.DeploymentSpecArgs{
			Replicas: pulumi.Int(1),
			Selector: &metav1.LabelSelectorArgs{
				MatchLabels: pulumi.StringMap{
					"app": pulumi.String(tei_host),
				},
			},
			Template: &corev1.PodTemplateSpecArgs{
				Metadata: &metav1.ObjectMetaArgs{
					Labels: pulumi.StringMap{
						"app": pulumi.String("tei"),
					},
				},
				Spec: &corev1.PodSpecArgs{
					Containers: corev1.ContainerArray{
						&corev1.ContainerArgs{
							Name:  pulumi.String("tei"),
							Image: pulumi.String("infiniflow/text-embeddings-inference:cpu-1.8"),
							Ports: corev1.ContainerPortArray{
								&corev1.ContainerPortArgs{
									ContainerPort: pulumi.Int(80),
								},
							},
							Args: pulumi.StringArray{
								pulumi.String("--model-id"),
								pulumi.String("/data/" + config.Env["TEI_MODEL"]),
								pulumi.String("--auto-truncate"),
							},
						},
					},
				},
			},
		},
	}, pulumi.Provider(provider))
	if err != nil {
		return nil, nil, err
	}

	// TEI Service
	teiService, err := corev1.NewService(ctx, "tei-service", &corev1.ServiceArgs{
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String(tei_host),
			Namespace: namespace.Metadata.Name(),
		},
		Spec: &corev1.ServiceSpecArgs{
			Selector: pulumi.StringMap{
				"app": pulumi.String(tei_host),
			},
			Ports: corev1.ServicePortArray{
				&corev1.ServicePortArgs{
					Port:       pulumi.Int(80),
					TargetPort: pulumi.Int(80),
				},
			},
		},
	}, pulumi.Provider(provider))
	if err != nil {
		return nil, nil, err
	}

	return teiDeployment, teiService, nil
}

func createRAGFlowDeployment(ctx *pulumi.Context, config *StackConfig, namespace *corev1.Namespace, provider pulumi.ProviderResource) (*v1.Deployment, *corev1.Service, error) {
	// RAGFlow Deployment

	// Build init containers conditionally
	var initContainers corev1.ContainerArray
	if bucket, exists := config.Env["MINIO_BUCKET"]; exists && bucket != "" {
		initContainers = append(initContainers, &corev1.ContainerArgs{
			Name:  pulumi.String("init-minio-bucket"),
			Image: pulumi.String("minio/mc:latest"),
			Env: func() corev1.EnvVarArray {
				var envVars corev1.EnvVarArray
				for key, value := range config.Env {
					envVars = append(envVars, &corev1.EnvVarArgs{
						Name:  pulumi.String(key),
						Value: pulumi.String(value),
					})
				}
				return envVars
			}(),
			Command: pulumi.StringArray{
				pulumi.String("sh"),
				pulumi.String("-c"),
				pulumi.Sprintf(`
					set -e
					echo "Setting MinIO alias..."
					mc alias set myminio http://%s:%s %s %s
					echo "Waiting for MinIO to be ready..."
					while true; do
						if mc ping --count 1 myminio >/dev/null 2>&1; then
							break
						fi
						echo "MinIO not ready, waiting..."
						sleep 5
					done
					echo "Creating MinIO bucket '%s'..."
					mc mb myminio/%s --ignore-existing
					echo "Bucket creation completed."
				`, config.Env["MINIO_HOST"], config.Env["MINIO_PORT"], config.Env["MINIO_USER"], config.Env["MINIO_PASSWORD"], config.Env["MINIO_BUCKET"], config.Env["MINIO_BUCKET"]),
			},
		})
	}

	ragflowDeployment, err := v1.NewDeployment(ctx, "ragflow-deployment", &v1.DeploymentArgs{
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String("ragflow"),
			Namespace: namespace.Metadata.Name(),
		},
		Spec: &v1.DeploymentSpecArgs{
			Replicas: pulumi.Int(1),
			Selector: &metav1.LabelSelectorArgs{
				MatchLabels: pulumi.StringMap{
					"app": pulumi.String("ragflow"),
				},
			},
			Template: &corev1.PodTemplateSpecArgs{
				Metadata: &metav1.ObjectMetaArgs{
					Labels: pulumi.StringMap{
						"app": pulumi.String("ragflow"),
					},
				},
				Spec: &corev1.PodSpecArgs{
					InitContainers: initContainers,
					Containers: corev1.ContainerArray{
						&corev1.ContainerArgs{
							Name:  pulumi.String("ragflow"),
							Image: pulumi.String(config.RAGFlow.Image),
							Ports: corev1.ContainerPortArray{
								&corev1.ContainerPortArgs{
									ContainerPort: pulumi.Int(9380),
								},
								&corev1.ContainerPortArgs{
									ContainerPort: pulumi.Int(9381),
								},
								&corev1.ContainerPortArgs{
									ContainerPort: pulumi.Int(9382),
								},
							},
							Env: func() corev1.EnvVarArray {
								var envVars corev1.EnvVarArray
								for key, value := range config.Env {
									envVars = append(envVars, &corev1.EnvVarArgs{
										Name:  pulumi.String(key),
										Value: pulumi.String(value),
									})
								}
								return envVars
							}(),
							VolumeMounts: corev1.VolumeMountArray{},
							Resources: &corev1.ResourceRequirementsArgs{
								Requests: pulumi.StringMap{
									"memory": pulumi.String("2Gi"),
									"cpu":    pulumi.String("1000m"),
								},
								Limits: pulumi.StringMap{
									"memory": pulumi.String("4Gi"),
									"cpu":    pulumi.String("2000m"),
								},
							},
							Args: pulumi.StringArray{
								pulumi.String("--enable-adminserver"),
							},
						},
					},
					Volumes: corev1.VolumeArray{},
				},
			},
		},
	}, pulumi.Provider(provider))
	if err != nil {
		return nil, nil, err
	}

	// RAGFlow Service
	ragflowService, err := corev1.NewService(ctx, "ragflow-service", &corev1.ServiceArgs{
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String("ragflow"),
			Namespace: namespace.Metadata.Name(),
		},
		Spec: &corev1.ServiceSpecArgs{
			Selector: pulumi.StringMap{
				"app": pulumi.String("ragflow"),
			},
			Ports: corev1.ServicePortArray{
				&corev1.ServicePortArgs{
					Name:       pulumi.String("api"),
					Port:       pulumi.Int(parsePort(config.Env["SVR_HTTP_PORT"])),
					TargetPort: pulumi.Int(9380),
				},
				&corev1.ServicePortArgs{
					Name:       pulumi.String("admin"),
					Port:       pulumi.Int(parsePort(config.Env["ADMIN_SVR_HTTP_PORT"])),
					TargetPort: pulumi.Int(9381),
				},
				&corev1.ServicePortArgs{
					Name:       pulumi.String("mcp"),
					Port:       pulumi.Int(parsePort(config.Env["SVR_MCP_PORT"])),
					TargetPort: pulumi.Int(9382),
				},
			},
			Type: pulumi.String("ClusterIP"),
		},
	}, pulumi.Provider(provider))
	if err != nil {
		return nil, nil, err
	}

	return ragflowDeployment, ragflowService, nil
}

func detectGatewayType(ctx *pulumi.Context, provider pulumi.ProviderResource) (string, error) {
	// Use client-go dynamic client to list GatewayClass resources in the cluster and pick by priority.

	// Build config using standard loading rules
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	configOverrides := &clientcmd.ConfigOverrides{}
	kubeConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, configOverrides)
	cfg, err := kubeConfig.ClientConfig()
	if err != nil {
		ctx.Log.Info("Failed to load kube config", &pulumi.LogArgs{})
		return "", fmt.Errorf("failed to load kube config: %w", err)
	}

	dyn, err := dynamic.NewForConfig(cfg)
	if err != nil {
		ctx.Log.Info("Failed to create dynamic client", &pulumi.LogArgs{})
		return "", fmt.Errorf("failed to create dynamic client: %w", err)
	}

	gvr := schema.GroupVersionResource{Group: "gateway.networking.k8s.io", Version: "v1", Resource: "gatewayclasses"}
	lst, err := dyn.Resource(gvr).List(context.Background(), k8smetav1.ListOptions{})
	if err != nil {
		ctx.Log.Info("Failed to list gatewayclasses", &pulumi.LogArgs{})
		return "", fmt.Errorf("failed to list gatewayclasses: %w", err)
	}

	var names []string
	for _, it := range lst.Items {
		n := it.GetName()
		if n == "" {
			continue
		}
		ctx.Log.Info(fmt.Sprintf("Found GatewayClass: %s", n), &pulumi.LogArgs{})
		// Prefer Cilium
		if strings.Contains(strings.ToLower(n), "cilium") {
			ctx.Log.Info(fmt.Sprintf("Selected GatewayClass (preferred Cilium): %s", n), &pulumi.LogArgs{})
			return n, nil
		}
		names = append(names, n)
	}

	// Otherwise, pick the first GatewayClass in sorted order for deterministic behavior.
	if len(names) > 0 {
		sort.Strings(names)
		ctx.Log.Info(fmt.Sprintf("Selected GatewayClass (first in sorted list): %s", names[0]), &pulumi.LogArgs{})
		return names[0], nil
	}

	ctx.Log.Info("No GatewayClass found in cluster", &pulumi.LogArgs{})
	return "", fmt.Errorf("no GatewayClass found in cluster")
}

// Helper struct for HTTPRoute configuration
type HTTPRouteConfig struct {
	Name        string
	SectionName string
	Port        int
}

func createCR(ctx *pulumi.Context, name, apiVersion, kind string, namespace pulumi.StringPtrInput, spec map[string]interface{}, provider pulumi.ProviderResource) error {
	_, err := apiextensions.NewCustomResource(ctx, name, &apiextensions.CustomResourceArgs{
		ApiVersion: pulumi.String(apiVersion),
		Kind:       pulumi.String(kind),
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String(name),
			Namespace: namespace,
		},
		OtherFields: kubernetes.UntypedArgs{
			"spec": spec,
		},
	}, pulumi.Provider(provider))
	return err
}

func createHTTPRoute(ctx *pulumi.Context, config HTTPRouteConfig, gatewayHost string, serviceName pulumi.StringOutput, httpRouteNsName, gatewayNsName string, provider pulumi.ProviderResource) error {
	//In the Gateway API, an HTTPRoute may only attach to a Gateway listener whose name exactly matches an entry in Gateway.spec.listeners[].name.
	//An HTTPRoute is designed to express routing rules for a single listener context (protocol, port, and semantic purpose), not to multiplex multiple listener endpoints or backend ports.
	//The Gateway API’s separation of concerns:
	// - Gateway / Listener defines where traffic enters.
	// - HTTPRoute defines how traffic is routed once accepted by a listener.
	routeSpec := map[string]interface{}{
		"parentRefs": []interface{}{
			map[string]interface{}{
				"name":        "ragflow-gateway",
				"namespace":   gatewayNsName,
				"sectionName": config.SectionName,
			},
		},
		"rules": []interface{}{
			map[string]interface{}{
				"matches": []interface{}{
					map[string]interface{}{
						"path": map[string]interface{}{"type": "PathPrefix", "value": "/"},
					},
				},
				"backendRefs": []interface{}{
					map[string]interface{}{"kind": "Service", "name": serviceName, "port": config.Port},
				},
			},
		},
	}

	if gatewayHost != "" {
		routeSpec["hostnames"] = []interface{}{gatewayHost}
	}

	routeName := fmt.Sprintf("ragflow-%s-route", config.Name)
	ctx.Log.Info(fmt.Sprintf("Creating %s HTTPRoute resource", config.Name), &pulumi.LogArgs{})
	if err := createCR(ctx, routeName, "gateway.networking.k8s.io/v1", "HTTPRoute", pulumi.String(httpRouteNsName), routeSpec, provider); err != nil {
		ctx.Log.Error(fmt.Sprintf("Failed to create %s HTTPRoute: %v", config.Name, err), &pulumi.LogArgs{})
		return err
	}
	ctx.Log.Info(fmt.Sprintf("%s HTTPRoute resource created successfully", strings.Title(config.Name)), &pulumi.LogArgs{})
	return nil
}

func createPathBasedHTTPRoute(ctx *pulumi.Context, gatewayHost string, serviceName pulumi.StringOutput, gatewayNsName, httpRouteNsName string, provider pulumi.ProviderResource) error {
	// Create HTTPRoute with path-based routing rules based on docker/nginx/ragflow.conf
	routeSpec := map[string]interface{}{
		"parentRefs": []interface{}{
			map[string]interface{}{
				"name":        "ragflow-gateway",
				"namespace":   gatewayNsName,
				"sectionName": "http",
			},
		},
		"rules": []interface{}{
			// Rule 1: /v1 or /api -> port 9380 (API service)
			map[string]interface{}{
				"matches": []interface{}{
					map[string]interface{}{
						"path": map[string]interface{}{
							"type":  "PathPrefix",
							"value": "/v1",
						},
					},
					map[string]interface{}{
						"path": map[string]interface{}{
							"type":  "PathPrefix",
							"value": "/api",
						},
					},
				},
				"backendRefs": []interface{}{
					map[string]interface{}{
						"kind": "Service",
						"name": serviceName,
						"port": 9380,
					},
				},
			},
			// Rule 2: /api/v1/admin -> port 9381 (admin service)
			map[string]interface{}{
				"matches": []interface{}{
					map[string]interface{}{
						"path": map[string]interface{}{
							"type":  "PathPrefix",
							"value": "/api/v1/admin",
						},
					},
				},
				"backendRefs": []interface{}{
					map[string]interface{}{
						"kind": "Service",
						"name": serviceName,
						"port": 9381,
					},
				},
			},
		},
	}

	ctx.Log.Info("Creating path-based HTTPRoute resource", &pulumi.LogArgs{})
	if err := createCR(ctx, "ragflow-http-route", "gateway.networking.k8s.io/v1", "HTTPRoute", pulumi.String(httpRouteNsName), routeSpec, provider); err != nil {
		ctx.Log.Error(fmt.Sprintf("Failed to create path-based HTTPRoute: %v", err), &pulumi.LogArgs{})
		return err
	}
	ctx.Log.Info("Path-based HTTPRoute resource created successfully", &pulumi.LogArgs{})
	return nil
}

// createGateway now expresses resources as compact specs and calls createCR to register them.
func createGateway(ctx *pulumi.Context, config *StackConfig, provider pulumi.ProviderResource, ragflowService *corev1.Service, gatewayClass string) (*apiextensions.CustomResource, error) {
	// Create Gateway in configured namespace (default: nginx-gateway)
	gatewayNsName := config.Gateway.Namespace

	ctx.Log.Info(fmt.Sprintf("Creating Gateway with GatewayClass: %s", gatewayClass), &pulumi.LogArgs{})

	gatewaySpec := map[string]interface{}{
		"gatewayClassName": gatewayClass,
		"listeners": []interface{}{
			map[string]interface{}{
				"name":     "http",
				"port":     80,
				"protocol": "HTTP",
				"allowedRoutes": map[string]interface{}{
					"namespaces": map[string]interface{}{
						"from": "All",
					},
				},
			},
		},
	}
	if config.Gateway.Host != "" {
		listeners := gatewaySpec["listeners"].([]interface{})
		for i := range listeners {
			listeners[i].(map[string]interface{})["hostname"] = config.Gateway.Host
		}
		ctx.Log.Info(fmt.Sprintf("Gateway hostname set to: %s for all listeners", config.Gateway.Host), &pulumi.LogArgs{})
	}

	ctx.Log.Info("Creating Gateway resource", &pulumi.LogArgs{})
	gateway, err := apiextensions.NewCustomResource(ctx, "ragflow-gateway", &apiextensions.CustomResourceArgs{
		ApiVersion: pulumi.String("gateway.networking.k8s.io/v1"),
		Kind:       pulumi.String("Gateway"),
		Metadata: &metav1.ObjectMetaArgs{
			Name:      pulumi.String("ragflow-gateway"),
			Namespace: pulumi.String(gatewayNsName),
		},
		OtherFields: kubernetes.UntypedArgs{
			"spec": gatewaySpec,
		},
	}, pulumi.Provider(provider))
	if err != nil {
		ctx.Log.Error(fmt.Sprintf("Failed to create Gateway: %v", err), &pulumi.LogArgs{})
		return nil, err
	}
	ctx.Log.Info("Gateway resource created successfully", &pulumi.LogArgs{})

	// Create a single HTTPRoute with path-based routing rules
	if err := createPathBasedHTTPRoute(ctx, config.Gateway.Host, ragflowService.Metadata.Name().Elem(), gatewayNsName, config.Namespace, provider); err != nil {
		return nil, err
	}

	return gateway, nil
}
