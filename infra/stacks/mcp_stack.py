"""
CDK Stack – Document Intelligence MCP Infrastructure
=====================================================
Provisions:
  • S3 bucket          – raw document storage
  • OpenSearch Serverless collection – vector store for embeddings
  • Bedrock Knowledge Base           – RAG retrieval
  • IAM roles                        – least-privilege access
  • SSM Parameters                   – runtime config for the MCP server

Deploy:
    cd infra
    cdk bootstrap
    cdk deploy
"""

from __future__ import annotations

import json

import aws_cdk as cdk
from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_iam as iam
from aws_cdk import aws_opensearchserverless as aoss
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_ssm as ssm
from aws_cdk import aws_bedrock as bedrock
from constructs import Construct


class McpDocumentIntelligenceStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── 1. S3 – Raw document storage ──────────────────────────────────────
        raw_docs_bucket = s3.Bucket(
            self,
            "RawDocsBucket",
            bucket_name=f"mcp-raw-docs-{self.account}-{self.region}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="MoveToIA",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30),
                        )
                    ],
                )
            ],
        )

        # ── 2. OpenSearch Serverless – Vector store ───────────────────────────
        collection_name = "mcp-doc-vectors"

        # Encryption policy
        encryption_policy = aoss.CfnSecurityPolicy(
            self,
            "OSSEncryptionPolicy",
            name=f"{collection_name}-enc",
            type="encryption",
            policy=json.dumps(
                {
                    "Rules": [
                        {
                            "ResourceType": "collection",
                            "Resource": [f"collection/{collection_name}"],
                        }
                    ],
                    "AWSOwnedKey": True,
                }
            ),
        )

        # Network policy – allow public access (restrict to VPC in production)
        network_policy = aoss.CfnSecurityPolicy(
            self,
            "OSSNetworkPolicy",
            name=f"{collection_name}-net",
            type="network",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "collection",
                                "Resource": [f"collection/{collection_name}"],
                            },
                            {
                                "ResourceType": "dashboard",
                                "Resource": [f"collection/{collection_name}"],
                            },
                        ],
                        "AllowFromPublic": True,
                    }
                ]
            ),
        )

        # OpenSearch Serverless collection
        oss_collection = aoss.CfnCollection(
            self,
            "OSSCollection",
            name=collection_name,
            type="VECTORSEARCH",
            description="Vector store for MCP document intelligence",
        )
        oss_collection.add_dependency(encryption_policy)
        oss_collection.add_dependency(network_policy)

        # ── 3. IAM Role – MCP Server / Application ────────────────────────────
        mcp_role = iam.Role(
            self,
            "McpServerRole",
            role_name="McpDocumentIntelligenceRole",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("lambda.amazonaws.com"),
                iam.AccountPrincipal(self.account),  # allow local dev assume-role
            ),
            description="Role used by the MCP document intelligence server",
        )

        # S3 permissions
        raw_docs_bucket.grant_read_write(mcp_role)

        # Bedrock permissions
        mcp_role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockInvoke",
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
                    f"arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-text-v2:0",
                ],
            )
        )

        # Bedrock Knowledge Base – retrieve
        mcp_role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockKBRetrieve",
                actions=[
                    "bedrock:Retrieve",
                    "bedrock:RetrieveAndGenerate",
                ],
                resources=["*"],  # narrowed after KB is created
            )
        )

        # OpenSearch Serverless permissions
        mcp_role.add_to_policy(
            iam.PolicyStatement(
                sid="OSSAccess",
                actions=["aoss:APIAccessAll"],
                resources=[oss_collection.attr_arn],
            )
        )

        # Data access policy for OpenSearch Serverless
        aoss.CfnAccessPolicy(
            self,
            "OSSDataAccessPolicy",
            name=f"{collection_name}-data",
            type="data",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "index",
                                "Resource": [f"index/{collection_name}/*"],
                                "Permission": [
                                    "aoss:CreateIndex",
                                    "aoss:DeleteIndex",
                                    "aoss:UpdateIndex",
                                    "aoss:DescribeIndex",
                                    "aoss:ReadDocument",
                                    "aoss:WriteDocument",
                                ],
                            },
                            {
                                "ResourceType": "collection",
                                "Resource": [f"collection/{collection_name}"],
                                "Permission": ["aoss:CreateCollectionItems"],
                            },
                        ],
                        "Principal": [mcp_role.role_arn],
                    }
                ]
            ),
        )

        # ── 4. Bedrock Knowledge Base ─────────────────────────────────────────
        # IAM role for Bedrock KB to access S3 and OpenSearch
        kb_role = iam.Role(
            self,
            "BedrockKBRole",
            role_name="BedrockKnowledgeBaseRole",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
        )
        raw_docs_bucket.grant_read(kb_role)
        kb_role.add_to_policy(
            iam.PolicyStatement(
                actions=["aoss:APIAccessAll"],
                resources=[oss_collection.attr_arn],
            )
        )
        kb_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-text-v2:0"
                ],
            )
        )

        # Data access for KB role
        aoss.CfnAccessPolicy(
            self,
            "OSSKBDataAccessPolicy",
            name=f"{collection_name}-kb-data",
            type="data",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "index",
                                "Resource": [f"index/{collection_name}/*"],
                                "Permission": [
                                    "aoss:CreateIndex",
                                    "aoss:DescribeIndex",
                                    "aoss:ReadDocument",
                                    "aoss:WriteDocument",
                                    "aoss:UpdateIndex",
                                    "aoss:DeleteIndex",
                                ],
                            },
                            {
                                "ResourceType": "collection",
                                "Resource": [f"collection/{collection_name}"],
                                "Permission": ["aoss:CreateCollectionItems"],
                            },
                        ],
                        "Principal": [kb_role.role_arn],
                    }
                ]
            ),
        )

        knowledge_base = bedrock.CfnKnowledgeBase(
            self,
            "BedrockKnowledgeBase",
            name="mcp-document-kb",
            description="Knowledge base for MCP document intelligence RAG",
            role_arn=kb_role.role_arn,
            knowledge_base_configuration=bedrock.CfnKnowledgeBase.KnowledgeBaseConfigurationProperty(
                type="VECTOR",
                vector_knowledge_base_configuration=bedrock.CfnKnowledgeBase.VectorKnowledgeBaseConfigurationProperty(
                    embedding_model_arn=f"arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-text-v2:0"
                ),
            ),
            storage_configuration=bedrock.CfnKnowledgeBase.StorageConfigurationProperty(
                type="OPENSEARCH_SERVERLESS",
                opensearch_serverless_configuration=bedrock.CfnKnowledgeBase.OpenSearchServerlessConfigurationProperty(
                    collection_arn=oss_collection.attr_arn,
                    vector_index_name="mcp-documents",
                    field_mapping=bedrock.CfnKnowledgeBase.OpenSearchServerlessFieldMappingProperty(
                        vector_field="embedding",
                        text_field="text",
                        metadata_field="metadata",
                    ),
                ),
            ),
        )
        knowledge_base.add_dependency(oss_collection)

        # S3 data source for the Knowledge Base
        bedrock.CfnDataSource(
            self,
            "KBS3DataSource",
            name="mcp-s3-datasource",
            knowledge_base_id=knowledge_base.attr_knowledge_base_id,
            data_source_configuration=bedrock.CfnDataSource.DataSourceConfigurationProperty(
                type="S3",
                s3_configuration=bedrock.CfnDataSource.S3DataSourceConfigurationProperty(
                    bucket_arn=raw_docs_bucket.bucket_arn,
                    inclusion_prefixes=["raw-docs/"],
                ),
            ),
            vector_ingestion_configuration=bedrock.CfnDataSource.VectorIngestionConfigurationProperty(
                chunking_configuration=bedrock.CfnDataSource.ChunkingConfigurationProperty(
                    chunking_strategy="FIXED_SIZE",
                    fixed_size_chunking_configuration=bedrock.CfnDataSource.FixedSizeChunkingConfigurationProperty(
                        max_tokens=1000,
                        overlap_percentage=20,
                    ),
                )
            ),
        )

        # ── 5. SSM Parameters – runtime config ───────────────────────────────
        ssm.StringParameter(
            self,
            "ParamS3Bucket",
            parameter_name="/mcp/S3_RAW_BUCKET",
            string_value=raw_docs_bucket.bucket_name,
        )
        ssm.StringParameter(
            self,
            "ParamOSSEndpoint",
            parameter_name="/mcp/OPENSEARCH_ENDPOINT",
            string_value=cdk.Fn.select(
                1, cdk.Fn.split("//", oss_collection.attr_collection_endpoint)
            ),
        )
        ssm.StringParameter(
            self,
            "ParamKBId",
            parameter_name="/mcp/KNOWLEDGE_BASE_ID",
            string_value=knowledge_base.attr_knowledge_base_id,
        )

        # ── 6. CloudFormation Outputs ─────────────────────────────────────────
        CfnOutput(self, "RawDocsBucketName", value=raw_docs_bucket.bucket_name)
        CfnOutput(self, "OSSCollectionEndpoint", value=oss_collection.attr_collection_endpoint)
        CfnOutput(self, "OSSCollectionArn", value=oss_collection.attr_arn)
        CfnOutput(self, "KnowledgeBaseId", value=knowledge_base.attr_knowledge_base_id)
        CfnOutput(self, "McpServerRoleArn", value=mcp_role.role_arn)
