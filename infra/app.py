#!/usr/bin/env python3
"""
CDK App entry point – Document Intelligence MCP
================================================
Usage:
    cd infra
    pip install -r requirements-cdk.txt
    cdk bootstrap aws://<ACCOUNT_ID>/<REGION>
    cdk deploy
"""

import aws_cdk as cdk
from stacks.mcp_stack import McpDocumentIntelligenceStack

app = cdk.App()

McpDocumentIntelligenceStack(
    app,
    "McpDocumentIntelligenceStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1",
    ),
    description="MCP Document Intelligence – S3, OpenSearch Serverless, Bedrock KB",
)

app.synth()
