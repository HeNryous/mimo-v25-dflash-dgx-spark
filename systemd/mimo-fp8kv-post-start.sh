#!/bin/bash
# After prod (re)start: re-trigger the spark2 gate that starts neo4j once vLLM is healthy.
ssh -o BatchMode=yes -o StrictHostKeyChecking=no 10.0.0.2 "sudo systemctl restart --no-block vllm-dependents" || true
