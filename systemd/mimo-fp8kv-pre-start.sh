#!/bin/bash
# Runs as admin user (from systemd service User=admin)
set -e

# Kill leftover containers
docker rm -f vllm_node 2>/dev/null || true
ssh -o BatchMode=yes -o StrictHostKeyChecking=no 10.0.0.2 "docker rm -f vllm_node 2>/dev/null || true" || true


# Kill guard-margin eaters (night-2026-07-10 hardening):
# hermes extract crons (memory-capped, kill-safe, re-run every 30m) ~3.2G
pkill -9 -f "[.]hermes/scripts/ex" 2>/dev/null || true
# ray leftovers from crashed boots
sudo pkill -9 -x raylet 2>/dev/null || true; sudo pkill -9 -x gcs_server 2>/dev/null || true
sudo pkill -9 -f "[r]ay start --block" 2>/dev/null || true
ssh -o BatchMode=yes -o StrictHostKeyChecking=no 10.0.0.2 "sudo pkill -9 -x raylet 2>/dev/null; sudo pkill -9 -x gcs_server 2>/dev/null; sudo pkill -9 -f \"[r]ay start --block\" 2>/dev/null" || true
# neo4j pauses during memory profiling; vllm-dependents gate restarts it after vLLM health
ssh -o BatchMode=yes -o StrictHostKeyChecking=no 10.0.0.2 "docker stop neo4j 2>/dev/null" || true
# Wait for processes to release UVM handles
sleep 3

# Reset nvidia_uvm to free leaked CUDA driver memory (needs sudo)
sudo modprobe -r nvidia_uvm 2>/dev/null && sudo modprobe nvidia_uvm 2>/dev/null || true
ssh -o BatchMode=yes -o StrictHostKeyChecking=no 10.0.0.2 \
  "sudo modprobe -r nvidia_uvm 2>/dev/null && sudo modprobe nvidia_uvm 2>/dev/null || true" || true

# Drop page caches (needs sudo for tee to /proc)
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
ssh -o BatchMode=yes -o StrictHostKeyChecking=no 10.0.0.2 \
  "echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null" || true

echo "mimo-pre-start: cleanup done"
