#!/bin/bash
# OOM-Schutz: Prod-Inferenz-Prozesse fuer den Kernel-OOM-Killer unattraktiv
# machen (Ray setzt +1000 -> Worker stirbt SONST ZUERST, Vorfall 2026-06-10).
# Kills sind damit nicht abgeschafft, aber treffen zuerst Entbehrliches.
for pat in "VLLM::EngineCore" "ray::RayWorkerWrapper" "vllm serve" "raylet" "gcs_server"; do
  for pid in $(pgrep -f "$pat"); do
    echo -500 > /proc/$pid/oom_score_adj 2>/dev/null
  done
done
