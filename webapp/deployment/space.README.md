---
title: Data Splitting Lab
emoji: 🧩
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

An interactive app for comparing DataSAIL, hypergraph and low-rank dataset splits.
Create the Space with **private** visibility; the owner and explicitly added
collaborators can access it. Visibility is an account setting, not this README.

Docker Spaces currently require an eligible paid account plan even when using
CPU Basic hardware. This bundle never requests a hardware upgrade.

The CPU profile queues one compute job at a time, persists jobs and caches under
`/data`, and caps uploads at 25 MB. Protein/transformer embeddings remain
expensive on CPU; feature choices are preserved and no GPU is automatically used.

Use a persistent volume or appropriate storage integration for retained uploads
and model caches. Standard ephemeral container storage does not survive rebuilds.
