# Isaac experiment job archive — 2026-10-27

RIFT and FastWAM RoboTwin C2R training and evaluation records from the shared `/scratch3/ton068/datasets/isaac/jobs/` directory are preserved in the public Hugging Face archive below. This archive contains multiple research projects; use the directory index to locate an original job name.

- **Archive:** [https://huggingface.co/datasets/research-vla/isaac-jobs](https://huggingface.co/datasets/research-vla/isaac-jobs)
- **Verified snapshot:** [`0fed75ebd6182e074fa1bd18403eed1c090af4ff`](https://huggingface.co/datasets/research-vla/isaac-jobs/tree/0fed75ebd6182e074fa1bd18403eed1c090af4ff)
- **Directory index:** [JOB_INDEX.md](https://huggingface.co/datasets/research-vla/isaac-jobs/blob/0fed75ebd6182e074fa1bd18403eed1c090af4ff/JOB_INDEX.md)
- **Completion receipt:** [backup-status.json](https://huggingface.co/datasets/research-vla/isaac-jobs/resolve/0fed75ebd6182e074fa1bd18403eed1c090af4ff/backup-status.json)
- **Content verification:** [verification.json](https://huggingface.co/datasets/research-vla/isaac-jobs/resolve/0fed75ebd6182e074fa1bd18403eed1c090af4ff/verification.json)
- **Download checksums:** [SHA256SUMS](https://huggingface.co/datasets/research-vla/isaac-jobs/resolve/0fed75ebd6182e074fa1bd18403eed1c090af4ff/SHA256SUMS)

The verified snapshot contains **84,400 regular files, 15,375 directories and 249 symbolic links**. The repository holds about **255.93 GB** of compressed data and metadata: one text-record archive and 30 binary archive parts. Original paths, hard links and symbolic-link targets are recorded in the manifests. Every archived file's size and SHA256, all 30 part hashes, and the compressed stream checksum passed verification. Remote object hashes and sizes were also checked.

## Find and restore a record

Use the directory index to find the original `jobs/<job-name>/...` path. Scripts, configuration, text logs and JSON/CSV results are in `jobs-text-records-20261027.tar.zst`; other files, including weights and caches, are in the binary stream. The two manifests identify each file's archive and SHA256.

Download the pinned dataset revision with the Hugging Face CLI:

```bash
hf download research-vla/isaac-jobs --repo-type dataset --revision 0fed75ebd6182e074fa1bd18403eed1c090af4ff --local-dir isaac-jobs-archive
cd isaac-jobs-archive
sha256sum -c SHA256SUMS
mkdir restored
tar --zstd -xf jobs-text-records-20261027.tar.zst -C restored
cat binary/jobs-binary-20261027.tar.zst.part-* | zstd -d | tar -xf - -C restored
```

All binary parts must be downloaded and concatenated in filename order. Restore into a new directory before selecting files to reuse. Symbolic links may refer to the original HPC environment; their external target data was not copied and may require separate restoration or path changes.

## Local retention and snapshot scope

The user authorized removal of local archived records after successful upload and verification. Local cleanup is restricted to unchanged files that match the verified snapshot. Running or queued jobs, their dependencies, and files created or changed after the inventory remain local. The original `jobs` location is therefore no longer a reliable historical archive; use the pinned URL above for historical records.

This is a manifest-based snapshot collected while experiments were running, not a continuous mirror or an atomic snapshot of all tasks. Files created after the inventory are outside this revision. The archive is public; existing source/model licenses remain applicable.
