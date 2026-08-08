# CPU-only Docker CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, smoke-test, and publish only the CPU Docker image in GitHub Actions while preserving optional GPU build assets for manual user builds.

**Architecture:** Replace both Docker workflow matrices with single CPU job settings. The smoke image remains tagged and cache-scoped as `cpu`; the publish image uses the existing unsuffixed GHCR name. Update the workflow contract tests to reject GPU Dockerfile references in CI while retaining independent static tests for the optional GPU Dockerfile.

**Tech Stack:** GitHub Actions YAML, Docker Buildx, pytest, uv.

## Global Constraints

- Keep `Dockerfile-Gpu` and `docker/gpu/` in the repository for manual builds.
- CI must reference only `Dockerfile`, `linux/amd64,linux/arm64`, and CPU Torch (`torch.version.cuda is None`).
- Preserve CPU image cache, smoke test, tags, and GHCR publication.

---

### Task 1: Make the Docker workflow CPU-only

**Files:**
- Modify: `tests/test_dockerfiles.py:113-156`
- Modify: `.github/workflows/docker.yml:21-112`

**Interfaces:**
- Consumes: Docker workflow with `smoke` and `build` jobs.
- Produces: CPU-only workflow with no `Dockerfile-Gpu`, `variant: gpu`, CUDA assertion, or `-gpu` image suffix.

- [x] **Step 1: Write the failing test**

```python
def test_ci_builds_and_publishes_only_the_cpu_image():
    workflow = Path(".github/workflows/docker.yml").read_text(encoding="utf-8")
    assert "dockerfile: Dockerfile" in workflow
    assert "torch.version.cuda is None" in workflow
    assert "Dockerfile-Gpu" not in workflow
    assert "torch.version.cuda is not None" not in workflow
    assert "image_suffix" not in workflow
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dockerfiles.py::test_ci_builds_and_publishes_only_the_cpu_image -q`

Expected: FAIL because the workflow currently contains a GPU matrix entry and CUDA smoke assertion.

- [x] **Step 3: Write minimal implementation**

Replace each matrix with direct CPU values. Reference `Dockerfile`, cache scope `cpu`, `jellysub-ai-smoke:cpu`, CPU CUDA assertion, multi-architecture CPU platforms, and unsuffixed GHCR image metadata.

- [x] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dockerfiles.py -q`

- [ ] **Step 5: Commit**

Run: `git add .github/workflows/docker.yml tests/test_dockerfiles.py && git commit -m "ci: build only CPU Docker image"`
