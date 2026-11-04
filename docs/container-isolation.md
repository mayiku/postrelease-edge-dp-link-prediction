# Container Isolation

`Dockerfile.train` builds a trainer image containing only:

- `scripts/train_postrelease.py`;
- `src/artifacts.py`;
- `src/postrelease_core.py`;
- the minimal NumPy and Torch dependencies.

The image does not contain:

- `src/postrelease_lp.py`;
- raw graph loading code;
- `release_graph.py`;
- raw dataset files.

## Run

Linux/macOS:

```bash
docker build -f Dockerfile.train -t postrelease-train .
docker run --rm \
  -v "$PWD/outputs/release:/artifact:ro" \
  -v "$PWD/outputs/train:/output" \
  postrelease-train \
  --artifact-dir /artifact \
  --output /output/scores.npz
```

PowerShell:

```powershell
docker build -f Dockerfile.train -t postrelease-train .
docker run --rm `
  -v "${PWD}\outputs\release:/artifact:ro" `
  -v "${PWD}\outputs\train:/output" `
  postrelease-train `
  --artifact-dir /artifact `
  --output /output/scores.npz
```

The artifact mount is read-only. The training image has no raw-data
argument and no raw-graph module.

CI builds the image and asserts that the raw graph modules do not exist
inside it.
