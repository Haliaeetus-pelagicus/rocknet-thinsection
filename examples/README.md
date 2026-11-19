# Example

From the repository root, after `pip install -r requirements.txt`:

```bash
python -m rocknet examples/sample_tile.jpg --topk 3 --output examples/sample_report.json
```

The command reads `examples/sample_tile.jpg`, loads the split round-8 checkpoint, and writes the identification JSON. A non-zero exit status means the checkpoint parts are missing or their SHA-256 digest does not match `weights/SHA256.txt`.
