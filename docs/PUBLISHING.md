# Repository release checks

Use the following checks before creating a release or manuscript archive.

## Code and tests

```bash
python -m pip install -e ".[dev,analysis]"
pytest
ruff check .
```

Confirm that the documented reference commands run with
`configs/reference_calibration.json` and that generated tables agree with
`docs/MANUSCRIPT_RESULTS.md`.

## Research materials

- Record the configuration file, seed ranges, and package version.
- Keep training, external assessment, and verification seeds separate.
- Regenerate manuscript figures with `scripts/generate_manuscript_figures.py`.
- Check that numerical claims can be traced to CSV or JSON output.
- Do not commit credentials, local paths, restricted data, or undocumented OSM
  extracts.

## GitHub release

- Review the staged diff and commit only files needed for the release.
- Push the tested commit to `main`.
- Confirm that the GitHub Actions workflow passes.
- Create a version tag only after the remote commit and documentation have been
  checked.
