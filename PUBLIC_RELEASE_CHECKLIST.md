# Public release checklist

Complete these items before making the GitHub repository public.

- [ ] Add the final author names and affiliations.
- [ ] Add the paper DOI, journal citation, and `CITATION.cff`.
- [ ] Select an open-source license and add a `LICENSE` file.
- [x] Replace the placeholder clone URL with the selected GitHub repository.
- [ ] Confirm that the station parameters and default walking times may be
      redistributed.
- [ ] Confirm that no restricted OSM-derived data, API keys, local paths, or
      participant-level data are included.
- [x] Separate appendix-derived mechanisms from newly added analysis modules in
      the documentation.
- [ ] Compare the added objective and intervention-cost definitions with the
      final manuscript and original analysis scripts.
- [ ] Validate shuttle, event, and joint scenario parameters against the final
      experiment configurations.
- [x] Export cross-validated surrogate diagnostics.
- [x] Export an independent-seed simulation check for selected Pareto solutions.
- [ ] Run `pytest` and `ruff check .` from a clean environment.
- [ ] Run the documented 30-replication baseline and compare its metadata with
      `examples/baseline_run_metadata.json`.
- [ ] Decide whether validated OSM-derived JSON may be published; if so, record
      snapshot date, routing profile, CRS, and attribution.
- [ ] Create a GitHub release and archive the release on Zenodo if a permanent
      software DOI is required.
