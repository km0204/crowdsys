# Contributing

Contributions are welcome after the repository license has been selected.

1. Create a branch from `main`.
2. Add tests for behavioural or numerical changes.
3. Run `pytest` and `ruff check .`.
4. Explain any change to the model equations, event order, random-number use,
   or output definitions in the pull request.

Changes that alter published baseline parameters should use a new configuration
file rather than modifying `configs/baseline.json`.

