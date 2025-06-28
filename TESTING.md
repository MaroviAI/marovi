# Running Tests

This project uses **pytest** for unit tests. You can run the entire suite with

```bash
make test
```

or directly with Poetry:

```bash
poetry run pytest
```

We considered Bazel for test orchestration, but the current codebase is pure
Python and has a straightforward structure. Using Bazel would add unnecessary
complexity without clear benefits at this stage. Pytest provides a simple and
robust solution for now.
