# Data-Quality Fixtures

The data-quality tests build their fixture JSONL files in temporary directories
so the repository does not commit generated corpora or protected references.
This directory exists to reserve the fixture namespace required by the
training-data validator work order.
