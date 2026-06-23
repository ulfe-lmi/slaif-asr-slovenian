# FLEURS v1 Gate Deprecation

Status: **deprecated; unsuitable for complete-split quality claims**

The historical gate `fleurs-sl-si-test-full-v1` is preserved for auditability,
but it must not be used as complete FLEURS test-split evidence.

## Reason

The v1 builder used the FLEURS source `id` field to construct both
`sample_id` and the WAV filename. That source field is not unique per test-set
audio occurrence. Repeated source IDs therefore caused later rows to reuse the
same manifest identity and overwrite earlier WAV files.

The v1 manifest had 834 rows, but those rows represented only 347 unique sample
IDs and audio paths. As a result, v1 FLEURS aggregate metrics are invalid as
complete-split quality evidence.

## Scope

- Historical v1 metadata and experiment records remain available only for
  auditability.
- Historical v1 manifest hashes and metrics are not silently rewritten.
- ARTUR-J measurements are unaffected.
- The canonical complete FLEURS development gate is
  `fleurs-sl-si-test-full-v2`, which uses occurrence-unique row-index-based
  sample IDs and WAV filenames.
