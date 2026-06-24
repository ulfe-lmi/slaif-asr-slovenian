# Corpus-v2 Selected Training

Status: `SELECTED_TRAINING_MANIFEST_READY`

This privacy-safe report records selected-training construction from the accepted candidate source only. It does not authorize model training and does not issue `TRAINING_ELIGIBLE`.

## Selection

- Total selected rows: 160
- Hard examples: 120
- Controls: 40
- Selected manifest SHA256: `84e10587af184be92571ab84e3bd58cd676866e2bd944534c759f0fc9a07fa13`
- Selected audio manifest SHA256: `4fe8ab008dd9725c65da510ed801a46299e1c03db0c00cb3fbf5dea40ff0be7b`
- Holdout exclusion: `{"selected_holdout_audio_hash_overlaps": 0, "selected_holdout_id_overlaps": 0, "selected_holdout_text_hash_overlaps": 0}`

## Constraints

- Selection policy: `corpus-v2-selected-training-policy-v1`
- Relaxation attempts: `[{"reason": "strict", "relax_cell_minimum": false, "relax_discovered_family_cap": false, "relax_domain_cap": false, "relax_source_family_cap": false, "selected_count": 72}, {"reason": "domain_cap_relaxed", "relax_cell_minimum": false, "relax_discovered_family_cap": false, "relax_domain_cap": true, "relax_source_family_cap": false, "selected_count": 72}, {"reason": "source_family_cap_relaxed", "relax_cell_minimum": false, "relax_discovered_family_cap": false, "relax_domain_cap": true, "relax_source_family_cap": true, "selected_count": 120}]`

## Limitations

- Selected training is single-voice synthetic Piper audio.
- Selected training is not real-speech evidence.
- Training remains unauthorized until a later training work order.
- The untouched Nemotron base remains the only accepted parent.
