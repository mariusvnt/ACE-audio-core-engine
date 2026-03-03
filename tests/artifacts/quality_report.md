# ACE Test Quality Report

- Generated (UTC): `2026-03-01T15:41:10Z`
- Overall status: `fail`
- Integrity gate: `pass`
- Performance gate: `fail`
- Integration gate: `pass`

## Environment

- Python: `3.13.12`
- Platform: `Windows-11-10.0.22631-SP0`
- NumPy available: `True`
- bpy available: `True`

## Category Summary

| Category | Status | Total | Executed | Passed | Failed | Error | Skipped | Pass Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| integrity | pass | 124 | 123 | 123 | 0 | 0 | 1 | 100.0% |
| performance | fail | 5 | 5 | 3 | 2 | 0 | 0 | 60.0% |
| integration | pass | 2 | 1 | 1 | 0 | 0 | 1 | 100.0% |

## Performance Harness

- Harness status: `fail`
- Baseline report: `C:\Users\mariu\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons\AudioCore Engine\tests\artifacts\perf_baseline_current.json`
- Engine report: `C:\Users\mariu\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons\AudioCore Engine\tests\artifacts\engine_perf_current.json`
- Baseline stage peak p95 (ms): `{'source_read': 0.222185, 'dsp': 17.914505, 'preview': 3.33378, 'proxy_write': 0.016405, 'ui_sync': 0.001205}`
- Engine key metrics: `ratio=1.538, queue_drop=0.292, fresh_min=0.117, sla_p95_ms=14.394`
- Performance violations:
  - engine scenario 24: dsp ratio 1.465 > 1.100
  - engine scenario 24: fresh-result rate 0.117 < 0.900
  - engine scenario 36: dsp ratio 1.538 > 1.100
  - engine scenario 36: queue-drop rate 0.067 > 0.030
  - engine scenario 36: fresh-result rate 0.214 < 0.900
  - engine scenario 50: dsp ratio 1.441 > 1.100
  - engine scenario 50: queue-drop rate 0.292 > 0.030
  - engine scenario 50: fresh-result rate 0.341 < 0.900

## Failing Tests

- `test_engine_perf_gate.TestEnginePerfGate.test_engine_perf_strict_gate` (failed)
  - AssertionError: 1.5819442597429412 not less than or equal to 1.1 : Engine DSP strict gate failed sid=24: processes/threads dsp p95 ratio=1.582 > 1.100
- `test_perf_gate.TestPerfGate.test_perf_gate_warning_mode_or_strict_threshold` (failed)
  - AssertionError: 0.4032250348245722 not less than or equal to 0.29931525030406186 : Perf regression sid=1 stage=dsp: p95=0.403ms baseline=0.222ms allowed=0.299ms

## Slowest Tests

- `test_perf_gate.TestPerfGate.test_perf_harness_generates_required_scenarios` [performance | passed] 10.193s
- `test_perf_gate.TestPerfGate.test_perf_gate_warning_mode_or_strict_threshold` [performance | failed] 10.145s
- `test_audio_tasks_registry_smoke.TestAudioTasksRegistrySmoke.test_imports_with_fake_numpy_module` [integration | passed] 0.006s
- `test_audio_tasks_registry.TestAudioTasksRegistry.test_preview_sink_sparse_indices_uses_masked_fft` [integrity | passed] 0.004s
- `test_audio_tasks_registry.TestAudioTasksRegistry.test_splitter_then_chromagram_preview_respects_parent_band_with_variable_chunk_size` [integrity | passed] 0.003s
- `test_regression_findings.TestRegressionFindings.test_auto_attr_ids_reinitialize_on_reconnect` [integrity | passed] 0.002s
- `test_regression_findings.TestRegressionFindings.test_handlers_are_persistent` [integrity | passed] 0.001s
- `test_engine_perf_gate.TestEnginePerfGate.test_engine_perf_strict_gate` [performance | failed] 0.001s
- `test_regression_findings.TestRegressionFindings.test_file_source_description_uses_original_sample_rate_metadata` [integrity | passed] 0.001s
- `test_audio_tasks_registry.TestAudioTasksRegistry.test_chromagram_fallback_keeps_sparse_mask_non_empty_in_coarse_band` [integrity | passed] 0.001s
