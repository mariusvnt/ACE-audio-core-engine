# ACE Test Quality Report

- Generated (UTC): `2026-03-19T23:00:34Z`
- Overall status: `fail`
- Integrity gate: `fail`
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
| integrity | fail | 161 | 160 | 154 | 2 | 4 | 1 | 96.2% |
| performance | fail | 5 | 5 | 3 | 2 | 0 | 0 | 60.0% |
| integration | pass | 2 | 1 | 1 | 0 | 0 | 1 | 100.0% |

## Performance Harness

- Harness status: `fail`
- Baseline report: `C:\Users\mariu\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons\AudioCore Engine\tests\artifacts\perf_baseline_current.json`
- Engine report: `C:\Users\mariu\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons\AudioCore Engine\tests\artifacts\engine_perf_current.json`
- Baseline stage peak p95 (ms): `{'source_read': 0.24589, 'dsp': 5.083645, 'preview': 0.4757, 'proxy_write': 0.020925, 'ui_sync': 0.001205}`
- Engine key metrics: `ratio=1.571, queue_drop=0.300, fresh_min=0.217, sla_p95_ms=7.869`
- Performance violations:
  - engine scenario 24: dsp ratio 1.112 > 1.100
  - engine scenario 24: fresh-result rate 0.217 < 0.900
  - engine scenario 36: dsp ratio 1.386 > 1.100
  - engine scenario 36: fresh-result rate 0.277 < 0.900
  - engine scenario 50: dsp ratio 1.571 > 1.100
  - engine scenario 50: queue-drop rate 0.300 > 0.030
  - engine scenario 50: fresh-result rate 0.369 < 0.900

## Failing Tests

- `test_audio_tasks_registry.TestAudioTasksRegistry.test_lufs_task_increases_with_signal_level` (failed)
  - AssertionError: 0.032544376207384026 not greater than 0.032544376207384026
- `test_engine_perf_gate.TestEnginePerfGate.test_engine_perf_strict_gate` (failed)
  - AssertionError: 1.1724931693001406 not less than or equal to 1.1 : Engine DSP strict gate failed sid=24: processes/threads dsp p95 ratio=1.172 > 1.100
- `setUpClass (test_multi_frontend_compile.TestMultiFrontendCompile)` (error)
  - AttributeError: 'types.SimpleNamespace' object has no attribute 'Collection'
- `test_parallel_dsp.TestParallelDSP.test_shared_memory_payload_resolution_reuses_worker_cache` (error)
  - ValueError: not enough values to unpack (expected 2, got 1)
- `test_parallel_dsp.TestParallelDSP.test_worker_supports_specs_splitter_stream_chain` (failed)
  - AssertionError: 0.0 not greater than 0.0
- `test_perf_gate.TestPerfGate.test_perf_gate_warning_mode_or_strict_threshold` (failed)
  - AssertionError: 0.36311500025476556 not less than or equal to 0.29931525030406186 : Perf regression sid=1 stage=dsp: p95=0.363ms baseline=0.222ms allowed=0.299ms
- `setUpClass (test_preferences_panel_ui.TestPreferencesPanelUI)` (error)
  - AttributeError: 'types.SimpleNamespace' object has no attribute 'PropertyGroup'
- `setUpClass (test_proxy_management.TestProxyManagement)` (error)
  - AttributeError: 'types.SimpleNamespace' object has no attribute 'Collection'

## Slowest Tests

- `test_perf_gate.TestPerfGate.test_perf_gate_warning_mode_or_strict_threshold` [performance | failed] 2.566s
- `test_perf_gate.TestPerfGate.test_perf_harness_generates_required_scenarios` [performance | passed] 2.492s
- `test_regression_findings.TestRegressionFindings.test_auto_attr_ids_reinitialize_on_reconnect` [integrity | passed] 0.002s
- `test_regression_findings.TestRegressionFindings.test_scene_frontend_sink_dispatch_contract_exists` [integrity | passed] 0.002s
- `test_regression_findings.TestRegressionFindings.test_handlers_are_persistent` [integrity | passed] 0.002s
- `test_regression_findings.TestRegressionFindings.test_file_source_description_uses_original_sample_rate_metadata` [integrity | passed] 0.002s
- `test_audio_tasks_registry.TestAudioTasksRegistry.test_lufs_task_increases_with_signal_level` [integrity | failed] 0.001s
- `test_audio_tasks_registry.TestAudioTasksRegistry.test_splitter_then_chromagram_preview_respects_parent_band_with_variable_chunk_size` [integrity | passed] 0.001s
- `test_regression_findings.TestRegressionFindings.test_n_panel_queues_non_blocking_recompiles_for_edit_operators` [integrity | passed] 0.001s
- `test_engine_perf_gate.TestEnginePerfGate.test_engine_perf_strict_gate` [performance | failed] 0.001s
