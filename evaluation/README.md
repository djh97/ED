# Evaluation Plan

This project uses two complementary evaluation tracks.

## Track 1: Synthetic / Expert ED Scenarios

Purpose: evaluate the full agentic decision-support behavior.

Default size: 180 scenarios.

- 6 core hand-written scenarios cover common ED operations and escalation cases.
- 174 mixed scenarios combine patient risk, crowding/flow pressure, staffing pressure, and bed/capacity pressure.
- Expected actions are generated from an expert/silver-standard rubric based on triage acuity, unstable vitals, wait time, queue pressure, staffing capacity, occupancy, boarding, and high-acuity bed availability.

Run:

```bash
python run_demo.py --save-results
```

Generated outputs are written to `evaluation_outputs/`, which is ignored by Git and not included in the public repository.

Main comparisons:

- prediction-only baseline
- rule-based baseline
- non-agentic integrated baseline
- ESI-like triage baseline
- NEWS2/qSOFA-like deterioration baseline
- NEDOCS/EDWIN-like crowding baseline
- agentic orchestration system

Metrics:

- precision
- recall
- average action quality
- alert burden
- simulated recommendation delay
- explanation quality
- response time
- escalation recall
- escalation target accuracy

## Additional Agentic Evaluations

Purpose: evaluate properties of the proposed system that are not captured by simple action-label matching.

Run:

```bash
py -3.12 evaluation/run_additional_evaluations.py --mode all
```

Additional evaluations:

- `ablation`: compares prediction-only, non-agentic fixed mapping, and rule-based no-LLM variants to show what is lost when action planning, reasoning, and orchestration are removed.
- `component`: runs the same 180-scenario benchmark while disabling safety validation, state management, and follow-up tracking one at a time.
- `safety_validation`: checks that the safety validator detects missing high/critical patient escalations and accepts plans that explicitly escalate all high/critical patients.
- `stateful_replanning`: applies three sequential ED updates and checks whether active patients are retained, new critical patients are incorporated, and recommendations change as conditions worsen.

Run the component contribution analysis:

```bash
py -3.12 evaluation/run_component_contribution.py
```

To run the stateful re-planning evaluation with the real LLM Orchestration Agent:

```bash
set OPENAI_API_KEY=your_key_here
py -3.12 evaluation/run_additional_evaluations.py --mode stateful --real-agentic
```

## Track 2: NHAMCS ED Public Dataset

Purpose: evaluate real public ED visit labels without credentialed MIMIC access.

Supported input:

- Kaggle NHAMCS 2018-2022 ZIP, for example `archive (5).zip`
- extracted `nhamcs_data_2018_22.csv`

Run a dataset summary:

```bash
python run_demo.py --nhamcs-data "C:\Users\ku500822\Downloads\archive (5).zip" --nhamcs-summary
```

Run a real-data prediction evaluation:

```bash
python run_demo.py --nhamcs-data "C:\Users\ku500822\Downloads\archive (5).zip" --nhamcs-task high_acuity
```

Supported NHAMCS tasks:

- `high_acuity`
- `critical_vitals`
- `prolonged_wait`
- `very_prolonged_wait`
- `revisit_72h`

Important limitation: this Kaggle NHAMCS package does not include admission/disposition fields, so it should not be used for admission prediction unless a version with disposition is added.

## Optional Track 3: MIMIC-IV-ED Real Dataset

Purpose: evaluate real ground-truth prediction labels.

Required local files from PhysioNet MIMIC-IV-ED:

- `edstays.csv.gz`
- `triage.csv.gz`

Run:

```bash
python run_demo.py --mimic-data path/to/mimic-iv-ed/2.2/ed --mimic-label admitted
```

Supported labels:

- `admitted`
- `high_acuity`
- `prolonged_los`
- `critical_proxy`

Important limitation: MIMIC-IV-ED has real patient-level labels, but it requires credentialed PhysioNet access and does not directly label agentic workflow actions such as `staffing_alert` or `reassign_bed`. Those actions are evaluated using synthetic/expert scenarios.
