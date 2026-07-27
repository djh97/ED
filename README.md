# Agentic AI for Emergency Department Operations and Care Escalation

This workspace contains a research prototype of an agentic emergency department decision-support system.

## Architecture

- `FastAPI` backend
- `Input Understanding Agent` for structured JSON or free-text ED notes
- one LLM-based `Orchestration Agent` that follows a Level 4 loop: Goal -> Plan -> Execute -> Monitor outcomes -> Re-plan -> Continue until goal achieved
- `ED State Manager` that keeps active patients and unresolved follow-up tasks across sequential updates
- four enhanced decision tools:
  - TCN-style flow prediction adapter
  - XGBoost-style patient risk adapter with SHAP-style explanations
  - two-stage staffing availability adapter
  - MILP-style bed management adapter
- `Follow-Up Tracking Agent` for owner, due time, status, and escalation rule
- structured recommendation output
- short `action_brief` shown before detailed recommendations

The Input Understanding Agent and Orchestration Agent use LLM prompts. If no `OPENAI_API_KEY` is available, the agentic system raises a clear configuration error instead of falling back to deterministic planning. Deterministic logic is used only for comparison baselines and synthetic scenario generation.

Runtime flow:

1. The user sends only the ED input to the LLM Orchestration Agent.
2. The Orchestration Agent invokes the Input Understanding Agent to normalize messy text or validate structured JSON.
3. The ED State Manager merges new patients with the active ED state and keeps unresolved follow-up tasks.
4. The Orchestration Agent invokes the four tools for patient risk, flow/crowding, staffing, and bed/capacity.
5. The Orchestration Agent applies Goal -> Plan -> Execute -> Monitor outcomes -> Re-plan -> Continue until goal achieved to coordinate all agent/tool outputs.
6. The Orchestration Agent invokes the Follow-Up Tracking Agent and returns recommendations, explanation, escalation target, and follow-up tasks.

Stateful endpoints:

- `POST /state/evaluate`: evaluate and store a full ED snapshot.
- `POST /state/update`: add new patients, update operations, remove discharged patients, and re-plan over the full active ED state.
- `GET /state`: view current active ED state and pending follow-up tasks.
- `POST /state/reset`: clear the prototype in-memory state.

## Project layout

- `app/main.py` FastAPI application
- `app/models.py` request and response schemas
- `app/agents.py` input-understanding and follow-up tracking agents
- `app/agentic_system.py` agentic orchestration, baselines, and evaluation logic
- `app/prompts.py` LLM prompt templates for input normalization, orchestration, and summary generation
- `app/state_manager.py` in-memory active ED state manager
- `app/evaluation.py` synthetic benchmark output generation
- `app/mimic_iv_ed.py` optional MIMIC-IV-ED real-data evaluation
- `app/orchestrator.py` compatibility import path for the orchestration logic
- `app/tools/` task-specific tools
- `evaluation/` runnable evaluation scripts and evaluation notes
- `docs/llm_prompts.md` human-readable prompt documentation
- `demo_data/sample_case.json` example request payload
- `run_demo.py` local request script
- `tests/test_orchestrator.py` lightweight unit tests

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Open:

- `http://127.0.0.1:8000/docs`

Optional LLM configuration:

```bash
set OPENAI_API_KEY=your_api_key_here
set OPENAI_INPUT_MODEL=gpt-5.2
set OPENAI_ORCHESTRATION_MODEL=gpt-5.2
set OPENAI_SUMMARY_MODEL=gpt-5.2
```

Recommended model roles:

- `OPENAI_INPUT_MODEL`: fast structured extraction model for messy ED notes.
- `OPENAI_ORCHESTRATION_MODEL`: stronger reasoning model for comparing tool outputs and selecting actions.
- `OPENAI_SUMMARY_MODEL`: concise language model for the final narrative summary.

If these variables are not set, the app uses the default `OPENAI_MODEL` value from `app/config.py`.

## Demo request

Use the sample JSON in `demo_data/sample_case.json` or run:

```bash
python run_demo.py
```

## Evaluation

Run the controlled comparison:

```bash
python run_demo.py --evaluate
```

This command requires `OPENAI_API_KEY` because the agentic orchestration system is LLM-based. The rule-based, non-agentic, and individual-score baselines remain deterministic comparison systems.

Print compact summary metrics only:

```bash
python run_demo.py --evaluate --compact
```

Save paper-ready evaluation artifacts:

```bash
python run_demo.py --save-results
```

This writes:

- `evaluation_outputs/synthetic_comparison_full.json`
- `evaluation_outputs/synthetic_summary_metrics.csv`
- `evaluation_outputs/example_agentic_output.json`

Generated evaluation outputs are ignored by Git and are not included in the public repository.

Run MIMIC-IV-ED patient-risk evaluation after downloading the dataset:

```bash
python run_demo.py --mimic-data path/to/mimic-iv-ed/2.2/ed --mimic-label admitted
```

The comparison includes:

- `esi_triage_baseline`: ESI-like triage acuity rules mapped to patient escalation.
- `news2_qsofa_baseline`: NEWS2/qSOFA-style deterioration scores mapped to patient escalation.
- `nedocs_edwin_crowding_baseline`: NEDOCS/EDWIN-style crowding score mapped to flow, staffing, and bed alerts.
- `prediction_only_baseline`: tools generate scores, but no action planner exists.
- `rule_based_baseline`: fixed thresholds convert tool outputs into actions.
- `non_agentic_integrated_baseline`: each model output maps to one fixed dashboard action.
- `agentic_orchestration`: the orchestration agent compares tools, resolves context, and recommends actions.

The summary metrics include precision, recall, action quality, alert burden, simulated recommendation delay, explanation quality, response time, escalation recall, and escalation target accuracy.

The default agentic workflow evaluation uses 180 deterministic expert/silver-standard ED scenarios:

- 6 core hand-written scenarios covering sepsis congestion, stable operations, isolated high-risk patient, staffing crunch, bed block, and queue overload.
- 174 mixed scenarios combining flow pressure, patient risk, staffing pressure, and bed availability conditions.

The NHAMCS ED public-data evaluation uses real visit-level labels such as high-acuity triage, critical-vitals proxy, prolonged wait, very prolonged wait, and 72-hour revisit. NHAMCS does not directly label operational actions such as `staffing_alert` or `reassign_bed`, so those are evaluated with expert/silver-standard scenarios. MIMIC-IV-ED remains optional if credentialed access becomes available.

## Notes

- The agentic orchestration layer requires an LLM for coordination and reasoning.
- Deterministic methods are retained only as explicit baselines for evaluation.
- This is a research prototype, not a clinical production system.
