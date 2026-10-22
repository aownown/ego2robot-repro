# Codex for Open Source application notes

This document contains a factual project summary and ready-to-adapt application
text. It is not part of the software contract. Replace the placeholders for the
applicant's name, ChatGPT email, and current usage evidence before submitting.

## Facts to report

- GitHub account: `aownown`
- Repository: `https://github.com/aownown/ego2robot-repro`
- Role: creator and maintainer
- License: Apache-2.0 for original code; upstream components remain separately licensed
- First release: `0.1.0`, 2026-10-22
- Tested scope: one 450-frame, 30 fps, 15-second clip through six stages
- External use, stars, downloads, and contributors must be reported from the
  GitHub/Analytics pages at submission time. Do not estimate them.

## Short description

An open, inspectable reproduction of the six-stage Ego2Robot data pipeline. It
turns a first-person human manipulation clip into camera-relative gripper
trajectories, inpainted scenes, and synchronized robot RGB, Z depth, and masks.
The repository focuses on reproducibility: pinned upstream revisions, isolated
environments, explicit data contracts, resumable stages, independent checks,
and honest quality flags.

## Why it matters

Large-scale ego-to-robot data synthesis is difficult to evaluate from a paper
alone. This project makes a concrete slice executable and debuggable: researchers
can inspect each model boundary, replace a component, compare robot morphologies,
and discover where quality is lost. It also exposes practical failure modes such
as duplicate hand detections, temporal gaps, inpainting ghosting, gripper-width
clipping, memory limits, sensor-depth mismatch, and the difference between
camera-relative and world-space trajectories.

## Maintainer responsibilities

I designed the six-stage orchestration, adapted WiLoR outputs to a documented
camera-relative Dyn-HaMR path, implemented hand-track association and gating,
built the hand-to-gripper conversion, constrained dual-arm IK and rendering
pipeline, added depth-aware compositing, and created the acceptance and
provenance checks. I maintain the setup scripts, pinned dependencies, issue
triage, documentation, and reproducible validation workflow.

## Honest limitations

This is an independent engineering reproduction. It does not claim the paper's
dataset scale, VLA benchmark results, or real-robot experiments. The published
acceptance clip completed numerically, but visual quality, physical scale, and
world trajectory recovery remain explicitly unaccepted. The next work is
aligned sensor depth, SLAM/world recovery, better human-shadow removal, a second
robot morphology, and additional external clips.

## How Codex supports the project

Codex accelerates repetitive engineering work: tracing failures across seven
Python environments, adapting configuration and path handling, writing
validation tools, reviewing provenance, triaging compatibility issues, and
maintaining documentation. The maintainer reviews and tests the changes before
release. API quota would support open-source maintenance passes, test triage,
documentation updates, and adapter work needed as contributors add hardware or
data variants.

## Proposed API quota use

1. Add automated compatibility checks for Ubuntu and GPU-driver variants.
2. Review and test community adapters for additional robot morphology.
3. Improve reproducible packaging and error diagnosis across seven runtimes.
4. Maintain documentation and triage public bug reports.
5. Run review and validation automations on pre-release changes.

## Official form draft

All response fields below fit the 500-character limit enforced by the official
form. Keep the name, ChatGPT email, and OpenAI Organization ID in the form
itself; they are intentionally not committed to a public repository.

| Field | Character count | Recommended English response |
|---|---:|---|
| Role | — | Primary maintainer (radio selection) |
| GitHub username | — | `aownown` |
| Repository URL | — | `https://github.com/aownown/ego2robot-repro` |
| Interest | — | Select `Codex Security` and `Project API credits` |
| Why this repository qualifies | 484 | This Apache-2.0 project makes ego-to-robot synthesis debuggable: six stages convert first-person RGB into camera-relative gripper targets and synchronized robot RGB, depth, and masks. v0.1.0, passing CI, contracts, provenance, and a 450-frame acceptance run expose integration failures instead of hiding them behind an exit code. It is newly public: 0 stars and 0 forks. Its value is a reproducible baseline and integration target for egocentric-vision and robot-learning researchers. |
| How API credits would be used | 455 | API credits would automate maintenance: triage failures across seven runtimes; test path, NumPy/SciPy, timestamp, mask, and MANO compatibility; audit dependency and model-source changes; draft issue reproducers; maintain installation and data contracts; run pre-release security, link, and packaging checks; and test new robot or input adapters. The maintainer reviews and tests releases. Credits support software review and integration, not GPU training. |
| Additional information | 438 | v0.1.0 was released publicly on 2026-10-22 and is maintained by its creator, aownown. It is independent from the paper authors. It covers one 450-frame camera-relative clip, with no claim to paper-scale data, VLA training, benchmarks, sensor-depth calibration, world recovery, or real-robot validation. Quality, scale, and world-recovery flags remain false. Data and weights are excluded; setup uses official or access-controlled sources. |

The official form asks for Codex Security rationale indirectly through the
interest selection and the additional-information field. The companion 497
character rationale below can replace the additional-information response when
security is the priority; it repeats provenance context to make the security
claim self-contained.

> The project clones third-party research code, builds seven Conda environments,
> downloads checkpoints, runs subprocesses, and processes external videos and
> annotations. Contributions could introduce command injection, unsafe URLs or
> deserialization, credential leakage, or dependency drift. We pin revisions,
> track licenses, scan secrets and paths, run CPU CI, and use read-only Actions
> permissions. Codex Security would review PRs and setup changes for these
> supply-chain and code-execution risks.

Before submitting, check the live repository indicators. They were 0 stars and
0 forks on 2026-10-22. If those numbers change, update the first response rather
than retaining stale figures. The project has no package download metric and no
third-party user survey, so the application must not invent one.

Official links:

- Form: https://openai.com/form/codex-for-oss/
- Program guidance: https://developers.openai.com/codex/codex-for-oss
- Program terms: https://developers.openai.com/codex/codex-for-oss-terms

## GitHub repository metadata

Description:

> Open six-stage Ego2Robot reproduction: ego RGB to robot trajectories, RGB-D,
> masks, and depth-aware composites.

Suggested topics:

`robotics`, `egocentric-vision`, `imitation-learning`, `robot-data`,
`wilor`, `dynhamr`, `sam3`, `propainter`, `mujoco`, `inverse-kinematics`,
`depth-estimation`, `reproducible-research`
