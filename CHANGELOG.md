# Changelog

All notable changes to this independent reproduction are recorded here. The
project follows Semantic Versioning. A tag identifies the exact source tree used
for the corresponding GitHub Release.

## [Unreleased]

## [0.1.3] - 2026-10-22

### Added

- A release consistency checker that requires the citation version, README
  version link, and newest dated changelog entry to agree.
- A public release checklist covering validation, scope changes, publishing,
  and post-release verification.
- Weekly Dependabot checks for the GitHub Actions used by CI.

### Changed

- CI and contributor instructions now include the release consistency check.
- Release-facing documentation now points to `0.1.3`.

## [0.1.2] - 2026-10-22

### Added

- A dependency-free Markdown link checker that fails on missing local targets,
  anchors attached to non-Markdown files, or paths escaping the repository.
- A dedicated link-validation step in CPU CI and the contributor checklist.

### Changed

- Release-facing documentation now points to `0.1.2`.

## [0.1.1] - 2026-10-22

### Added

- Form-ready Codex for Open Source application notes, with responses checked
  against the program's 500-character field limits.
- A release history and maintenance policy visible to downstream users.

### Changed

- Release-facing documentation now points to the current maintenance version.
- The application notes distinguish verified project evidence from public
  adoption metrics that must be checked at submission time.

### Fixed

- Replaced the stale internal roadmap reference with the reproducibility guide.

## [0.1.0] - 2026-10-22

### Added

- First public six-stage, camera-relative Ego2Robot reproduction.
- Pinned upstream sources, isolated runtime setup, data contracts, provenance,
  CPU checks, acceptance evidence, and representative result images.

[Unreleased]: https://github.com/aownown/ego2robot-repro/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/aownown/ego2robot-repro/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/aownown/ego2robot-repro/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/aownown/ego2robot-repro/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/aownown/ego2robot-repro/releases/tag/v0.1.0
