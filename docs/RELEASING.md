# Release checklist

Public releases use Semantic Versioning and an annotated GitHub tag. Every tag
must point to a tree whose repository, link, syntax, smoke, and release checks
have passed in GitHub Actions.

Before publishing a release:

1. Update `CITATION.cff`, the README version link, and `CHANGELOG.md` together.
2. Describe only completed work and name the boundary of the reproduced scope.
3. Run the local checks:

   ```bash
   python3 tools/repo_check.py
   python3 tools/check_links.py
   python3 tools/check_release.py
   python3 -m compileall -q workflow test_data/_test1019/scripts tools
   bash tests/smoke.sh
   git diff --check
   ```

4. Push the source commit and require the CPU checks workflow to pass.
5. Create the GitHub Release from the exact validated commit on `main`.
6. Re-open the release and confirm the tag, commit, notes, and source archives.
7. Confirm the workflow triggered by the tag also passes.

The release notes must state changed behavior, validation evidence, and any
remaining quality boundaries. Do not backdate releases, invent adoption metrics,
or label numerical completion as scientific validation. Model weights, source
videos, generated frames, environments, and third-party checkouts remain outside
Git history.

Dependabot checks GitHub Actions weekly. Dependency updates run through the same
CPU checks and receive maintainer review before merge.
