# Releasing

Checklist for a Kyojin release (`X.Y.Z`).

1. Merge the release pull requests into `main`; bump `exllamav3/version.py`.
2. Build the release on the Strix Halo box and the RTX 3090 box and run the served gate: no prefill or decode cell more than 5 % below its reference.
3. Run the CPU test suite and the targeted GPU tests of the changed paths.
4. Tag `vX.Y.Z` on the merge commit and publish the GitHub release with the notes below.
5. Update the README speed table if a figure moved, and `doc/credits.md` for new contributors.

## Release notes template

```markdown
## Kyojin X.Y.Z

- **Headline change (#PR).** What changed, for whom, and the default.
- **Fixes (#PR, #PR).** One line each.
- **Tests and docs (#PR).**

Served gate (Strix Halo box for the GLM and Flash-Next cells, RTX 3090 box for the RTX cells):

| Cell | Prefill tok/s X.Y.Z | previous | Decode tok/s X.Y.Z | previous |
|---|---|---|---|---|

Thanks to @author1 (#PR), @author2 (#PR) for their pull requests in this release, and to @reporter for the report behind #PR.
```

The **Thanks** line is required: list every author of a merged pull request in the release (from `gh pr list --state merged --search "merged:>=<previous release date>"`), plus the people whose reports led to a fix. Skip it only when every change came from the maintainers.
