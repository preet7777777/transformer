# Open Source Setup Checklist

Use this checklist to keep the repository easy to contribute to and easy to trust.

## Repository settings

- Make the repository public.
- Protect the main branch.
- Require pull requests for changes.
- Require checks to pass before merging.
- Prefer squash merges for a clean history.
- Restrict direct pushes to maintainers only.
- Enable issues and discussions if useful.

## Collaboration flow

- Contributors open issues or fork the repo.
- Changes go through pull requests.
- CI runs on every pull request.
- A maintainer reviews and merges after checks pass.

## Files already in place

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [ROADMAP.md](ROADMAP.md)
- [implementation.md](implementation.md)
- issue and pull request templates under [.github](.github)

## Good open-source habits

- keep the README current
- keep tests green
- keep benchmark and results files updated
- respond to issues and PRs clearly
- label beginner-friendly tasks

## Suggested GitHub branch protection rules

- Require a pull request before merging
- Require status checks to pass
- Require at least one review if available
- Dismiss stale approvals when new commits are pushed
- Block force pushes
- Block branch deletion

## If you want contributions from others

The best path is:

1. make the repo public
2. protect main
3. keep CI green
4. make the contribution docs easy to find
5. label a few small starter issues
6. review PRs quickly
