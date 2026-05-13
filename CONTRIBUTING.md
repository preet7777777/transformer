# Contributing

Thanks for helping improve Transformer From Scratch.

## How to contribute

1. Pick an issue or propose an improvement.
2. Read [implementation.md](implementation.md) before making changes.
3. Keep the change small and focused.
4. Add or update tests if the behavior changes.
5. Run the checks before opening a pull request.

## Good contribution ideas

- improve documentation
- add an evaluation metric or baseline
- add a small ablation
- improve generation or benchmarking tools
- add tests for existing features
- fix bugs or formatting issues

## Local checks

Run:

- PYTHONPATH=src pytest -q
- make checks

If your change affects experiments or published numbers, update the relevant results file.

## Pull request tips

- describe what changed and why
- mention any new commands or files
- include outputs or screenshots if useful
- keep the PR focused on one idea when possible

## Style

- preserve the existing code style
- prefer clear names and small functions
- avoid unrelated refactors
- do not overclaim results

## Need ideas?

If you want to help but do not know where to start, look for:

- small bugs
- doc improvements
- extra tests
- benchmark polish
- clearer examples in the README
