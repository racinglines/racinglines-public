# Contributing

Thanks for looking. How this repository works:

- **It holds the releases.** Each commit here is one PyPI version, exported from the development repository and pushed
  with its tag (`vX`). Development happens elsewhere, so there's no day-to-day history here.
- **Issues are the best way in.** Bug reports, questions and ideas are all welcome. Include the version
  (`racinglines --version`), the command you ran and what it printed.
- **Pull requests are welcome too, but they aren't merged here.** An accepted change is carried into the development
  repository by hand and comes back in the next release, credited in [CHANGELOG.md](CHANGELOG.md). The next export
  replaces this repository's tree, so a change merged here directly would be lost.
- **Keep a change small and focused**, one fix or feature per pull request, with a test where it fits
  (`pytest -m "not live"`).
- **License.** The code is MIT ([LICENSE](LICENSE)), and contributions are accepted under the same license. The
  racinglines name and branding are not covered by it.
