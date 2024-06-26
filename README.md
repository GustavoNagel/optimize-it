# optimize-it

<div align="center">

[![Build status](https://github.com/GustavoNagel/optimize-it/actions/workflows/build.yml/badge.svg?branch=master&event=push)](https://github.com/GustavoNagel/optimize-it/actions/workflows/build.yml?branch=master&event=push)
[![Python Version](https://img.shields.io/pypi/pyversions/optimize-it.svg)](https://pypi.org/project/optimize-it/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/GustavoNagel/optimize-it/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/GustavoNagel/optimize-it/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/GustavoNagel/optimize-it/releases)
[![License](https://img.shields.io/github/license/GustavoNagel/optimize-it)](https://github.com/GustavoNagel/optimize-it/blob/master/LICENSE)
![Coverage Report](assets/images/coverage.svg)

Meta heuristic algorithms used for solving optimization problems.

</div>

## Basic usage

Given one function to be minimized and the lower and upper bounds defining 
search space to the algorithm, BSA algorithm can be called and minimum solution should be available.

```python
    from scipy.optimize import Bounds, OptimizeResult

    func = lambda x: x ** 4 + 4 * x ** 3 - 13 * x ** 2 - 14 * x + 24
    bounds = Bounds(lb=np.array([-20]), ub=np.array([20]))
    bsa = BSA(generations=500)
    result = bsa.run(func, bounds)

    assert isinstance(result, OptimizeResult)
    print(result.x)  # Global minimum solution for function
```

Firefly algorithm is also available.

```python
    firefly = Firefly(generations=500)
    result = firefly.run(func, bounds)
```

## Installation

```bash
pip install -U optimize-it
```

or install with `Poetry`

```bash
poetry add optimize-it
```

Then you can run

```bash
optimize-it --help
```

or with `Poetry`:

```bash
poetry run optimize-it --help
```

## 🚀 Features

### Development features

- Supports for `Python 3.10` and higher.
- [`Poetry`](https://python-poetry.org/) as the dependencies manager. See configuration in [`pyproject.toml`](https://github.com/optimize_it/optimize-it/blob/master/pyproject.toml) and [`setup.cfg`](https://github.com/optimize_it/optimize-it/blob/master/setup.cfg).
- Automatic codestyle with [`black`](https://github.com/psf/black), [`isort`](https://github.com/timothycrosley/isort) and [`pyupgrade`](https://github.com/asottile/pyupgrade).
- Ready-to-use [`pre-commit`](https://pre-commit.com/) hooks with code-formatting.
- Type checks with [`mypy`](https://mypy.readthedocs.io); docstring checks with [`darglint`](https://github.com/terrencepreilly/darglint);
- Testing with [`pytest`](https://docs.pytest.org/en/latest/).
- Ready-to-use [`.editorconfig`](https://github.com/optimize_it/optimize-it/blob/master/.editorconfig), [`.dockerignore`](https://github.com/optimize_it/optimize-it/blob/master/.dockerignore), and [`.gitignore`](https://github.com/optimize_it/optimize-it/blob/master/.gitignore). You don't have to worry about those things.

### Deployment features

- `GitHub` integration: issue and pr templates.
- `Github Actions` with predefined [build workflow](https://github.com/optimize_it/optimize-it/blob/master/.github/workflows/build.yml) as the default CI/CD.
- Everything is already set up for security checks, codestyle checks, code formatting, testing, linting, docker builds, etc with [`Makefile`](https://github.com/optimize_it/optimize-it/blob/master/Makefile#L89). More details in [makefile-usage](#makefile-usage).
- [Dockerfile](https://github.com/optimize_it/optimize-it/blob/master/docker/Dockerfile) for your package.
- Always up-to-date dependencies with [`@dependabot`](https://dependabot.com/). You will only [enable it](https://docs.github.com/en/github/administering-a-repository/enabling-and-disabling-version-updates#enabling-github-dependabot-version-updates).
- Automatic drafts of new releases with [`Release Drafter`](https://github.com/marketplace/actions/release-drafter). You may see the list of labels in [`release-drafter.yml`](https://github.com/optimize_it/optimize-it/blob/master/.github/release-drafter.yml). Works perfectly with [Semantic Versions](https://semver.org/) specification.

### Open source community features

- Ready-to-use [Pull Requests templates](https://github.com/optimize_it/optimize-it/blob/master/.github/PULL_REQUEST_TEMPLATE.md) and several [Issue templates](https://github.com/optimize_it/optimize-it/tree/master/.github/ISSUE_TEMPLATE).
- Files such as: `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md` are generated automatically.
- [Semantic Versions](https://semver.org/) specification with [`Release Drafter`](https://github.com/marketplace/actions/release-drafter).

### Makefile usage

[`Makefile`](https://github.com/optimize_it/optimize-it/blob/master/Makefile) contains a lot of functions for faster development.

<details>
<summary>1. Download and remove Poetry</summary>
<p>

To download and install Poetry run:

```bash
make poetry-download
```

To uninstall

```bash
make poetry-remove
```

</p>
</details>

<details>
<summary>2. Install all dependencies and pre-commit hooks</summary>
<p>

Install requirements:

```bash
make install
```

Pre-commit hooks coulb be installed after `git init` via

```bash
make pre-commit-install
```

</p>
</details>

<details>
<summary>3. Codestyle</summary>
<p>

Automatic formatting uses `pyupgrade`, `isort` and `black`.

```bash
make codestyle

# or use synonym
make formatting
```

Codestyle checks only, without rewriting files:

```bash
make check-codestyle
```

> Note: `check-codestyle` uses `isort`, `black` and `darglint` library

Update all dev libraries to the latest version using one comand

```bash
make update-dev-deps
```

<details>
<summary>4. Code security</summary>
<p>

```bash
make check-safety
```

This command launches `Poetry` integrity checks as well as identifies security issues with `Safety` and `Bandit`.

```bash
make check-safety
```

</p>
</details>

</p>
</details>

<details>
<summary>5. Type checks</summary>
<p>

Run `mypy` static type checker

```bash
make mypy
```

</p>
</details>

<details>
<summary>6. Tests with coverage badges</summary>
<p>

Run `pytest`

```bash
make test
```

</p>
</details>

<details>
<summary>7. All linters</summary>
<p>

Of course there is a command to ~~rule~~ run all linters in one:

```bash
make lint
```

the same as:

```bash
make test && make check-codestyle && make mypy
```

</p>
</details>

<details>
<summary>8. Docker</summary>
<p>

```bash
make docker-build
```

which is equivalent to:

```bash
make docker-build VERSION=latest
```

Remove docker image with

```bash
make docker-remove
```

More information [about docker](https://github.com/optimize_it/optimize-it/tree/master/docker).

</p>
</details>

<details>
<summary>9. Cleanup</summary>
<p>
Delete pycache files

```bash
make pycache-remove
```

Remove package build

```bash
make build-remove
```

Delete .DS_STORE files

```bash
make dsstore-remove
```

Remove .mypycache

```bash
make mypycache-remove
```

Or to remove all above run:

```bash
make cleanup
```

</p>
</details>

## 📈 Releases

You can see the list of available releases on the [GitHub Releases](https://github.com/optimize_it/optimize-it/releases) page.

We follow [Semantic Versions](https://semver.org/) specification.

We use [`Release Drafter`](https://github.com/marketplace/actions/release-drafter). As pull requests are merged, a draft release is kept up-to-date listing the changes, ready to publish when you’re ready. With the categories option, you can categorize pull requests in release notes using labels.
