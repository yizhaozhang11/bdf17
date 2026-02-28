# Agent Instructions

## Project Context

- Read `README.md` first for academic and technical details about this project.

## Conda Setup

- In non-interactive shell sessions, initialize conda first with `source "$HOME/miniconda3/etc/profile.d/conda.sh"`.
- Use `conda run -n <env> <command>` for environment-specific commands.
- Preferred pattern: `source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda run -n <env> <command>`.

## SageMath Scripts

- Use the conda environment `sage` for all `*.sage` files.
- Run Sage scripts with `conda run -n sage sage <sagefile>.sage`.
- Preferred one-liner: `source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda run -n sage sage <sagefile>.sage`.
