set dotenv-load

export EDITOR := 'nvim'

alias r := run

default:
  just --list

ci: check fmt-check

[group: 'check']
check:
  uv run ruff check

[group: 'format']
fmt:
  uv run ruff check --select I --fix && ruff format

[group: 'check']
fmt-check:
  uv run ruff format --check .

run:
  uv run oterl
