set dotenv-load

export EDITOR := 'nvim'

alias r := run

default:
  just --list

[group: 'format']
fmt:
  uv run ruff check --select I --fix && ruff format

run:
  uv run main.py
