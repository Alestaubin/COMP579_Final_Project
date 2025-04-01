## COMP579 Final Project

To get started, follow the steps to get [mise](https://mise.jdx.dev/getting-started.html) 
(optional) and [uv](https://docs.astral.sh/uv/) installed on your system. 

For **macOS** users, the easiest way is with [homebrew](https://brew.sh/):

```bash
brew install mise uv
```

`mise` makes it easy to manage multiple Python versions on your system. For this
project we pin the Python version to `3.9`. If you have it installed, `mise` will
switch your system Python to the pinned version upon entering the project
directory.

We use `uv` to manage dependencies.

Once these are setup, you can run the entry-point with:

```bash
uv run main.py
```
