"""Keep archived experiment inputs outside live test and doctest discovery."""

# Results contain immutable scripts with historical paths and command-line inputs.
# Executable tests and package doctests remain collected from their original paths.
collect_ignore = ["results"]
