# CLI Reference

## faust chat

Start an interactive multi-turn session.

Examples:
- `faust chat`

In-session commands:
- `/exit` or `/quit` — end the session

## faust run

Run a single prompt and print the response. No session history is kept.

Example:
- `faust run "Explain recursion in one sentence"`

## faust config show

Display the current configuration file contents.

## faust config set

Override a top-level config key.

Examples:
- `faust config set model deepseek-r1:8b`
- `faust config set temperature 0.2`
