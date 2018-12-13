
# Headline parser

Standardize news article headlines. As much as possible, the goal is to remove any element of content-independent "house style." Eg, if every AP headline looks like:

> AP News: Today A and B happened...

> AP News: Yesterday X and Y happened...

We don't care about things like the `AP News:` prefix, just the actual headline.

Steps:
1. Standardize quotes, hyphens; drop Twitter `@` and `#`.
1. Break on any kind of "separator" characters that don't appear in vanilla English sentences - `:`, `|`, `~`, etc. And the word `via`, which is used by some outlets to identify the source - `... via @dailycaller`.
1. Take the longest sequence of unbroken tokens.
1. Clean the tokens - downcase, strip punctuation. Keep `$`, but replace `[0-9]+` digits with a single `#` character, since different outlets have different conventions for how numbers are reported / formatted.

Generally, if we imagine a classifier that tries to predict `(headling -> outlet)`, the idea is to get rid of anything that would give the model an "unfair advantage," and force it to just consider the raw linguistic content of the headline.
