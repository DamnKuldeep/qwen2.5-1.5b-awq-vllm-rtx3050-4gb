# ui/

The chat page. One HTML file, no framework, no build step, served by the gateway at `/chat` so it is same-origin with the API and CORS never has to be loosened on an authenticated endpoint.

Renders markdown (headings, lists, bold, fenced code) with a small inline renderer, continues answers automatically when the model hits `max_tokens` mid-sentence, and surfaces 503s and context trimming instead of hiding them. Shows TTFT, tokens/s, context size and remaining budget live.

`node test_markdown.mjs` — 25 tests for the renderer, five of them XSS cases, extracted from the shipped page so they cannot drift from what actually runs.
