/**
 * Tests for the chat UI's markdown renderer.
 *
 * Run:  node ui/test_markdown.mjs
 *
 * WHY THE FUNCTIONS ARE EXTRACTED FROM index.html RATHER THAN IMPORTED
 * --------------------------------------------------------------------
 * The renderer lives inline in the page, because the page is deliberately a
 * single dependency-free file served same-origin by the gateway. Copying the
 * functions into this test would create two versions that drift apart, and the
 * copy would keep passing after the real one broke - which is the failure mode
 * Stage 5 already hit once with a stub that reproduced the wrong delivery mode.
 * So this reads the shipped file and evaluates the real source.
 *
 * The security cases are not decoration. Model output is untrusted input: it is
 * text generated from a user's prompt and inserted into the DOM as HTML. If a
 * user can get the model to echo a <script> tag or a javascript: URL and that
 * reaches innerHTML, the page has an XSS hole driven by prompt injection.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, "index.html"), "utf8");

// Pull the four renderer functions out of the page and evaluate them here.
const names = ["escapeHtml", "safeHref", "renderInline", "renderMarkdown"];
const sources = names.map(name => {
  const start = html.indexOf(`function ${name}(`);
  if (start === -1) throw new Error(`function ${name} not found in index.html`);
  // Walk braces to find the end of the function body.
  let depth = 0, i = html.indexOf("{", start);
  const open = i;
  for (; i < html.length; i++) {
    if (html[i] === "{") depth++;
    else if (html[i] === "}") { depth--; if (depth === 0) break; }
  }
  return html.slice(start, i + 1);
});

const { renderMarkdown, renderInline, escapeHtml } =
  new Function(sources.join("\n") + "\nreturn { renderMarkdown, renderInline, escapeHtml };")();

let passed = 0, failed = 0;
function check(name, condition, detail = "") {
  if (condition) { passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? "  -> " + detail : ""}`); }
}

console.log("\nmarkdown rendering");

const heading = renderMarkdown("# Title\n\nSome **bold** and *italic* text.");
check("h1 heading", heading.includes("<h1>Title</h1>"), heading);
check("bold", heading.includes("<strong>bold</strong>"));
check("italic", heading.includes("<em>italic</em>"));
check("no raw asterisks survive", !heading.includes("**"));

const lists = renderMarkdown("- one\n- two\n\n1. first\n2. second");
check("unordered list", lists.includes("<ul>") && (lists.match(/<li>/g) || []).length === 4);
check("ordered list", lists.includes("<ol>"));

const fenced = renderMarkdown("Here:\n\n```python\ndef f(x):\n    return x * 2\n```\n\nDone.");
check("fenced code block", fenced.includes("<pre") && fenced.includes("<code>def f(x):"));
check("code language recorded", fenced.includes('data-lang="python"'));
check("prose around code survives", fenced.includes("Done."));

// While streaming, a code fence is open for many seconds before it closes.
// If that rendered as literal backticks the answer would visibly "flicker"
// from broken to correct at the end of every code block.
const partial = renderMarkdown("Start\n\n```python\ndef f(x):\n    return x");
check("unterminated fence still renders as code", partial.includes("<pre") && !partial.includes("```"));

const inlineCode = renderMarkdown("Call `foo(**kwargs)` now.");
check("inline code", inlineCode.includes("<code>foo(**kwargs)</code>"), inlineCode);
check("no emphasis inside inline code", !inlineCode.includes("<strong>"), inlineCode);

const misc = renderMarkdown("> quoted\n\n---\n\n[link](https://example.com)");
check("blockquote", misc.includes("<blockquote>quoted</blockquote>"));
check("horizontal rule", misc.includes("<hr>"));
check("link", misc.includes('href="https://example.com"'));

console.log("\nsecurity (model output is untrusted)");

const xss = renderMarkdown('<script>alert(1)</script>\n\n<img src=x onerror="alert(2)">');
check("script tag escaped", !xss.includes("<script>") && xss.includes("&lt;script&gt;"), xss);
check("img/onerror escaped", !xss.includes("<img"), xss);

const badLink = renderMarkdown("[click](javascript:alert(1))");
check("javascript: href neutralised", !badLink.includes("javascript:"), badLink);

const dataLink = renderMarkdown("[x](data:text/html;base64,PHNjcmlwdD4=)");
check("data: href neutralised", !dataLink.includes("data:text/html"), dataLink);

const quoteBreak = renderMarkdown('He said "hi" & left <here>');
check("quotes and ampersands escaped", quoteBreak.includes("&amp;") && quoteBreak.includes("&lt;here&gt;"));

console.log("\nregression: the exact shape this model emits");

// Verbatim structure from Qwen2.5-1.5B-Instruct-AWQ answering the Stage 6
// control question - heading, sub-headings, numbered list with bold labels,
// bulleted list, and a python fence.
const real = renderMarkdown(`# KV Cache in LLM Inference

## How it Works:

1. **Data Storage**: Data is stored as key-value pairs.
2. **Fast Access**: Keys can be quickly looked up.

## Key Terms:
- **Key**: A unique identifier.
- **Value**: The actual data item.

\`\`\`python
class KVCache:
    def __init__(self):
        self.cache = {}
\`\`\`
`);
check("real answer: headings", real.includes("<h1>") && real.includes("<h2>"));
check("real answer: numbered list with bold", real.includes("<ol>") && real.includes("<strong>Data Storage</strong>"));
check("real answer: bulleted list", real.includes("<ul>"));
check("real answer: code block", real.includes("<pre") && real.includes("class KVCache:"));
check("real answer: zero raw markdown left", !/\*\*|^#|```/m.test(real.replace(/<[^>]+>/g, "")), real.replace(/<[^>]+>/g, ""));

console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed ? 1 : 0);
