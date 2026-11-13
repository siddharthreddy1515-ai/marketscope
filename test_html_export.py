
"""Regression tests for browser-side HTML report export."""

from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess

import pytest


INDEX = Path(__file__).with_name("index.html")


class ExportButtonParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.buttons = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)

        if (
            tag == "button"
            and attributes.get("id") == "downloadReportButton"
        ):
            self.buttons.append(attributes)


def test_export_button_is_unique_and_disabled_initially():
    parser = ExportButtonParser()

    parser.feed(
        INDEX.read_text(encoding="utf-8")
    )

    assert len(parser.buttons) == 1
    assert "disabled" in parser.buttons[0]
    assert parser.buttons[0].get("type") == "button"


# Run the actual dashboard JavaScript using Node.js
# and a small fake DOM. No API or npm packages needed.

NODE_CHECK = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const page = fs.readFileSync(process.argv[1], 'utf8');

const script = page.match(
  /<script>([\s\S]*?)<\/script>/i
);

assert.ok(script, 'Expected an inline dashboard script');

const nodes = {};
const downloads = [];
const blobs = [];
let fetchCalls = 0;

class FakeNode {
  constructor(tag) {
    this.tag = tag;
    this.style = {};
    this.children = [];
    this.handlers = {};
    this.value = '';
    this.disabled = false;
    this.textContent = '';
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  replaceChildren(...children) {
    this.children = children;
  }

  setAttribute(name, value) {
    this[name] = value;
  }

  addEventListener(name, fn) {
    this.handlers[name] = fn;
  }

  click() {
    if (this.tag === 'a') {
      downloads.push(this.download);
    }
  }

  remove() {}
}

for (const id of [
  'searchForm',
  'companyInput',
  'websiteInput',
  'analyzeBtn',
  'loader',
  'output',
  'demoLabel',
  'downloadReportButton'
]) {
  nodes[id] = new FakeNode(id);
}

nodes.downloadReportButton.disabled = true;

const document = {
  getElementById: id => nodes[id],

  createElement: tag => new FakeNode(tag),

  createTextNode: text => ({
    textContent: text
  }),

  body: new FakeNode('body')
};

URL.createObjectURL = blob => {
  blobs.push(blob);
  return 'blob:test';
};

URL.revokeObjectURL = () => {};

const context = vm.createContext({
  document,
  URL,
  URLSearchParams,
  Blob,
  location: { search: '' },
  setTimeout: () => {},

  fetch: async () => {
    fetchCalls += 1;

    return {
      ok: false,
      json: async () => ({
        detail: 'Simulated error'
      })
    };
  }
});

vm.runInContext(script[1], context);

// Button starts disabled.
assert.equal(
  nodes.downloadReportButton.disabled,
  true
);

assert.equal(
  typeof nodes.downloadReportButton.handlers.click,
  'function'
);

// Fictional report containing both safe and unsafe text.
context.sample = {
  company_name: 'ExampleCo',
  status: 'completed',

  report: '<script>alert("bad")</script>',

  executive_summary: {
    text: 'Verified <strong>claim</strong>',
    sources: [{
      title: 'Official source',
      url: 'https://example.com/press'
    }]
  },

  insight_items: [{
    category: 'Other',
    text: '<img src=x onerror=alert(1)>',

    sources: [{
      title: 'Unsafe source',
      url: 'javascript:alert(1)'
    }]
  }],

  competitors: [{
    name: 'Competitor',
    overlap: 'Similar offering',

    sources: [{
      title: 'Comparison',
      url: 'https://example.com/compare'
    }]
  }],

  source_sentiments: [],
  sentiment_distribution: null,

  source_stats: {
    returned: 1,
    accepted: 1,
    discarded: 0
  },

  research_warnings: ['Partial coverage'],
  research_attempts: 1,

  sources: [{
    title: 'Official source',
    url: 'https://example.com/press',
    published_at: '2026-11-01'
  }]
};

// A completed preview should enable export.
vm.runInContext('render(sample, true)', context);

assert.equal(
  nodes.downloadReportButton.disabled,
  false
);

// Generate the actual exported HTML.
const html = vm.runInContext(
  'buildReportHtml(sample, true)',
  context
);

// Important report sections must exist.
for (const heading of [
  'Executive Summary',
  'Strategic Insights',
  'Competitor Analysis',
  'Research Coverage',
  'Evidence Sources',
  'Research Warnings'
]) {
  assert.ok(
    html.includes(heading),
    `Missing ${heading}`
  );
}

// Preview must be clearly identified as fictional.
assert.ok(
  html.includes('fictional ExampleCo data')
);

// Safe source links must be preserved.
assert.ok(
  html.includes('href="https://example.com/press"')
);

// Untrusted HTML must be escaped.
assert.ok(
  html.includes('&lt;script&gt;alert')
);

assert.ok(
  html.includes('&lt;img src=x onerror=alert(1)&gt;')
);

assert.ok(!html.includes('<script>'));

// Dangerous citation URLs must not become links.
assert.ok(
  !html.includes('href="javascript:')
);

// Simulate clicking Download.
nodes.downloadReportButton.handlers.click();

assert.equal(
  downloads[0],
  'marketscope-ExampleCo-preview.html'
);

assert.equal(blobs.length, 1);

// Export must not trigger another research request.
assert.equal(
  fetchCalls,
  0,
  'Export must not call research API'
);

// After a failed new analysis, an old report
// must not remain downloadable.
nodes.companyInput.value = 'Tesla';

(async () => {
  await vm.runInContext('analyze()', context);

  assert.equal(fetchCalls, 1);

  assert.equal(
    nodes.downloadReportButton.disabled,
    true
  );

  assert.equal(
    vm.runInContext('latestReportData', context),
    null
  );

  process.stdout.write(
    'Export runtime checks passed\n'
  );

})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
"""


def test_export_runtime_and_stale_report_guard():
    node = shutil.which("node")

    if node is None:
        pytest.skip(
            "Node.js not installed; runtime export test unavailable"
        )

    result = subprocess.run(
        [node, "-e", NODE_CHECK, str(INDEX)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, (
        result.stderr or result.stdout
    )
