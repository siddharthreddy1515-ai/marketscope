
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
const malformedAnalysis = process.argv[3] === 'malformed';
const errorMessage = malformedAnalysis
  ? 'The analysis response could not be read. Please try again.'
  : 'Simulated error';

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
  'downloadReportButton',
  'downloadEvidenceButton'
]) {
  nodes[id] = new FakeNode(id);
}

nodes.downloadReportButton.disabled = true;
nodes.downloadEvidenceButton.disabled = true;

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
      status: malformedAnalysis ? 502 : 500,
      json: async () => ({
        detail: errorMessage
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
  'Findings',
  'Competitor Analysis',
  'Limited research coverage',
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

const descendants = node => [node, ...(node.children || []).flatMap(descendants)];
const dashboardNodes = () => descendants(nodes.output);
const coverageHeading = 'Limited research coverage';
const sourceLink = node => node.tag === 'a' &&
  node.href === 'https://example.com/press' && node.textContent === 'Official source';
assert.ok(dashboardNodes().some(sourceLink), 'Dashboard citations show source names');
assert.ok(dashboardNodes().some(node => node.textContent === 'Partial coverage'));
assert.ok(!dashboardNodes().some(node => ['Market Position', 'Threat Level'].includes(node.textContent)));

const originalSample = context.sample;
context.sample = {
  ...originalSample,
  analysis_status: 'complete',
  research_attempts: 2,
  source_stats: {returned: 2, accepted: 1, discarded: 1},
  research_warnings: ['One duplicate URL removed.', 'Only one usable company source.'],
  sources: [{...originalSample.sources[0], published_at: null}],
  sentiment_distribution: {positive: 0, neutral: 1, negative: 0, assessed: 1, unassessed: 0, total_sources: 1},
  insight_items: [{category: 'Funding & Investment', text: 'Saved profile finding.', sources: originalSample.sources}]
};
vm.runInContext('render(sample)', context);
const limitedHtml = vm.runInContext('buildReportHtml(sample)', context);
const visibleText = dashboardNodes().map(node => node.textContent);
assert.ok(visibleText.indexOf(coverageHeading) >= 0);
assert.ok(visibleText.indexOf(coverageHeading) < visibleText.indexOf('Saved profile finding.'));
assert.ok(limitedHtml.indexOf(coverageHeading) >= 0);
assert.ok(limitedHtml.indexOf(coverageHeading) < limitedHtml.indexOf('Executive Summary'));
for (const text of ['One duplicate URL removed.', 'Only one usable company source.', '1 accepted company source']) {
  assert.ok(visibleText.some(line => line.includes(text)), text);
  assert.ok(limitedHtml.includes(text), text);
}
assert.ok(visibleText.some(line => line.includes('Publication date unavailable')));
assert.ok(!visibleText.some(line => /100%|STRATEGIC FINDING/.test(line)));
assert.ok(visibleText.includes('1 company source assessed as neutral.'));
assert.ok(visibleText.includes('1 finding in Funding & Investment.'));
assert.ok(dashboardNodes().some(sourceLink));

// A single dated announcement can still support a substantive finding.
context.sample = {
  ...originalSample,
  analysis_status: 'complete',
  insight_items: [{category: 'Product & Innovation', text: 'ExampleCo launched a new product.', sources: originalSample.sources}]
};
vm.runInContext('render(sample)', context);
assert.ok(dashboardNodes().some(node => node.textContent === 'ExampleCo launched a new product.'));
assert.ok(dashboardNodes().some(sourceLink));
assert.ok(!dashboardNodes().some(node => /background.only/i.test(node.textContent)));

context.sample = {
  ...originalSample,
  research_coverage: {
    status: 'background_only', background_sources: 1, development_sources: 0, uncertain_sources: 0,
    limitations: ['The collected sources provide background information. Important events may be missing.']
  },
  competitor_research_warnings: ['No company-specific comparison was established.']
};
vm.runInContext('render(sample)', context);
assert.ok(dashboardNodes().some(node => node.textContent === 'Background information report'));
assert.ok(dashboardNodes().some(node => node.textContent.includes('Important events may be missing.')));
assert.ok(dashboardNodes().some(node => node.textContent === 'No company-specific comparison was established.'));
const backgroundHtml = vm.runInContext('buildReportHtml(sample)', context);
assert.ok(backgroundHtml.includes('Background information report'));
assert.ok(backgroundHtml.includes('No company-specific comparison was established.'));
assert.ok(backgroundHtml.includes('href="https://example.com/press"'));

context.sample = {
  ...originalSample,
  executive_summary: null,
  insight_items: [{category: 'Product & Innovation', text: 'ExampleCo launched a new product.', sources: originalSample.sources}]
};
vm.runInContext('render(sample)', context);
assert.ok(dashboardNodes().some(node => node.textContent === 'From retained findings'));
const fallbackHtml = vm.runInContext('buildReportHtml(sample)', context);
assert.ok(fallbackHtml.includes('Assembled from retained findings.'));
assert.ok(fallbackHtml.includes('ExampleCo launched a new product.'));
assert.equal(context.sample.executive_summary, null, 'Presentation fallback must not alter API data');

context.sample = {
  ...context.sample,
  insight_items: [{
    event: 'Product launch', category: 'Product & Innovation',
    text: 'ExampleCo launched a new product.', sources: originalSample.sources,
    evidence: [{source_id: 'S1', title: 'Official source', url: 'https://example.com/press',
                quote: 'ExampleCo launched a product. <img src=x onerror=alert(1)>'}]
  }]
};
vm.runInContext('render(sample)', context);
assert.ok(dashboardNodes().some(node => node.tag === 'blockquote' && node.textContent.includes('<img src=x')));
assert.ok(dashboardNodes().some(node => node.textContent === 'Product launch'));
const quotedHtml = vm.runInContext('buildReportHtml(sample)', context);
assert.ok(quotedHtml.includes('Quoted source excerpts'));
assert.ok(quotedHtml.includes('&lt;img src=x onerror=alert(1)&gt;'));
assert.ok(!quotedHtml.includes('<img src=x'));

// Structured source briefs must not export unselected details or model drafts.
const crypto = require('node:crypto');
const fundingSource = {title: 'Company source', url: 'https://example.com/funding',
  content: '# ExampleCo\n- Rounds: 15\n- Total raised: 169.37M USD\n- Last round: Corporate Round 2025-07-25', published_at: null};
const binding = crypto.createHash('sha256').update(JSON.stringify(['S1', fundingSource.title, fundingSource.url, fundingSource.content])).digest('hex').slice(0, 16);
const selected = ['- Rounds: 15', '- Last round: Corporate Round 2025-07-25'].map(quote => ({
  source_id: 'S1', quote, title: fundingSource.title, url: fundingSource.url,
  evidence_id: `S1-${binding}-${fundingSource.content.indexOf(quote)}-${fundingSource.content.indexOf(quote) + quote.length}`
}));
context.sourceSample = {...originalSample, analysis_format: 'evidence_ids_v2', analysis_status: 'complete',
  analysis_window: {start: '2026-08-20', end: '2026-11-18'},
  sources: [fundingSource], report: 'Unsafe generated total 169.37M and recently completed.',
  executive_summary: {text: 'Unsafe generated total 169.37M.', sources: [fundingSource]},
  insight_items: [{text: 'ExampleCo raised 169.37M USD recently.', event: 'Unsafe label', category: 'Funding & Investment',
    sources: [{title: fundingSource.title, url: fundingSource.url}], evidence: selected}],
  competitors: [], context_items: []};
const sourceHtml = vm.runInContext('buildReportHtml(sourceSample)', context);
assert.ok(!sourceHtml.includes('169.37M'), 'Unselected amount or raw model draft leaked into shared report');
assert.ok(!sourceHtml.includes('recently completed'), 'Generated recency wording leaked into shared report');
assert.ok(sourceHtml.includes('Rounds: 15') && sourceHtml.includes('2025-07-25'), 'Supported selected details were lost');
assert.ok(!sourceHtml.includes('Full Report Text'), 'Raw model report must not enter source-only exports');
const selectedExport = JSON.parse(vm.runInContext('JSON.stringify(buildSelectedEvidenceExport(sourceSample))', context));
assert.ok(!JSON.stringify(selectedExport).includes('169.37M'));
assert.equal(selectedExport.observations[0].passages.length, 2);
vm.runInContext('render(sourceSample)', context);
assert.ok(!dashboardNodes().some(node => node.textContent === context.sourceSample.insight_items[0].text));
assert.ok(!dashboardNodes().some(node => /AI interpretation/.test(node.textContent)));
assert.ok(dashboardNodes().some(node => node.textContent.includes('Rounds: 15')));
assert.equal(vm.runInContext('safeUrl("https://user:password@example.com/secret")', context), null);
context.badComparison = {...context.sourceSample, competitors: [{name: 'Unverified alternative', overlap: 'Unsupported comparison',
  sources: [{title: fundingSource.title, url: fundingSource.url}], evidence: selected}]};
assert.equal(vm.runInContext('buildSelectedEvidenceExport(badComparison).comparison_passages.length', context), 0, 'S evidence cannot establish a C comparison');
context.tamperedSource = JSON.parse(JSON.stringify(context.sourceSample));
context.tamperedSource.insight_items[0].evidence[0].evidence_id = 'S1-0000000000000000-0-1';
assert.equal(vm.runInContext('buildSelectedEvidenceExport(tamperedSource).observations.length', context), 0, 'Incorrect span bounds must be withheld');
context.badStatus = {...context.sourceSample, analysis_status: 'malformed'};
assert.throws(() => vm.runInContext('buildSelectedEvidenceExport(badStatus)', context));
const unicodeSource = {title: 'Unicode source', url: 'https://example.com/unicode', content: '\ud83d\ude00 Context\nExampleCo launched a product.', published_at: null};
const unicodeQuote = 'ExampleCo launched a product.';
const unicodeStart = Array.from(unicodeSource.content.split('\n')[0]).length + 1;
const unicodeBinding = crypto.createHash('sha256').update(JSON.stringify(['S1', unicodeSource.title, unicodeSource.url, unicodeSource.content])).digest('hex').slice(0,16);
context.unicodeSample = {...context.sourceSample, sources: [unicodeSource], insight_items: [{text: 'Unsupported embellishment', sources: [unicodeSource],
  evidence: [{source_id: 'S1', quote: unicodeQuote, title: unicodeSource.title, url: unicodeSource.url,
    evidence_id: `S1-${unicodeBinding}-${unicodeStart}-${unicodeStart + Array.from(unicodeQuote).length}`}]}]};
assert.equal(vm.runInContext('buildSelectedEvidenceExport(unicodeSample).observations[0].passages[0].quote', context), unicodeQuote, 'Python code-point offsets must preserve emoji-prefixed evidence');

context.sample = {company_name: 'ExampleCo', status: 'completed'};
vm.runInContext('render(sample)', context);
assert.ok(dashboardNodes().some(node => node.textContent.includes('Accepted company source count unavailable')));

context.sample = originalSample;
vm.runInContext('render(sample, true)', context);

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
  assert.ok(nodes.output.children.some(node =>
    node.textContent === errorMessage
  ));

  if (malformedAnalysis) {
    nodes.downloadReportButton.handlers.click();
    assert.equal(blobs.length, 1, 'Malformed analysis must not export stale data');
    return;
  }

  const warning = process.argv[2] === 'partial'
    ? 'Some generated findings were excluded because they could not be validated.'
    : 'Insufficient evidence for a company overview.';
  context.sample = {
    ...context.sample,
    analysis_status: process.argv[2],
    analysis_warnings: process.argv[2] === 'partial' ? [warning] : [],
    ...(process.argv[2] === 'insufficient_evidence'
      ? {executive_summary: null, insight_items: []}
      : {})
  };
  vm.runInContext('render(sample)', context);
  const containsWarning = node => node.textContent === warning
    || (node.children || []).some(containsWarning);
  assert.ok(containsWarning(nodes.output), 'Missing dashboard analysis notice');
  const exportHtml = vm.runInContext('buildReportHtml(sample)', context);
  assert.ok(exportHtml.includes(warning), 'Missing exported analysis notice');
  assert.ok(exportHtml.includes('href="https://example.com/press"'));
  if (process.argv[2] === 'insufficient_evidence') {
    assert.ok(exportHtml.includes('Evidence-only report'));
  }
  assert.equal(nodes.downloadReportButton.disabled, false);

  process.stdout.write(
    'Export runtime checks passed\n'
  );

})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
"""


@pytest.mark.parametrize("analysis_status", ["partial", "insufficient_evidence"])
def test_export_runtime_and_stale_report_guard(analysis_status):
    run_export_runtime_check(analysis_status, "generic")


def test_malformed_analysis_clears_stale_report():
    run_export_runtime_check("complete", "malformed")


def run_export_runtime_check(analysis_status, error_kind):
    node = shutil.which("node")

    if node is None:
        pytest.skip(
            "Node.js not installed; runtime export test unavailable"
        )

    result = subprocess.run(
        [node, "-e", NODE_CHECK, str(INDEX), analysis_status, error_kind],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, (
        result.stderr or result.stdout
    )
