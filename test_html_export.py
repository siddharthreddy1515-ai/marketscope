
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

const [pagePath, analysisStatus, errorKind] = process.argv.slice(2);
const page = fs.readFileSync(pagePath, 'utf8');

const script = page.match(
  /<script>([\s\S]*?)<\/script>/i
);

assert.ok(script, 'Expected an inline dashboard script');

const nodes = {};
const downloads = [];
const blobs = [];
let fetchCalls = 0;
const timedOut = errorKind === 'timed_out_with_sources';
const retainedResearch = errorKind === 'malformed_with_sources' || timedOut;
const malformedAnalysis = errorKind === 'malformed_with_sources' || errorKind === 'malformed';
const errorMessage = timedOut ? 'The analysis provider timed out.' : malformedAnalysis
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
      status: timedOut ? 504 : malformedAnalysis ? 502 : 500,
      json: async () => ({
        detail: errorMessage,
        ...(retainedResearch ? {research: {
          ...(timedOut ? {analysis_warnings: [errorMessage]} : {}),
          company_name: 'Tesla', sources: [{
            title: 'Saved company page', url: 'https://example.com/collected',
            content: 'Retrieved company description remains available.', published_at: '2026-11-04',
          }],
          comparison_sources: [{title: 'Comparison article', url: 'https://example.com/comparison-article'}],
        }} : {}),
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
assert.deepEqual(dashboardNodes().filter(node => node.className === 'metric-label').map(node => node.textContent),
  ['Core offering', 'Recent developments', 'Research coverage', 'Competitive alternatives']);
const metricLinks = dashboardNodes().filter(node => node.className === 'metric-jump');
assert.deepEqual(metricLinks.map(node => node.href), ['#company-snapshot', '#company-findings', '#company-research', '#company-comparisons']);
for (const card of metricLinks) {
  assert.equal(card.tag, 'a');
  assert.equal(dashboardNodes().filter(node => '#' + node.id === card.href).length, 1);
}
const libraryTarget = dashboardNodes().find(node => node.id === 'company-research');
nodes['company-research'] = libraryTarget;
metricLinks[2].handlers.click();
assert.equal(libraryTarget.open, true);
assert.ok(descendants(metricLinks[2]).some(node => node.textContent === 'View source library'));
assert.ok(!dashboardNodes().some(node => ['Market Position', 'Threat Level'].includes(node.textContent)));
assert.ok(dashboardNodes().some(node => node.textContent === 'Executive Snapshot'));
assert.ok(dashboardNodes().some(node => node.textContent === 'Event dates unavailable'));

const originalSample = context.sample;
const datedSentence = 'On October 2, 2026, ExampleCo opened a new distribution center.';
const makeDated = (quote) => {
  const source = {source_id: 'S1', title: 'News', url: 'https://example.com/news', content: quote};
  return {...originalSample, analysis_window: {start: '2026-08-20', end: '2026-11-18'},
    sources: [source], insight_items: [{text: quote, sources: [source], evidence: [{...source, quote}]}]};
};
for (const [quote, expected] of [[datedSentence,'2026-10-02'],
    ['ExampleCo launched drone delivery in Houston on November 17, 2026.', '2026-11-17'],
    ['ExampleCo plans to launch drone delivery in Houston on November 17, 2026.', null],
    ['ExampleCo launched a service announced by its partner on November 17, 2026.', null],
    ['ExampleCo launched a service after its partner entered the market on November 17, 2026.', null],
    ['ExampleCo launched a service for an expo on November 17, 2026.', null],
    ['Another company acquired ExampleCo on November 17, 2026.', null],
    [datedSentence.replace('opened','may open'),null],
    [datedSentence.replace('October','July'),null],
    [datedSentence.replace('October 2','February 31'),null],
    [datedSentence.replace('ExampleCo opened','another company acquired ExampleCo and opened'),null]]) {
  context.dated = makeDated(quote);
  assert.equal(vm.runInContext('recentAnnouncement(dated.insight_items[0],dated)?.day || null', context),expected);
}

const plan = 'ExampleCo plans to open a battery assembly plant.';
context.reportedPlan = makeDated(plan);
context.reportedPlan.sources[0].published_at = 'Sat, 10 Oct 2026 06:33:33 GMT';
context.reportedPlan.research_coverage = {status: 'uncertain'};
vm.runInContext('render(reportedPlan)', context);
const planNodes = dashboardNodes();
const planCard = planNodes.find(node => node.className === 'insight');
assert.ok(descendants(planCard).some(node => node.textContent === plan));
assert.ok(descendants(planCard).some(node => node.textContent === 'Source published 10 Oct 2026'));
assert.ok(descendants(planCard).some(node => node.textContent === 'Event date unconfirmed'));
assert.ok(descendants(planCard).some(node => node.href === 'https://example.com/news'));
assert.ok(planNodes.indexOf(planCard) < planNodes.findIndex(node => node.className === 'source-notes'));
assert.ok(planNodes.some(node => node.textContent === 'Event dates unavailable'));
assert.equal(planNodes.filter(node => node.className === 'metric-value')[1].textContent, '—');
assert.equal(planNodes.filter(node => node.className === 'metric-note')[1].textContent,
  '1 observation with an unconfirmed date');
assert.equal(planNodes.filter(node => node.className === 'metric-note')[2].textContent,
  '1 cited in findings or context · Limited coverage');

const datedReport = makeDated(datedSentence);
context.mixedDates = {
  ...context.reportedPlan, sources: [
    ...context.reportedPlan.sources,
    {...datedReport.sources[0], url: 'https://example.com/opening', published_at: '2026-10-03'},
  ], insight_items: [
    ...context.reportedPlan.insight_items,
    {...datedReport.insight_items[0], evidence: [
      {...datedReport.insight_items[0].evidence[0], source_id: 'S2', url: 'https://example.com/opening'},
    ]},
  ],
};
vm.runInContext('render(mixedDates)', context);
const mixedNodes = dashboardNodes();
const mixedTitles = mixedNodes.filter(node => node.className === 'insight-title');
assert.deepEqual(mixedTitles.map(node => node.textContent), [datedSentence, plan]);
assert.ok(mixedNodes.indexOf(mixedTitles[1]) < mixedNodes.findIndex(node => node.className === 'source-notes'));
assert.equal(mixedNodes.filter(node => node.className === 'metric-value')[1].textContent, '1');
assert.equal(mixedNodes.filter(node => node.className === 'metric-note')[1].textContent,
  'Dated in the report window · 1 observation with an unconfirmed date');
const dateProof = descendants(mixedNodes.filter(node => node.className === 'brief-metric')[1]);
const proofGroups = dateProof.filter(node => node.className === 'metric-proof-group');
assert.deepEqual(proofGroups.map(group => group.children[0].textContent),
  ['Dated developments', 'Observations with unconfirmed dates']);
const datedProof = descendants(proofGroups[0]);
assert.ok(datedProof.some(node => node.href === 'https://example.com/opening'));
assert.ok(datedProof.some(node => node.tag === 'blockquote' && node.textContent === datedSentence));
assert.ok(!datedProof.some(node => node.href === 'https://example.com/news'));
const undatedProof = descendants(proofGroups[1]);
assert.ok(undatedProof.some(node => node.href === 'https://example.com/news'));
assert.ok(undatedProof.some(node => node.tag === 'blockquote' && node.textContent === plan));

context.reportedPlan.sources[0].published_at = 'invalid';
vm.runInContext('render(reportedPlan)', context);
assert.ok(!dashboardNodes().some(node => node.textContent.startsWith('Source published')));

const companyDescription = 'ExampleCo makes electric motorcycles.';
const originalCitation = { title: 'Company products', url: 'https://example.com/products' };
const overviewEvidence = {
  ...originalCitation, source_id: 'S1', quote: companyDescription,
  evidence_id: 'S1-0000000000000000-0-' + companyDescription.length,
};
context.sample = {
  ...originalSample, analysis_format: 'evidence_ids_v2', analysis_status: 'partial',
  source_stats: {returned: 2, accepted: 2, discarded: 0},
  sources: [{ ...originalCitation, content: companyDescription }, {
    title: 'Unselected retail page', url: 'https://example.com/retail',
    content: 'DatabaseChargesRegulatory AlertsLitigationsDetailed FinancialsCredit RatingsGlobal BrandsFunding\nThe retailer offers a commuter and a cargo model.',
  }],
  insight_items: [], context_items: [{
    event: 'Company overview', text: 'Unsupported overview prose.', evidence: [overviewEvidence],
  }],
};
vm.runInContext('render(sample)', context);
const overviewNodes = dashboardNodes();
assert.ok(overviewNodes.some(node => node.className === 'snapshot-text' && node.textContent === companyDescription));
assert.ok(overviewNodes.some(node => node.tag === 'a' && node.href === originalCitation.url));
assert.ok(!overviewNodes.some(node => node.textContent === 'Unsupported overview prose.'));
assert.ok(overviewNodes.some(node => node.textContent === 'The retailer offers a commuter and a cargo model.'));
assert.ok(!overviewNodes.some(node => node.className === 'source-preview' && node.textContent.includes('DatabaseCharges')));
assert.ok(overviewNodes.some(node => node.tag === 'a' && node.href === 'https://example.com/retail'));
assert.ok(overviewNodes.findIndex(node => node.textContent === 'Executive Snapshot') <
  overviewNodes.findIndex(node => node.textContent === 'No source-supported developments available.'));
assert.equal(overviewNodes.find(node => node.className === 'source-notes').children.length, 2);
const coverageMetric = descendants(overviewNodes.filter(node => node.className === 'brief-metric')[2]);
assert.ok(coverageMetric.some(node => node.className === 'metric-value' && node.textContent === '2 pages'));
assert.ok(coverageMetric.some(node => node.className === 'metric-note' && node.textContent === '1 cited in findings or context'));
const citedFilter = overviewNodes.find(node => node.type === 'checkbox');
const librarySearch = overviewNodes.find(node => node.type === 'search');
citedFilter.checked = true;
citedFilter.handlers.change();
assert.equal(dashboardNodes().filter(node => node.className === 'library-source').length, 1);
assert.ok(dashboardNodes().filter(node => node.className === 'library-source')
  .some(row => descendants(row).some(node => node.href === originalCitation.url)));
librarySearch.value = 'Unselected retail page';
librarySearch.handlers.input();
assert.equal(dashboardNodes().filter(node => node.className === 'library-source').length, 0);
citedFilter.checked = false;
citedFilter.handlers.change();
assert.equal(dashboardNodes().filter(node => node.className === 'library-source').length, 1);
librarySearch.value = '';
librarySearch.handlers.input();
assert.equal(dashboardNodes().filter(node => node.className === 'library-source').length, 2);

const longExcerpt = 'Company profile. ' + companyDescription + ' ' +
  'Additional information appears in the source. '.repeat(18).trimEnd();
context.longSnapshot = {
  ...context.sample,
  sources: [{...context.sample.sources[0], content: longExcerpt}, context.sample.sources[1]],
  context_items: [{...context.sample.context_items[0], text: 'ExampleCo makes electric motorcycles for an unknown customer.',
    evidence: [{...overviewEvidence, quote: longExcerpt, evidence_id: 'S1-0000000000000000-0-' + longExcerpt.length}]}],
};
vm.runInContext('render(longSnapshot)', context);
assert.ok(dashboardNodes().some(node => node.className === 'snapshot-text' && node.textContent === companyDescription));
assert.ok(dashboardNodes().some(node => node.tag === 'blockquote' && node.textContent === longExcerpt));
assert.ok(!dashboardNodes().some(node => node.textContent.includes('unknown customer')));

const productText = context.sample.sources[1].content.split('\n').at(-1);
const productStart = context.sample.sources[1].content.indexOf(productText);
context.sample.context_items.push({
  event: 'Products and services', text: productText,
  evidence: [{source_id: 'S2', evidence_id: 'S2-0000000000000000-' + productStart + '-' + (productStart + productText.length),
    url: 'https://example.com/retail', title: 'Unselected retail page', quote: productText}],
});
vm.runInContext('render(sample)', context);
const productSection = dashboardNodes().find(node => node.className === 'snapshot-offering');
assert.ok(descendants(productSection).some(node => node.textContent === productText));
assert.ok(descendants(productSection).some(node => node.tag === 'a' && node.href === 'https://example.com/retail'));
assert.equal(vm.runInContext('buildSelectedEvidenceExport(sample).context.length', context), 2);

const agreement = 'ExampleCo signed a supply agreement with Atlas.';
const opening = 'ExampleCo opened a warehouse in Austin.';
const sharedExcerpt = agreement + ' ' + opening;
const sharedSource = {title: 'Company update', url: 'https://example.com/update', content: sharedExcerpt};
const sharedProof = {...sharedSource, source_id: 'S1', quote: sharedExcerpt,
  evidence_id: 'S1-0000000000000000-0-' + sharedExcerpt.length};
const agreementItem = {text: agreement, category: 'M&A / Partnerships', evidence: [sharedProof]};
context.sharedEvidence = {...context.sample, sources: [sharedSource], context_items: [], insight_items: [
  agreementItem,
  {text: opening, category: 'Market Expansion', evidence: [sharedProof]},
  {...agreementItem, text: '  ' + agreement + '  '},
]};
vm.runInContext('render(sharedEvidence)', context);
const sharedCards = dashboardNodes().filter(node => node.className === 'insight').map(descendants);
assert.deepEqual(sharedCards.map(card => card.find(node => node.className === 'insight-title').textContent), [agreement, opening]);
assert.ok(sharedCards[0].some(node => node.textContent === 'M&A / Partnerships'));
assert.ok(sharedCards[1].some(node => node.textContent === 'Market Expansion'));
assert.equal(vm.runInContext('buildSelectedEvidenceExport(sharedEvidence).observations.length', context), 2);

// A selected catalog field can share a passage with specifications and marketing copy.
const catalogList = 'Folding bicycles, Cargo bicycles, Electric bicycles, Connected diagnostics app';
const catalogQuote = 'Products: ' + catalogList + ' [...] Description: Models have different maximum speeds.';
const catalogReport = JSON.parse(JSON.stringify(context.sample));
catalogReport.sources[1].content = catalogQuote;
const catalogItem = catalogReport.context_items[1];
catalogItem.text = 'Unsupported catalog paraphrase';
Object.assign(catalogItem.evidence[0], {quote: catalogQuote,
  evidence_id: 'S2-0000000000000000-0-' + catalogQuote.length});
context.catalogReport = catalogReport;
vm.runInContext('render(catalogReport)', context);
assert.equal(dashboardNodes().find(n => n.className === 'metric-value').textContent, 'Folding bicycles');
const offeringProof = descendants(dashboardNodes().find(node => node.className === 'brief-metric'));
assert.ok(offeringProof.some(node => node.href === 'https://example.com/retail'));
assert.ok(offeringProof.some(node => node.tag === 'blockquote' && node.textContent === catalogQuote));
assert.ok(!offeringProof.some(node => node.href === 'https://example.com/products'));
const catalogSnapshot = descendants(dashboardNodes().find(n => n.className === 'snapshot-offering'));
assert.ok(catalogSnapshot.some(n => n.className === 'snapshot-text' && n.textContent === catalogList));
assert.ok(catalogSnapshot.some(n => n.tag === 'blockquote' && n.textContent === catalogQuote));

const backgroundPages = Array.from({ length: 7 }, (_, index) => ({
  title: 'Background page ' + index, url: 'https://example.net/background-' + index, content: 'Background text.',
}));
context.sample = {
  ...context.sample, official_domain: 'example.com',
  sources: [...backgroundPages, ...context.sample.sources],
  context_items: [{ ...context.sample.context_items[0], evidence: [{
    ...overviewEvidence, source_id: 'S8', evidence_id: overviewEvidence.evidence_id.replace(/^S1/, 'S8'),
  }] }],
};
vm.runInContext('render(sample)', context);
const sourceGrids = dashboardNodes().filter(node => node.className === 'source-notes');
assert.equal(sourceGrids[0].children.length, 6);
assert.ok(descendants(sourceGrids[0].children[0]).some(node => node.href === originalCitation.url));
assert.ok(descendants(sourceGrids[0].children[1]).some(node => node.href === 'https://example.com/retail'));
assert.equal(sourceGrids[1].children.length, 3);

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
assert.ok(visibleText.includes('Insufficient data'));
assert.ok(visibleText.includes('No selected developments to classify.'));
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
assert.ok(dashboardNodes().some(node => node.textContent === 'Legacy analysis note'));
const fallbackHtml = vm.runInContext('buildReportHtml(sample)', context);
assert.ok(fallbackHtml.includes('Assembled from retained findings.'));
assert.ok(fallbackHtml.includes('ExampleCo launched a new product.'));
assert.equal(context.sample.executive_summary, null, 'Presentation fallback must not alter API data');

context.sample = {
  ...context.sample,
  sources: [{...originalSample.sources[0], content: 'ExampleCo launched a product. <img src=x onerror=alert(1)>'}],
  insight_items: [{
    event: 'Product launch', category: 'Product & Innovation',
    text: 'ExampleCo launched a new product.', sources: originalSample.sources,
    evidence: [{source_id: 'S1', title: 'Official source', url: 'https://example.com/press',
                quote: 'ExampleCo launched a product. <img src=x onerror=alert(1)>'}]
  }]
};
vm.runInContext('render(sample)', context);
assert.ok(dashboardNodes().some(node => node.tag === 'blockquote' && node.textContent.includes('<img src=x')));
assert.ok(dashboardNodes().some(node => node.textContent === 'Product & Innovation'));
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
const evidenceDrawer = dashboardNodes().find(node => node.className === 'evidence-detail');
assert.ok(evidenceDrawer && !evidenceDrawer.open);
assert.deepEqual(descendants(evidenceDrawer).filter(node => node.tag === 'blockquote')
  .map(node => node.textContent), selected.map(item => item.quote));
assert.ok(descendants(evidenceDrawer).some(node => node.textContent === selected[0].evidence_id));

context.sentimentSample = {
  ...context.sourceSample,
  sources: [
    {url: 'https://example.com/one', content: 'Article one'},
    {url: 'https://example.org/two', content: 'Article two'},
    {url: 'https://example.net/three', content: 'Article three'}
  ],
  source_sentiments: [
    {source_id: 'S1', url: 'https://example.com/one', label: 'Positive'},
    {source_id: 'S2', url: 'https://example.org/two', label: 'Positive'},
    {source_id: 'S3', url: 'https://example.net/three', label: 'Negative'}
  ],
  sentiment_distribution: {positive: 0, neutral: 100, negative: 0, assessed: 100}
};
vm.runInContext('render(sentimentSample)', context);
assert.ok(dashboardNodes().some(node => node.textContent === '2 · 67%'));
assert.ok(dashboardNodes().some(node => node.textContent === '1 · 33%'));
assert.ok(!dashboardNodes().some(node => node.textContent === '100 · 100%'));
const classifications = () => JSON.parse(vm.runInContext('JSON.stringify(sourceSentiment(sentimentSample))', context));
assert.equal(classifications().assessed, 3);
context.sentimentSample.sources[2] = {url: 'https://www.example.com/one/?utm_source=copy#top', content: 'A different excerpt'};
context.sentimentSample.source_sentiments[2] = {source_id: 'S3', url: 'https://www.example.com/one/?utm_source=copy#top', label: 'Negative'};
assert.equal(classifications().assessed, 1, 'Conflicting labels on an alias are excluded');
vm.runInContext('render(sentimentSample)', context);
assert.ok(dashboardNodes().some(node => node.textContent === 'Insufficient data'));
assert.ok(!dashboardNodes().some(node => node.className === 'sentiment-legend-row'));
context.sentimentSample.sources[2] = {url: 'https://example.net/three', content: 'Article two'};
context.sentimentSample.source_sentiments[2] = {source_id: 'S3', url: 'https://example.net/three', label: 'Positive'};
assert.equal(classifications().assessed, 2, 'An identical excerpt at a different URL is not a third source');
context.sentimentSample.sources[2].content = 'Article three';
context.sentimentSample.source_sentiments[2].url = 'https://example.net/unretained';
assert.equal(classifications().assessed, 2, 'An assessment must match a retained company source');

context.backgroundEvidence = {...context.sourceSample,
  research_coverage: {status: 'background_only', limitations: ['Only background evidence was retained.']}};
vm.runInContext('render(backgroundEvidence)', context);
const developmentCard = dashboardNodes().filter(node => node.className === 'brief-metric')[1];
assert.ok(descendants(developmentCard).some(node => node.textContent === 'Background evidence only'));
assert.ok(!dashboardNodes().some(node => node.className === 'breakdown-item'));
assert.ok(dashboardNodes().some(node => node.textContent === 'Only background evidence was retained.'));
assert.ok(dashboardNodes().some(node => node.className === 'snapshot-label' && node.textContent === 'From the sources'));

const comparisonSource = {title: 'Comparison source', url: 'https://example.org/compare', source_id: 'C1',
  evidence_id: 'C1-0000000000000000-0-33', quote: 'Acme and Atlas are both discussed.'};
context.comparisonSample = {...context.sourceSample, competitors: [
  {name: 'Atlas', overlap: 'Unsupported positioning', sources: [comparisonSource], evidence: [comparisonSource]},
  {name: 'ATLAS', overlap: 'Duplicate name', sources: [comparisonSource], evidence: [comparisonSource]},
  {name: 'Invented competitor', overlap: 'Unsupported threat', sources: [comparisonSource], evidence: [comparisonSource]}
]};
vm.runInContext('render(comparisonSample)', context);
assert.ok(dashboardNodes().some(node => node.className === 'competitor-name' && node.textContent === 'Atlas'));
assert.ok(dashboardNodes().some(node => node.textContent === 'Source excerpt · Acme and Atlas are both discussed.'));
assert.ok(!dashboardNodes().some(node => ['Invented competitor', 'Unsupported positioning', 'Unsupported threat'].includes(node.textContent)));
assert.equal(vm.runInContext('comparisonEntries(comparisonSample)[0].supported', context), false);
const alternativeCard = descendants(dashboardNodes().filter(node => node.className === 'brief-metric')[3]);
assert.ok(alternativeCard.some(node => node.className === 'metric-value' && node.textContent === '1 mentioned'));
assert.ok(alternativeCard.some(node => node.textContent === 'Product overlap unconfirmed'));
assert.ok(alternativeCard.some(node => node.href === comparisonSource.url));
assert.ok(alternativeCard.some(node => node.tag === 'blockquote' && node.textContent === comparisonSource.quote));
assert.ok(dashboardNodes().some(node => node.tag === 'details' && node.open &&
  node.children[0].textContent.startsWith('Comparison mentions')));
const comparisonCases = [
  ["Harbor Logistics", "Acme faces competition from Orion, Harbor Logistics, and other freight operators.", "Harbor Logistics", true],
  ["Tern (Cedar Group)", "Project Tern, a freight carrier owned by Cedar Group, serves suburban customers.", "Tern", false],
  ["Atlas", "Atlas provides delivery services. Acme faces competition from Orion and Atlas.", "Atlas", true],
  ["Atlas", "Acme faces competition from Atlas Ventures and Orion.", "Atlas", false],
  ["Atlas", "Acme previously faced competition from Atlas and Orion.", "Atlas", false],
  ["Atlas", "Acme does not face competition from Atlas and Orion.", "Atlas", false],
];
for (const [name, quote, shownName, supported] of comparisonCases) {
  const source = {...comparisonSource, quote};
  context.explicitComparison = {...context.sourceSample, company_name: 'Acme', competitors: [
    {name, sources: [source], evidence: [source]},
  ]};
  const entry = vm.runInContext('comparisonEntries(explicitComparison)[0]', context);
  assert.equal(entry.name, shownName);
  assert.equal(entry.supported, supported);
  if (supported) assert.match(entry.preview, /faces competition from/);
}
assert.equal(vm.runInContext("sourceKind('https://mediacenter.example.com/story', 'in.example.com', 'Example')", context), 'Corporate site');
assert.equal(vm.runInContext("sourceKind('https://elsewhere.github.io/story', 'in.github.io', 'Example')", context), 'External source');
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
  if (retainedResearch) {
    assert.ok(dashboardNodes().some(node => node.textContent === 'Analysis unavailable. The collected sources are shown below.'));
    assert.ok(dashboardNodes().some(node => node.className === 'source-preview' &&
      node.textContent === 'Retrieved company description remains available.'));
    assert.ok(dashboardNodes().some(node => node.tag === 'a' && node.href === 'https://example.com/collected'));
    assert.ok(dashboardNodes().some(node => node.tag === 'a' && node.href === 'https://example.com/comparison-article'));
    assert.ok(dashboardNodes().some(node => node.className === 'timeline-date' && node.textContent.includes('2026')));
    assert.equal(dashboardNodes().find(node => node.className === 'publication-detail').open, true);
    assert.ok(dashboardNodes().some(node => node.textContent === 'Event dates unavailable'));
    assert.equal(nodes.downloadEvidenceButton.disabled, true);
    if (timedOut) assert.ok(dashboardNodes().some(node => node.textContent === errorMessage));
  } else {
    assert.ok(nodes.output.children.some(node => node.textContent === errorMessage));
  }

  if (malformedAnalysis || timedOut) {
    nodes.downloadReportButton.handlers.click();
    assert.equal(blobs.length, 1, 'A failed analysis must not export stale data');
    return;
  }

  const warning = analysisStatus === 'partial'
    ? 'Some generated findings were excluded because they could not be validated.'
    : 'Insufficient evidence for a company overview.';
  context.sample = {
    ...context.sample,
    analysis_status: analysisStatus,
    analysis_warnings: analysisStatus === 'partial' ? [warning] : [],
    ...(analysisStatus === 'insufficient_evidence'
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
  if (analysisStatus === 'insufficient_evidence') {
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


def test_malformed_analysis_keeps_collected_sources_visible():
    run_export_runtime_check("complete", "malformed_with_sources")


def test_timed_out_analysis_keeps_collected_sources_visible():
    run_export_runtime_check("complete", "timed_out_with_sources")


def run_export_runtime_check(analysis_status, error_kind):
    node = shutil.which("node")

    if node is None:
        pytest.skip(
            "Node.js not installed; runtime export test unavailable"
        )

    result = subprocess.run(
        [node, "-", str(INDEX), analysis_status, error_kind],
        input=NODE_CHECK,
        encoding="utf-8",
        capture_output=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, (
        result.stderr or result.stdout
    )
