"""Run the homepage language controller without a browser dependency."""

import subprocess


def test_home_language_switch_preserves_preference_and_translates_every_label():
    result = subprocess.run(['node', '-e', r'''
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('static/home.html', 'utf8');
const source = fs.readFileSync('static/home.js', 'utf8');
const labels = [...html.matchAll(/data-i18n="([^"]+)"/g)].map(match => ({ dataset: { i18n: match[1] } }));
const ariaLabels = [...html.matchAll(/data-i18n-aria="([^"]+)"/g)].map(match => ({
    dataset: { i18nAria: match[1] }, setAttribute(key, value) { this[key] = value; }
}));
const store = {};
function boot(storage) {
    let click;
    const button = { dataset: {}, setAttribute() {}, addEventListener(event, fn) { click = fn; } };
    const document = {
        documentElement: {},
        querySelector: () => ({}),
        querySelectorAll: selector => selector === '[data-i18n]' ? labels : ariaLabels,
        getElementById: () => button,
    };
    const context = { document, localStorage: storage };
    vm.runInNewContext(source, context);
    return { document, click: () => click() };
}
const storage = { getItem: key => store[key], setItem: (key, value) => store[key] = value };
let app = boot(storage);
if (app.document.documentElement.lang !== 'zh-CN') throw Error('Chinese default missing');
app.click();
if (store.lang !== 'en' || app.document.documentElement.lang !== 'en') throw Error('Switch failed');
if (labels.some(el => !el.textContent) || ariaLabels.some(el => !el['aria-label'])) throw Error('Translation missing');
app = boot(storage);
if (app.document.documentElement.lang !== 'en') throw Error('Preference lost');
app = boot({ getItem() { throw Error('storage denied'); }, setItem() { throw Error('storage denied'); } });
app.click();
if (app.document.documentElement.lang !== 'en') throw Error('Unavailable storage broke language switching');
console.log('passed');
'''], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == 'passed'
