"""Exercise completed-task result loading and rendering with the real browser script."""

import subprocess


def test_task_result_uses_exact_task_and_renders_safe_bilingual_segments():
    result = subprocess.run(["node", "-e", r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const nodes = Object.fromEntries(['result-title', 'result-meta', 'result-content'].map(id => [id, {}]));
const pending = [];
const context = {
  document: {getElementById: id => nodes[id]},
  fetch: url => new Promise(resolve => pending.push({url, resolve})),
  escHtml: value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
};
vm.createContext(context);
vm.runInContext(fs.readFileSync('static/admin-results.js', 'utf8'), context);
(async () => {
  const old = context.loadTaskResult('7');
  const current = context.loadTaskResult('8');
  assert.equal(pending[1].url, '/api/tasks/8');
  pending[1].resolve({ok:true, json:async () => ({id:8, status:'done', item_name:'Movie', video_path:'/media/a.mkv',
    source_segments:[{start:0,end:1,text:'<img src=x onerror=alert(1)>'}],
    translated_segments:[{start:0,end:1,text:'你好 & 再见'}]})});
  await current;
  assert.equal(nodes['result-title'].textContent, 'Movie');
  assert(nodes['result-content'].innerHTML.includes('&lt;img'));
  assert(!nodes['result-content'].innerHTML.includes('<img'));
  assert(nodes['result-content'].innerHTML.includes('你好 &amp; 再见'));
  assert(nodes['result-content'].innerHTML.includes('00:00:01.000'));
  pending[0].resolve({ok:true, json:async () => ({status:'done', item_name:'Stale', source_segments:[]})});
  await old;
  assert.equal(nodes['result-title'].textContent, 'Movie');
  const empty = context.loadTaskResult('9');
  pending[2].resolve({ok:true,json:async () => ({id:9,status:'done',source_segments:[],translated_segments:[]})});
  await empty;
  assert(nodes['result-content'].textContent.includes('没有可查看的字幕'));
  const missing = context.loadTaskResult('10');
  pending[3].resolve({ok:false,status:404});
  await missing;
  assert(nodes['result-content'].textContent.includes('任务不存在'));
  await context.loadTaskResult('not-an-id');
  assert.equal(pending.length, 4);
  assert(nodes['result-content'].textContent.includes('无效'));
  console.log('passed');
})().catch(e => {console.error(e);process.exitCode=1;});
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "passed"
