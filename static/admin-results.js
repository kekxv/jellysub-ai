/* Load the selected task's snapshot, even when a newer task exists for its video. */
let taskResultRequest = 0;

function formatResultTime(seconds) {
    const ms = Math.max(0, Math.round((Number(seconds) || 0) * 1000));
    const h = String(Math.floor(ms / 3600000)).padStart(2, '0');
    const m = String(Math.floor(ms / 60000) % 60).padStart(2, '0');
    const s = String(Math.floor(ms / 1000) % 60).padStart(2, '0');
    return `${h}:${m}:${s}.${String(ms % 1000).padStart(3, '0')}`;
}

async function loadTaskResult(taskId) {
    const request = ++taskResultRequest;
    const title = document.getElementById('result-title');
    const meta = document.getElementById('result-meta');
    const content = document.getElementById('result-content');
    title.textContent = '字幕结果';
    meta.textContent = '';
    content.textContent = '正在加载字幕…';
    if (!/^[1-9]\d*$/.test(String(taskId))) {
        content.textContent = '任务编号无效，请返回任务中心重新选择。';
        return;
    }
    try {
        const response = await fetch(`/api/tasks/${taskId}`);
        if (!response.ok) {
            throw new Error(response.status === 404 ? '任务不存在，可能已被删除。' :
                response.status === 401 ? '登录已过期，请重新登录后查看。' : '字幕加载失败，请稍后重试。');
        }
        const task = await response.json();
        if (request !== taskResultRequest) return;
        title.textContent = task.item_name || '字幕结果';
        meta.textContent = `任务 #${task.id} · ${task.video_path || '未关联媒体路径'}`;
        if (task.status !== 'done') {
            content.textContent = '该任务尚未完成，请返回任务中心查看进度。';
            return;
        }
        const source = Array.isArray(task.source_segments) ? task.source_segments : [];
        const translated = Array.isArray(task.translated_segments) ? task.translated_segments : [];
        const count = Math.max(source.length, translated.length);
        if (!count) {
            content.textContent = '该任务没有可查看的字幕内容。';
            return;
        }
        meta.textContent += ` · ${count} 条字幕`;
        content.innerHTML = `<div class="result-column-labels" aria-hidden="true"><span>时间轴</span><span>原文</span><span>译文</span></div>` +
            Array.from({length: count}, (_, index) => {
                const segment = source[index] || translated[index];
                return `<article class="result-segment">
                    <div class="result-time"><span>#${index + 1}</span>${formatResultTime(segment.start)}<br>${formatResultTime(segment.end)}</div>
                    <div class="result-source"><span class="result-mobile-label">原文</span>${escHtml(source[index]?.text || '—')}</div>
                    <div class="result-translation"><span class="result-mobile-label">译文</span>${escHtml(translated[index]?.text || '—')}</div>
                </article>`;
            }).join('');
    } catch (error) {
        if (request === taskResultRequest) content.textContent = error.message || '字幕加载失败，请稍后重试。';
    }
}

function openTaskResult(taskId) {
    AdminUI.showView('result', {taskId});
}
