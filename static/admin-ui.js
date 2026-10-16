/* Settings panels remain mounted so switching tabs never discards edits. */
function selectSettingsTab(name) {
    const tabs = [...document.querySelectorAll('[data-settings-tab]')];
    if (!tabs.some(tab => tab.dataset.settingsTab === name)) name = 'engines';
    tabs.forEach(tab => {
        const active = tab.dataset.settingsTab === name;
        tab.setAttribute('aria-selected', String(active));
        tab.setAttribute('tabindex', active ? '0' : '-1');
        document.getElementById('settings-' + tab.dataset.settingsTab).hidden = !active;
    });
}

/* Query-string routes coexist with the existing #videos= folder location. */
const ADMIN_VIEWS = {
    media: { label: '媒体库', title: '字幕工作台', description: '从媒体到字幕，一处轻松管理。' },
    tasks: { label: '任务中心', title: '任务中心', description: '跟踪字幕处理进度，查看结果或重试失败任务。' },
    settings: { label: '系统设置', title: '系统设置', description: '配置处理引擎、媒体目录与 Jellyfin 连接。' },
    result: { label: '任务中心', title: '字幕结果', description: '按时间轴对照查看已完成任务的原文与译文。' },
    webhook: { label: 'Webhook 教程', title: 'Webhook 设置教程', description: '连接 Jellyfin，让新入库的媒体自动生成字幕。' },
};

function showAdminView(name, { tab, taskId, historyMode = 'push', focus = true } = {}) {
    if (!Object.hasOwn(ADMIN_VIEWS, name)) name = 'media';
    const url = new URL(window.location.href);
    const oldView = document.body.dataset.view;
    if (name === 'settings') selectSettingsTab(tab || url.searchParams.get('tab') || 'engines');
    url.searchParams.set('view', name);
    if (name === 'settings') {
        const activeTab = document.querySelector('[data-settings-tab][aria-selected="true"]');
        url.searchParams.set('tab', activeTab.dataset.settingsTab);
    } else url.searchParams.delete('tab');
    if (name === 'result') url.searchParams.set('task', taskId || url.searchParams.get('task') || '');
    else url.searchParams.delete('task');
    if (historyMode !== 'none' && url.href !== window.location.href) {
        window.history[historyMode === 'replace' ? 'replaceState' : 'pushState']({}, '', url);
    }
    document.querySelectorAll('[data-admin-view]').forEach(panel => {
        panel.hidden = panel.dataset.adminView !== name;
    });
    document.querySelectorAll('[data-nav-view]').forEach(link => {
        const active = link.dataset.navView === (name === 'result' ? 'tasks' : name);
        link.classList.toggle('active', active);
        if (active) link.setAttribute('aria-current', 'page');
        else link.removeAttribute('aria-current');
    });
    document.body.dataset.view = name;
    document.getElementById('workspace-title').textContent = ADMIN_VIEWS[name].title;
    document.getElementById('workspace-section').textContent = ADMIN_VIEWS[name].label;
    document.getElementById('workspace-description').textContent = ADMIN_VIEWS[name].description;
    document.title = ADMIN_VIEWS[name].label + ' — JellySub-AI';
    document.getElementById('topbar-test-btn').hidden = ['settings', 'webhook', 'result'].includes(name);
    document.querySelectorAll('.mobile-account[open]').forEach(menu => menu.open = false);
    if (oldView !== name) {
        window.scrollTo(0, 0);
        if (focus) document.getElementById('main-content').focus({ preventScroll: true });
    }
    if (name === 'tasks') loadTasks().catch(() => {
        document.getElementById('task-history-summary-copy').textContent = '任务加载失败，请点击刷新重试。';
    });
    if (name === 'result') loadTaskResult(url.searchParams.get('task'));
    if (name === 'webhook') document.getElementById('webhook-endpoint').textContent = new URL('/webhook', window.location.origin).href;
}

// Overlay management: keyboard access, focus restoration and scroll containment.
(() => {
    window.AdminUI = {
        selectSettingsTab,
        showView: showAdminView,
        currentView: () => document.body.dataset.view || 'media',
    };
    document.querySelectorAll('[data-nav-view]').forEach(link => {
        link.addEventListener('click', event => {
            if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button !== 0) return;
            event.preventDefault();
            showAdminView(link.dataset.navView, { tab: 'engines' });
        });
    });
    const restoreView = () => {
        const params = new URLSearchParams(window.location.search);
        showAdminView(params.get('view') || 'media', { tab: params.get('tab'), historyMode: 'none', focus: false });
    };
    window.addEventListener('popstate', restoreView);
    restoreView();
    const tabs = [...document.querySelectorAll('[data-settings-tab]')];
    tabs.forEach((tab, index) => {
        tab.addEventListener('click', () => showAdminView('settings', { tab: tab.dataset.settingsTab, historyMode: 'replace' }));
        tab.addEventListener('keydown', event => {
            let next;
            if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
            if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
            if (event.key === 'Home') next = 0;
            if (event.key === 'End') next = tabs.length - 1;
            if (next === undefined) return;
            event.preventDefault();
            showAdminView('settings', { tab: tabs[next].dataset.settingsTab, historyMode: 'replace' });
            tabs[next].focus();
        });
    });

    // Only transient playback, preview and generation dialogs remain overlays.
    const closers = {
        'player-modal': () => closePlayer(),
        'sub-preview-modal': () => closeSubPreview(),
        'lang-modal': () => closeLangModal(),
    };
    const overlays = Object.keys(closers).map(id => document.getElementById(id)).filter(Boolean);
    const open = [];
    const previousFocus = new Map();
    const focusable = panel => [...panel.querySelectorAll(
        'button:not(:disabled), a[href], input:not(:disabled):not([type="hidden"]), select:not(:disabled), textarea:not(:disabled), [tabindex="0"]'
    )].filter(el => el.getClientRects().length && !el.closest('[inert]'));

    const syncOverlays = () => {
        overlays.forEach(panel => {
            const visible = panel.style.display !== 'none';
            if (visible && !open.includes(panel)) {
                previousFocus.set(panel, document.activeElement);
                open.push(panel);
                panel.inert = false;
                panel.setAttribute('aria-hidden', 'false');
                (focusable(panel)[0] || panel).focus();
            } else if (!visible && open.includes(panel)) {
                open.splice(open.indexOf(panel), 1);
                panel.setAttribute('aria-hidden', 'true');
                const previous = previousFocus.get(panel);
                if (previous?.isConnected) previous.focus();
                previousFocus.delete(panel);
            }
        });
        document.body.classList.toggle('overlay-open', open.length > 0);
    };
    const observer = new MutationObserver(syncOverlays);
    overlays.forEach(panel => {
        panel.setAttribute('role', 'dialog');
        panel.setAttribute('aria-modal', 'true');
        panel.setAttribute('tabindex', '-1');
        if (!panel.hasAttribute('aria-labelledby')) {
            const title = panel.querySelector('h2');
            if (title) {
                if (!title.id) title.id = panel.id + '-title';
                panel.setAttribute('aria-labelledby', title.id);
            }
        }
        observer.observe(panel, { attributes: true, attributeFilter: ['class', 'style'] });
        panel.addEventListener('click', event => {
            if (event.target === panel && panel.classList.contains('modal-overlay')) closers[panel.id]();
        });
    });
    syncOverlays();
    document.addEventListener('keydown', event => {
        const current = open.at(-1);
        if (!current) return;
        if (event.key === 'Escape') {
            event.preventDefault();
            closers[current.id]();
        } else if (event.key === 'Tab') {
            const items = focusable(current);
            const first = items[0], last = items.at(-1);
            if (!first) { event.preventDefault(); current.focus(); return; }
            if (event.shiftKey && (document.activeElement === first || !current.contains(document.activeElement))) {
                event.preventDefault(); last.focus();
            } else if (!event.shiftKey && (document.activeElement === last || !current.contains(document.activeElement))) {
                event.preventDefault(); first.focus();
            }
        }
    });

    // Hidden settings panels still participate in form validation. Bring an
    // invalid field into view before the browser focuses it.
    document.getElementById('config-form').addEventListener('invalid', event => {
        const panel = event.target.closest('[role="tabpanel"]');
        if (panel) selectSettingsTab(panel.id.replace('settings-', ''));
    }, true);

    // Make the original model selectors operable with keyboard and a screen reader.
    document.querySelectorAll('.tag-selector').forEach(group => {
        group.setAttribute('role', 'group');
        group.querySelectorAll('.tag-option').forEach(option => {
            option.setAttribute('role', 'button');
            option.tabIndex = 0;
            const update = () => option.setAttribute('aria-pressed', String(option.classList.contains('active')));
            update();
            new MutationObserver(update).observe(option, { attributes: true, attributeFilter: ['class'] });
            option.addEventListener('keydown', event => {
                if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault(); option.click();
                }
            });
        });
    });
})();
