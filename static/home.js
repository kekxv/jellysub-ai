const homeTranslations = {
    zh: {
        title: 'JellySub-AI — 自动化 AI 字幕生成', skip: '跳转到主要内容', navLabel: '主导航', navFeatures: '功能介绍', navAdmin: '管理后台',
        eyebrow: '为你的 Jellyfin 媒体库而生', heroTitle: '让每一段故事，', heroAccent: '都有字幕。',
        heroDesc: '从语音识别到多语言翻译，为你喜爱的视频自动生成字幕。把时间留给观影，其余交给 JellySub。',
        heroCTA: '打开字幕工作台', heroLearn: '了解工作流程', heroNote: '本地模型与在线服务，自由选择。',
        demoAria: '双语字幕效果示意', demoTitle: '一段旅程，两种语言', demoLabel: '字幕效果示意', demoCaption: '旅途的意义，是发现新的自己。', demoFooter: '让故事跨越语言，自然呈现。', bilingual: '双语字幕',
        capabilities: '支持的能力', capASR: '语音识别', capTranslate: '多语言翻译', capOffline: '离线处理', capBilingual: '双语 SRT',
        featuresEyebrow: '少一点操作，多一点享受', featuresTitle: '字幕这件事，可以很简单。',
        feat1Title: '入库，即刻开始', feat1Desc: '通过 Jellyfin Webhook 触发处理，也可以从工作台选择视频，按需生成字幕。', feat1Note: '自动触发 · 批量处理',
        feat2Title: '听懂，也看得懂', feat2Desc: '从音频识别或现有字幕开始，翻译为目标语言，生成单语或双语字幕。', feat2Note: '语音识别 · 多语言翻译',
        feat3Title: '用你喜欢的方式', feat3Desc: '使用本地模型离线处理，或连接兼容的在线 API。模型与语言，在工作台中随时调整。', feat3Note: '本地模型 · 在线 API',
        workflowEyebrow: '从视频到字幕', workflowTitle: '设置一次，让流程继续。',
        step1Title: '添加媒体目录', step1Desc: '连接 Jellyfin 或选择本地视频目录。', step2Title: '选择引擎与语言', step2Desc: '设置语音识别、翻译引擎与字幕语言。', step3Title: '交给字幕工作台', step3Desc: '查看处理进度，预览结果，随时继续观影。',
        closingEyebrow: '下一部好电影，准备好了', closingTitle: '让字幕，跟上你的片单。', footerTag: '为每一段故事，多一份理解。',
    },
    en: {
        title: 'JellySub-AI — Automated AI Subtitles', skip: 'Skip to content', navLabel: 'Main navigation', navFeatures: 'Features', navAdmin: 'Dashboard',
        eyebrow: 'Made for your Jellyfin library', heroTitle: 'Every story.', heroAccent: 'Now with subtitles.',
        heroDesc: 'From speech recognition to translation, give your favorite videos the subtitles they deserve. Spend your time watching. Leave the rest to JellySub.',
        heroCTA: 'Open your workspace', heroLearn: 'See how it works', heroNote: 'Local models or online services. Your choice.',
        demoAria: 'Illustration of bilingual subtitles', demoTitle: 'One journey. Two languages.', demoLabel: 'Subtitle illustration', demoCaption: '旅途的意义，是发现新的自己。', demoFooter: 'Stories that travel across languages.', bilingual: 'Bilingual subtitles',
        capabilities: 'Capabilities', capASR: 'Speech recognition', capTranslate: 'Translation', capOffline: 'Offline processing', capBilingual: 'Bilingual SRT',
        featuresEyebrow: 'Less managing. More watching.', featuresTitle: 'Subtitles, without the extra steps.',
        feat1Title: 'New in your library? Ready to go.', feat1Desc: 'Trigger processing with a Jellyfin webhook, or select videos in your workspace to generate subtitles on demand.', feat1Note: 'Automatic triggers · Batch processing',
        feat2Title: 'Hear it. Read it. Understand it.', feat2Desc: 'Start with audio or existing subtitles, translate into your chosen language, and save single or bilingual subtitles.', feat2Note: 'Speech recognition · Translation',
        feat3Title: 'Your setup. Your way.', feat3Desc: 'Run local models offline or connect a compatible online API. Change your models and languages from the workspace.', feat3Note: 'Local models · Online APIs',
        workflowEyebrow: 'From video to subtitles', workflowTitle: 'Set it up. Let the story continue.',
        step1Title: 'Add your media', step1Desc: 'Connect Jellyfin or choose a local video directory.', step2Title: 'Choose models and languages', step2Desc: 'Set up speech recognition, translation, and your subtitle language.', step3Title: 'Let your workspace handle it', step3Desc: 'Track progress, preview subtitles, and get back to watching.',
        closingEyebrow: 'Your next great movie is waiting', closingTitle: 'Subtitles that keep up with your watchlist.', footerTag: 'A little more understanding. For every story.',
    },
};
function setHomeLanguage(language, persist = true) {
    const code = Object.hasOwn(homeTranslations, language) ? language : 'zh';
    const copy = homeTranslations[code];
    document.documentElement.lang = code === 'zh' ? 'zh-CN' : 'en';
    document.title = copy.title;
    document.querySelector('meta[name="description"]').content = copy.heroDesc;
    document.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = copy[el.dataset.i18n]; });
    document.querySelectorAll('[data-i18n-aria]').forEach(el => { el.setAttribute('aria-label', copy[el.dataset.i18nAria]); });
    const button = document.getElementById('lang-toggle');
    button.textContent = code === 'zh' ? 'English' : '中文';
    button.setAttribute('aria-label', code === 'zh' ? 'Switch to English' : '切换为中文');
    button.dataset.language = code;
    if (persist) { try { localStorage.setItem('lang', code); } catch (_) {} }
}
let initialHomeLanguage = 'zh';
try { initialHomeLanguage = localStorage.getItem('lang') || 'zh'; } catch (_) {}
setHomeLanguage(initialHomeLanguage, false);
document.getElementById('lang-toggle').addEventListener('click', () => {
    setHomeLanguage(document.documentElement.lang === 'en' ? 'zh' : 'en');
});
