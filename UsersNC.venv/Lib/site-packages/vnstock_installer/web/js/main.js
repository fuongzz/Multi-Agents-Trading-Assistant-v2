// Main JavaScript for Vnstock Installer

let currentLang = 'vi';
let appConfig = {};
let pythonVersions = [];

// Initialize app
async function init() {
    try {
        appConfig = await eel.initialise()();
        console.log('App initialized:', appConfig);

        // Set language hint if provided
        if (appConfig.languageHint) {
            currentLang = appConfig.languageHint;
            updateLanguageUI();
        }

        // Update version display
        document.getElementById('appVersion').textContent = `v${appConfig.version}`;
        const versionSpan = document.querySelector('.footer .copyright');
        if (versionSpan) {
            versionSpan.textContent = `© 2026 Vnstocks.com • v${appConfig.version}`;
        }

        // Load Python versions
        await loadPythonVersions();

        // Update system info
        updateSystemInfo();

        // Setup event listeners
        setupEventListeners();

        addLog('✅ Ứng dụng khởi động thành công');
        updateStatus('Sẵn sàng');
    } catch (error) {
        console.error('Initialization error:', error);
        addLog(`❌ Lỗi khởi động: ${error}`);
        updateStatus('Lỗi khởi động');
    }
}

// Load Python versions
async function loadPythonVersions() {
    try {
        const select = document.getElementById('pythonVersion');
        const refreshBtn = document.getElementById('refreshPython');

        select.innerHTML = '<option value="">⏳ Đang quét...</option>';
        select.disabled = true;
        refreshBtn.disabled = true;
        refreshBtn.classList.add('spinning');

        pythonVersions = await eel.detect_python_versions()();

        refreshBtn.classList.remove('spinning');
        select.disabled = false;
        refreshBtn.disabled = false;

        if (!pythonVersions || pythonVersions.length === 0) {
            select.innerHTML = '<option value="">❌ Không tìm thấy Python >= 3.10</option>';
            addLog('❌ Không tìm thấy Python phiên bản >= 3.10. Vui lòng cài đặt Python từ python.org');
            updateStatus('Thiếu Python');
            return;
        }

        select.innerHTML = '';
        pythonVersions.forEach((py, index) => {
            const option = document.createElement('option');
            option.value = py.executable;
            option.textContent = py.display;
            if (index === 0) option.selected = true;
            select.appendChild(option);
        });

        addLog(`✅ Tìm thấy ${pythonVersions.length} phiên bản Python`);
    } catch (error) {
        console.error('Error loading Python versions:', error);
        const select = document.getElementById('pythonVersion');
        select.innerHTML = '<option value="">❌ Lỗi khi tìm Python</option>';
        select.disabled = false;
        document.getElementById('refreshPython').disabled = false;
        addLog(`❌ Lỗi khi tìm Python: ${error}`);
    }
}

// Setup event listeners
function setupEventListeners() {
    // Open API page button
    document.getElementById('openApiPageBtn').addEventListener('click', () => {
        addLog('🌐 Đang mở trang Vnstocks.com để lấy khóa API...');
        window.open('https://vnstocks.com/account', '_blank');
        updateStatus('Đang chờ nhập API key...');
    });

    // API key toggle visibility
    const toggleBtn = document.getElementById('toggleApiKey');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', toggleApiKeyVisibility);
    }

    // API key input listener - change button color when filled
    const apiKeyInput = document.getElementById('apiKey');
    const loadPackagesBtn = document.getElementById('loadPackagesBtn');
    if (apiKeyInput && loadPackagesBtn) {
        apiKeyInput.addEventListener('input', () => {
            // Nút Tiếp tục luôn có màu xanh lá, không cần class ready
            const hasApiKey = apiKeyInput.value.trim().length > 0;
            loadPackagesBtn.disabled = !hasApiKey;
        });
    }

    // Refresh Python button
    document.getElementById('refreshPython').addEventListener('click', async () => {
        addLog('🔄 Đang quét lại môi trường Python...');
        await loadPythonVersions();
    });

    // Load Packages button
    document.getElementById('loadPackagesBtn').addEventListener('click', async () => {
        const apiKey = document.getElementById('apiKey').value.trim();
        const pythonExe = document.getElementById('pythonVersion').value;

        if (!apiKey) {
            addLog('❌ Vui lòng nhập khóa API');
            updateStatus('Thiếu API key');
            return;
        }

        if (!pythonExe) {
            addLog('❌ Vui lòng chọn phiên bản Python');
            updateStatus('Thiếu Python');
            return;
        }

        addLog('🔄 Đang đăng ký thiết bị...');
        updateStatus('Đang đăng ký...');

        try {
            const result = await eel.register_device(apiKey, pythonExe)();

            if (result.success) {
                addLog(`✅ Đăng ký thành công! Tier: ${result.tier}`);
                addLog(`📊 Thiết bị: ${result.devicesUsed}/${result.deviceLimit}`);

                // Now load packages
                addLog('📦 Đang tải danh sách packages...');
                const pkgResult = await eel.list_packages()();

                if (pkgResult.success && pkgResult.packages.length > 0) {
                    displayPackages(pkgResult.packages);
                    addLog(`✅ Tìm thấy ${pkgResult.packages.length} packages`);

                    // Show and enable install button
                    const installBtn = document.getElementById('installBtn');
                    installBtn.style.display = 'inline-flex';
                    installBtn.disabled = false;
                    document.getElementById('installInfo').style.display = 'block';
                    updateStatus('Sẵn sàng cài đặt');
                } else {
                    addLog('❌ Không tìm thấy packages');
                    updateStatus('Không có packages');
                }
            } else {
                addLog(`❌ Đăng ký thất bại: ${result.message}`);
                updateStatus('Đăng ký thất bại');
            }
        } catch (error) {
            addLog(`❌ Lỗi: ${error}`);
            updateStatus('Lỗi');
        }
    });

    // Venv checkbox
    document.getElementById('useVenv').addEventListener('change', (e) => {
        const pathGroup = document.getElementById('venvPathGroup');
        pathGroup.style.display = e.target.checked ? 'flex' : 'none';
    });

    // Settings buttons
    document.getElementById('openConfigBtn').addEventListener('click', async () => {
        addLog('📂 Mở thư mục cấu hình...');
        const result = await eel.open_config_folder()();
        if (result) {
            updateStatus('Đã mở thư mục cấu hình');
        }
    });

    // Log buttons
    document.getElementById('clearLogBtn').addEventListener('click', () => {
        document.getElementById('logContent').textContent = 'Log đã được xóa.\n';
        updateStatus('Đã xóa log');
    });

    document.getElementById('copyLogBtn').addEventListener('click', () => {
        const logContent = document.getElementById('logContent').textContent;
        navigator.clipboard.writeText(logContent).then(() => {
            updateStatus('Đã sao chép log');
            addLog('📋 Đã sao chép log vào clipboard');
        }).catch(err => {
            addLog(`❌ Lỗi khi sao chép: ${err}`);
        });
    });
}

// Toggle API key visibility
function toggleApiKeyVisibility() {
    const input = document.getElementById('apiKey');
    const btn = document.getElementById('toggleApiKey');
    const svg = btn.querySelector('svg');

    if (input.type === 'password') {
        input.type = 'text';
        // Change SVG to "eye slash"
        svg.innerHTML = '<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/>';
    } else {
        input.type = 'password';
        // Change SVG back to "eye"
        svg.innerHTML = '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>';
    }
}

// Toggle language
function toggleLanguage() {
    currentLang = currentLang === 'vi' ? 'en' : 'vi';
    updateLanguageUI();
    addLog(`🌐 Đã chuyển ngôn ngữ: ${currentLang === 'vi' ? 'Tiếng Việt' : 'English'}`);
}

function updateLanguageUI() {
    const icon = document.getElementById('langIcon');
    icon.textContent = currentLang === 'vi' ? 'VN' : 'EN';
    // TODO: Add full translation support
}

// Update system info
function updateSystemInfo() {
    const osInfo = document.getElementById('osInfo');
    const pythonInfo = document.getElementById('pythonInfo');
    const configDir = document.getElementById('configDir');

    // Get OS info
    const platform = navigator.platform;
    const userAgent = navigator.userAgent;
    let osDisplay = 'Unknown';

    if (platform.includes('Win')) {
        osDisplay = 'Windows';
        if (userAgent.includes('Windows NT 10')) osDisplay += ' 10/11';
    } else if (platform.includes('Mac')) {
        osDisplay = 'macOS';
    } else if (platform.includes('Linux')) {
        osDisplay = 'Linux';
    }

    osInfo.textContent = osDisplay;

    // Get Python info
    if (pythonVersions.length > 0) {
        pythonInfo.textContent = `Python ${pythonVersions[0].version}`;
    } else {
        pythonInfo.textContent = 'Chưa phát hiện';
    }

    // Config directory
    configDir.textContent = appConfig.defaultVenvPath || '~/.vnstock';
}

// Display packages in UI
function displayPackages(packages) {
    const packageList = document.getElementById('packageList');
    packageList.innerHTML = '';

    packages.forEach(pkg => {
        const item = document.createElement('div');
        item.className = 'package-item';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `pkg-${pkg.name}`;
        checkbox.value = pkg.download_url;
        checkbox.checked = true;  // Select all by default

        const label = document.createElement('label');
        label.htmlFor = `pkg-${pkg.name}`;

        const nameSpan = document.createElement('span');
        nameSpan.className = 'package-name';
        nameSpan.textContent = pkg.name;

        const versionSpan = document.createElement('span');
        versionSpan.className = 'package-version';
        versionSpan.textContent = pkg.version || 'latest';

        label.appendChild(nameSpan);
        label.appendChild(versionSpan);

        item.appendChild(checkbox);
        item.appendChild(label);
        packageList.appendChild(item);
    });
}

// Add log entry
function addLog(message) {
    const logContent = document.getElementById('logContent');
    const timestamp = new Date().toLocaleTimeString('vi-VN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    const entry = `[${timestamp}] ${message}\n`;

    if (logContent.textContent === 'Chờ thao tác...') {
        logContent.textContent = '';
    }

    logContent.textContent += entry;
    logContent.scrollTop = logContent.scrollHeight;
}

// Update status
function updateStatus(message) {
    const statusText = document.getElementById('statusText');
    // Keep the indicator, just update text
    const indicator = statusText.querySelector('.status-indicator');
    statusText.innerHTML = '';
    if (indicator) {
        statusText.appendChild(indicator.cloneNode(true));
    }
    statusText.appendChild(document.createTextNode(message));
}

// Exposed function for progress updates
eel.expose(updateProgress);
function updateProgress(progress, message) {
    const progressBar = document.getElementById('progressBar');
    const progressPercent = document.getElementById('progressPercent');
    const progressText = document.getElementById('progressText');
    const progressSection = document.getElementById('progressSection');

    progressSection.style.display = 'block';
    const percent = Math.round(progress * 100);
    progressBar.style.width = `${percent}%`;
    progressPercent.textContent = `${percent}%`;
    progressText.textContent = message;

    updateStatus(message);
    addLog(message);
}

// Initialize on load
window.addEventListener('DOMContentLoaded', async () => {
    await init();
    // Load saved API key after initialization
    await loadSavedApiKey();
});
