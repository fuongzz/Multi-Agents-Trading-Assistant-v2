// Installation logic

let registeredDeviceInfo = null;
let availablePackages = [];

// Load saved API key on page init
async function loadSavedApiKey() {
    try {
        const savedKey = await eel.load_saved_api_key()();
        if (savedKey && savedKey.trim()) {
            const apiKeyInput = document.getElementById('apiKey');
            apiKeyInput.value = savedKey;

            // Kích hoạt event để enable nút Tiếp tục
            const event = new Event('input', { bubbles: true });
            apiKeyInput.dispatchEvent(event);

            addLog('✅ Đã tải API key đã lưu trước đó');
        }
    } catch (error) {
        console.debug('No saved API key or error loading:', error);
    }
}

// Combined register + install flow
async function registerDevice() {
    // Get API key from either OAuth input or direct input
    let apiKey = document.getElementById('oauthApiKey')?.value.trim();
    if (!apiKey || currentAuthMethod === 'apikey') {
        apiKey = document.getElementById('apiKey').value.trim();
    }

    const pythonExe = document.getElementById('pythonVersion').value;

    if (!apiKey) {
        alert('Vui lòng nhập khóa API!');
        addLog('❌ Chưa nhập khóa API');
        return false;
    }

    if (!pythonExe) {
        alert('Vui lòng chọn phiên bản Python!');
        addLog('❌ Chưa chọn Python');
        return false;
    }

    try {
        addLog('📋 Đang đăng ký thiết bị...');
        updateStatus('Đang đăng ký...');
        const result = await eel.register_device(apiKey, pythonExe)();

        if (result.success) {
            registeredDeviceInfo = result;
            addLog(`✅ Đăng ký thiết bị thành công!`);
            addLog(`   Gói: ${result.tier}`);
            addLog(`   Thiết bị: ${result.devicesUsed}/${result.deviceLimit}`);

            // Load packages
            await loadPackages();

            updateStatus('✅ Đã đăng ký thiết bị');
            return true;
        } else {
            addLog(`❌ Đăng ký thất bại: ${result.message}`);
            alert(`Đăng ký thất bại:\n${result.message}`);
            return false;
        }
    } catch (error) {
        console.error('Registration error:', error);
        addLog(`❌ Lỗi: ${error}`);
        alert(`Lỗi: ${error}`);
        return false;
    }
}

// Load packages
async function loadPackages() {
    try {
        addLog('📦 Loading available packages...');
        const result = await eel.list_packages()();

        if (result.success && result.packages.length > 0) {
            availablePackages = result.packages;
            displayPackages(result.packages);

            // Enable install button
            document.getElementById('installBtn').disabled = false;

            addLog(`✅ Found ${result.packages.length} package(s)`);
        } else {
            const msg = result.message || 'No packages available';
            addLog(`⚠️ ${msg}`);
            document.getElementById('packageList').innerHTML =
                `<p class="text-muted">${msg}</p>`;
        }
    } catch (error) {
        console.error('Error loading packages:', error);
        addLog(`❌ Error loading packages: ${error}`);
    }
}

// Display packages
function displayPackages(packages) {
    const container = document.getElementById('packageList');
    container.innerHTML = '';

    packages.forEach(pkg => {
        const item = document.createElement('div');
        item.className = 'package-item';
        item.innerHTML = `
            <input type="checkbox" checked data-package="${pkg.name}">
            <div class="package-info">
                <div class="package-name">${pkg.displayName || pkg.name}</div>
                <div class="package-desc">${pkg.description || 'No description'}</div>
            </div>
        `;
        container.appendChild(item);
    });
}

// Start installation (now includes registration)
document.getElementById('installBtn').addEventListener('click', async () => {
    const btn = document.getElementById('installBtn');
    btn.disabled = true;

    try {
        // Auto-register if not registered yet
        if (!registeredDeviceInfo) {
            btn.textContent = '⏳ Đang đăng ký...';
            const registered = await registerDevice();
            if (!registered) {
                btn.disabled = false;
                btn.innerHTML = `
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="7 10 12 15 17 10" />
                        <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    Bắt Đầu Cài Đặt
                `;
                return;
            }
        }

        btn.textContent = '⏳ Đang cài đặt...';

        const pythonExe = document.getElementById('pythonVersion').value;
        const useVenv = document.getElementById('useVenv').checked;
        const venvPath = document.getElementById('venvPath').value.trim();

        // Get selected packages
        const checkboxes = document.querySelectorAll('#packageList input[type="checkbox"]:checked');
        const selectedPackages = Array.from(checkboxes).map(cb => cb.dataset.package);

        if (selectedPackages.length === 0) {
            alert('Vui lòng chọn ít nhất 1 package!');
            btn.disabled = false;
            btn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                Bắt Đầu Cài Đặt
            `;
            return;
        }

        addLog(`🚀 Starting installation of ${selectedPackages.length} package(s)`);

        const result = await eel.start_installation(
            pythonExe,
            selectedPackages,
            useVenv,
            venvPath || null
        )();

        if (result.success) {
            const summary = result.summary;
            addLog('');
            addLog('='.repeat(60));
            addLog('🎉 INSTALLATION SUMMARY');
            addLog('='.repeat(60));
            addLog(`✅ Successful: ${summary.success_count}`);
            summary.installed.forEach(pkg => {
                addLog(`   • ${pkg}`);
            });

            if (summary.fail_count > 0) {
                addLog(`❌ Failed: ${summary.fail_count}`);
                summary.failed.forEach(([pkg, msg]) => {
                    addLog(`   • ${pkg}: ${msg}`);
                });
            }

            addLog('='.repeat(60));
            updateStatus(`✅ Installation completed: ${summary.success_count}/${summary.total}`);

            // Show detailed success modal
            showSuccessModal(pythonExe, useVenv, venvPath, summary);
        } else {
            addLog(`❌ Installation failed: ${result.message}`);
            alert(`Installation failed:\n${result.message}`);
        }
    } catch (error) {
        console.error('Installation error:', error);
        addLog(`❌ Installation error: ${error}`);
        alert(`Error: ${error}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Bắt Đầu Cài Đặt
        `;
        document.getElementById('progressSection').style.display = 'none';
    }
});

// Show success modal with detailed instructions
async function showSuccessModal(pythonExe, useVenv, venvPath, summary) {
    const isWindows = navigator.platform.toLowerCase().includes('win');

    // Get username from backend
    let username = 'User';
    try {
        username = await eel.get_username()();
    } catch (e) {
        console.warn('Could not get username:', e);
    }

    // Handle empty venvPath
    venvPath = venvPath ? venvPath.trim() : '';
    if (useVenv && !venvPath) {
        venvPath = isWindows ? `C:\\Users\\${username}\\.venv` : '~/.venv';
    }

    let html = `
        <div style="background: #F0FDF4; border: 1px solid #86EFAC; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
            <p style="margin: 0; color: #15803D; font-weight: 600;">
                ✅ Đã cài đặt thành công ${summary.success_count} gói
            </p>
            ${summary.fail_count > 0 ? `<p style="margin: 8px 0 0 0; color: #DC2626; font-weight: 600;">❌ ${summary.fail_count} gói lỗi</p>` : ''}
        </div>
    `;

    if (useVenv && venvPath) {
        // Virtual environment instructions
        // For Windows: use relative path or & operator
        // For Unix: use source command
        let activateCmd, activateNote;

        if (isWindows) {
            // Extract just the folder name if it's a full path
            const venvFolder = venvPath.includes('\\')
                ? venvPath.split('\\').pop()
                : venvPath;
            activateCmd = `.\\${venvFolder}\\Scripts\\activate`;
            activateNote = `Hoặc nếu ở thư mục khác: <code>& "${venvPath}\\Scripts\\Activate.ps1"</code>`;
        } else {
            activateCmd = `source ${venvPath}/bin/activate`;
            activateNote = '';
        }

        html += `
            <h3>📦 Môi trường ảo đã được cài đặt tại:</h3>
            <code>${venvPath}</code>
            
            <h3>🚀 Bước 1: Kích hoạt môi trường ảo</h3>
            <p>Mở terminal tại thư mục chứa môi trường ảo và chạy:</p>
            <code>${activateCmd}</code>
            ${activateNote ? `<p style="margin-top: 8px; font-size: 13px; color: var(--text-muted);">${activateNote}</p>` : ''}
            
            <h3>✅ Bước 2: Kiểm tra cài đặt</h3>
            <p>Sau khi kích hoạt, kiểm tra các gói đã cài:</p>
            <code>python -c "import vnstock_data, vnstock_ta, vnstock_news, vnstock_pipeline; print('✅ All packages imported successfully!')"</code>
            
            <h3>📝 Các gói đã cài:</h3>
            <ul>
                ${summary.installed.map(pkg => `<li><strong>${pkg}</strong></li>`).join('')}
            </ul>
        `;
    } else {
        // System Python instructions
        html += `
            <h3>🐍 Python đã sử dụng:</h3>
            <code>${pythonExe}</code>
            
            <h3>✅ Kiểm tra cài đặt:</h3>
            <p>Chạy các lệnh sau để kiểm tra:</p>
            <code>${pythonExe} -c "import vnstock_data, vnstock_ta, vnstock_news, vnstock_pipeline; print('✅ All packages imported successfully!')"</code>
            
            <h3>📝 Các gói đã cài:</h3>
            <ul>
                ${summary.installed.map(pkg => `<li><strong>${pkg}</strong></li>`).join('')}
            </ul>
        `;
    }

    if (summary.fail_count > 0) {
        html += `
            <h3 style="color: var(--error);">❌ Các gói cài lỗi:</h3>
            <ul style="color: var(--error);">
                ${summary.failed.map(([pkg, msg]) => `<li><strong>${pkg}</strong>: ${msg}</li>`).join('')}
            </ul>
        `;
    }

    // Warning about skipped dependencies
    if (summary.skipped_deps && summary.skipped_deps.length > 0) {
        html += `
            <div style="background: #FEF3C7; border: 1px solid #F59E0B; border-radius: 8px; padding: 16px; margin: 20px 0;">
                <h3 style="color: #92400E; margin-top: 0;">⚠️ Cảnh báo: Một số gói phụ trợ chưa được cài đặt</h3>
                <p style="color: #78350F; margin-bottom: 12px;">
                    Các gói sau cần công cụ biên dịch (Visual Studio Build Tools) và đã bị bỏ qua:
                </p>
                <ul style="color: #78350F;">
                    ${summary.skipped_deps.map(dep => `<li><strong>${dep}</strong></li>`).join('')}
                </ul>
                <p style="color: #78350F; margin-bottom: 8px;">
                    <strong>Tác động:</strong> Một số tính năng nâng cao có thể không hoạt động (chart, xử lý dữ liệu nhanh).
                </p>
                <p style="color: #78350F; margin: 0;">
                    📖 <strong>Hướng dẫn khắc phục:</strong> 
                    <a href="https://vnstocks.com/onboard-member/cai-dat-go-loi/giai-quyet-loi-thuong-gap" 
                       target="_blank" 
                       style="color: #92400E; text-decoration: underline; font-weight: 600;">
                        Xem hướng dẫn cài đặt Build Tools
                    </a>
                </p>
            </div>
        `;
    }

    html += `
        <h3>💡 Ghi chú:</h3>
        <ul>
            <li>Bạn có thể bắt đầu sử dụng vnstock ngay bây giờ</li>
            <li>Xem tài liệu tại: <a href="https://vnstocks.com/docs/vnstock-insider-api/index" target="_blank">vnstocks.com</a></li>
            <li>Hỗ trợ: <a href="https://www.facebook.com/groups/vnstock.insiders" target="_blank">Nhóm Insider</a></li>
        </ul>
    `;

    document.getElementById('successModalBody').innerHTML = html;
    document.getElementById('successModal').style.display = 'flex';
}
