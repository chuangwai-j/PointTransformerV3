// 颜色与中文映射
const colorMap = {0:'#ffffff',1:'#00ff00',2:'#ffff00',3:'#ff8800',4:'#ff0000'};
const nameMap  = {0:'无',1:'轻度',2:'中度',3:'强烈',4:'严重'};

// 初始化函数 - 确保 Plotly 已加载
function initializeApp() {
    console.log('检查 Plotly 状态:', typeof Plotly);

    if (typeof Plotly === 'undefined') {
        console.error('Plotly 未定义，等待加载...');
        setTimeout(initializeApp, 100);
        return;
    }

    console.log('Plotly 已加载，初始化应用...');

    // 页面加载完成后绑定事件
    document.addEventListener('DOMContentLoaded', function() {
        console.log('DOM 加载完成，绑定表单...');

        const uploadForm = document.getElementById('uploadForm');
        if (uploadForm) {
            uploadForm.addEventListener('submit', handleFormSubmit);
            console.log('表单绑定成功');

            // 初始显示一个空图形
            initializeEmptyPlot();
        } else {
            console.error('找不到上传表单元素');
        }
    });
}

// 初始化空图形
function initializeEmptyPlot() {
    try {
        const plotDiv = document.getElementById('plot3d');
        if (plotDiv && typeof Plotly !== 'undefined') {
            Plotly.newPlot(plotDiv, [{
                x: [0], y: [0], z: [0],
                mode: 'markers',
                marker: { size: 1, opacity: 0 },
                type: 'scatter3d'
            }], {
                title: '选择CSV文件并上传以显示风切变强度3D分布',
                margin: { l: 0, r: 0, b: 0, t: 50 },
                scene: {
                    xaxis: { title: 'X (m)' },
                    yaxis: { title: 'Y (m)' },
                    zaxis: { title: 'Z (m)' }
                },
                height: 600
            });
            console.log('初始空图形创建成功');
        }
    } catch (error) {
        console.error('创建初始图形失败:', error);
    }
}

// 处理表单提交
async function handleFormSubmit(e) {
    e.preventDefault();
    console.log('表单提交事件触发');

    const form = e.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;

    try {
        // 显示加载状态
        submitBtn.textContent = '处理中...';
        submitBtn.disabled = true;

        const formData = new FormData(form);
        console.log('发送请求到服务器...');

        const res = await fetch('/api/predict/', {
            method: 'POST',
            body: formData
        });

        const json = await res.json();
        console.log('服务器响应:', json);

        if (!res.ok) {
            throw new Error(json.error || json.detail || '预测失败');
        }

        // 显示成功消息
        showMessage(`预测成功! 处理了 ${json.points.length} 个点`, 'success');

        // 绘制3D图形
        draw3D(json.points);

    } catch (error) {
        console.error('请求失败:', error);
        showMessage(`错误: ${error.message}`, 'error');
    } finally {
        // 恢复按钮状态
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    }
}

// 显示消息
function showMessage(message, type) {
    // 创建或获取消息容器
    let messageDiv = document.getElementById('message');
    if (!messageDiv) {
        messageDiv = document.createElement('div');
        messageDiv.id = 'message';
        document.querySelector('body').insertBefore(messageDiv, document.getElementById('plot3d'));
    }

    messageDiv.innerHTML = `<div class="${type}">${message}</div>`;

    // 如果是统计信息，也显示
    if (type === 'success') {
        setTimeout(() => {
            messageDiv.innerHTML += `<div>统计信息: 请查看下方3D可视化</div>`;
        }, 100);
    }
}

// 3D 绘图
function draw3D(points) {
    console.log('开始绘制3D图形，点数:', points.length);

    if (!points || points.length === 0) {
        showMessage('没有数据可显示', 'error');
        return;
    }

    // 再次检查 Plotly 是否可用
    if (typeof Plotly === 'undefined') {
        console.error('Plotly 未定义，无法绘图');
        showMessage('绘图错误: Plotly 未加载，请刷新页面重试', 'error');
        return;
    }

    try {
        const traces = [];
        for (let lbl = 0; lbl < 5; lbl++) {
            const sub = points.filter(p => p.label === lbl);
            console.log(`标签 ${lbl}: ${sub.length} 个点`);

            if (sub.length === 0) continue;

            traces.push({
                x: sub.map(p => p.x),
                y: sub.map(p => p.y),
                z: sub.map(p => p.z),
                mode: 'markers',
                type: 'scatter3d',
                name: `等级 ${lbl} (${nameMap[lbl]})`,
                marker: {
                    size: 3,
                    color: colorMap[lbl],
                    opacity: 0.8
                }
            });
        }

        console.log('创建了', traces.length, '个轨迹');

        const layout = {
            title: `风切变强度 3D 分布 (${points.length} 个点)`,
            margin: { l: 0, r: 0, b: 0, t: 50 },
            scene: {
                xaxis: { title: 'X (m)' },
                yaxis: { title: 'Y (m)' },
                zaxis: { title: 'Z (m)' },
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.5 }
                }
            },
            height: 600
        };

        const plotDiv = document.getElementById('plot3d');
        if (plotDiv) {
            Plotly.newPlot('plot3d', traces, layout, {
                responsive: true,
                displaylogo: false
            });
            console.log('3D图形绘制完成');
        } else {
            console.error('找不到绘图容器 #plot3d');
            showMessage('绘图错误: 找不到绘图容器', 'error');
        }

    } catch (error) {
        console.error('绘图失败:', error);
        showMessage(`绘图错误: ${error.message}`, 'error');
    }
}

// 添加CSS样式
function addStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .success { 
            color: green; 
            font-weight: bold; 
            margin: 10px 0; 
            padding: 10px;
            background: #f0fff0;
            border: 1px solid #d0ffd0;
            border-radius: 4px;
        }
        .error { 
            color: red; 
            font-weight: bold; 
            margin: 10px 0; 
            padding: 10px;
            background: #fff0f0;
            border: 1px solid #ffd0d0;
            border-radius: 4px;
        }
        #plot3d { 
            width: 100%; 
            height: 600px; 
            border: 1px solid #ccc; 
            margin-top: 20px; 
            border-radius: 4px;
        }
        button:disabled { 
            opacity: 0.6; 
            cursor: not-allowed; 
        }
        form {
            margin-bottom: 20px;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 5px;
        }
    `;
    document.head.appendChild(style);
}

// 启动应用
addStyles();
initializeApp();