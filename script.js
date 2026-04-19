// ==================== 全局变量 ====================
let currentImageData = null;      // 存储当前图片的像素数据
let currentClusters = null;       // 存储聚类结果 {centers, counts, labels}
let currentImage = null;           // 存储Image对象
let chartInstance = null;          // ECharts实例

// 示例图片的Base64 (使用内置简单图片避免网络请求)
const sampleImages = {
    landscape: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 300'%3E%3Crect width='400' height='300' fill='%23228B22'/%3E%3Ccircle cx='100' cy='200' r='40' fill='%23FFD700'/%3E%3Crect x='0' y='200' width='400' height='100' fill='%238B4513'/%3E%3Crect x='250' y='150' width='80' height='150' fill='%238B6913'/%3E%3C/svg%3E",
    portrait: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 300'%3E%3Crect width='400' height='300' fill='%23FFDAB9'/%3E%3Ccircle cx='200' cy='120' r='50' fill='%23FFC0A0'/%3E%3Ccircle cx='175' cy='110' r='5' fill='%23333333'/%3E%3Ccircle cx='225' cy='110' r='5' fill='%23333333'/%3E%3Cpath d='M180 140 Q200 160 220 140' stroke='%23333333' fill='none' stroke-width='2'/%3E%3Crect x='150' y='180' width='100' height='120' fill='%234A6E8A'/%3E%3C/svg%3E",
    abstract: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 300'%3E%3Crect width='400' height='300' fill='%231A1A2E'/%3E%3Ccircle cx='100' cy='100' r='60' fill='%23FF3366'/%3E%3Ccircle cx='300' cy='200' r='80' fill='%2333FF66'/%3E%3Crect x='150' y='150' width='100' height='100' fill='%233366FF'/%3E%3C/svg%3E"
};

// ==================== 工具函数 ====================

// RGB 转 LAB (使用简化公式，实际生产可用colorjs.io)
function rgbToLab(r, g, b) {
    // 使用 colorjs.io 库进行精确转换
    if (typeof Color !== 'undefined') {
        const color = new Color('srgb', [r/255, g/255, b/255]);
        const lab = color.to('lab');
        return [lab.coords[0], lab.coords[1], lab.coords[2]];
    }
    // 降级方案：简化转换
    let [R, G, B] = [r/255, g/255, b/255];
    const gammaCorrect = (c) => c > 0.04045 ? Math.pow((c + 0.055) / 1.055, 2.4) : c / 12.92;
    R = gammaCorrect(R);
    G = gammaCorrect(G);
    B = gammaCorrect(B);
    let X = R * 0.4124564 + G * 0.3575761 + B * 0.1804375;
    let Y = R * 0.2126729 + G * 0.7151522 + B * 0.0721750;
    let Z = R * 0.0193339 + G * 0.1191920 + B * 0.9503041;
    const epsilon = 0.008856;
    const kappa = 903.3;
    const xyz2lab = (t) => t > epsilon ? Math.pow(t, 1/3) : (kappa * t + 16) / 116;
    const x = X / 0.95047;
    const y = Y / 1.00000;
    const z = Z / 1.08883;
    const fx = xyz2lab(x);
    const fy = xyz2lab(y);
    const fz = xyz2lab(z);
    const L = 116 * fy - 16;
    const a = 500 * (fx - fy);
    const bVal = 200 * (fy - fz);
    return [L, a, bVal];
}

// LAB 转 RGB
function labToRgb(L, a, bVal) {
    if (typeof Color !== 'undefined') {
        const color = new Color('lab', [L, a, bVal]);
        const rgb = color.to('srgb');
        return [Math.round(rgb.coords[0]*255), Math.round(rgb.coords[1]*255), Math.round(rgb.coords[2]*255)];
    }
    // 降级方案返回灰度
    const gray = Math.min(255, Math.max(0, Math.round(L * 2.55)));
    return [gray, gray, gray];
}

// 将颜色转换为向量（根据选择的颜色空间）
function colorToVector(r, g, b, colorSpace) {
    if (colorSpace === 'rgb') {
        return [r, g, b];
    } else {
        return rgbToLab(r, g, b);
    }
}

// 将向量转换回RGB颜色
function vectorToRgb(vec, colorSpace) {
    if (colorSpace === 'rgb') {
        return [Math.round(vec[0]), Math.round(vec[1]), Math.round(vec[2])];
    } else {
        return labToRgb(vec[0], vec[1], vec[2]);
    }
}

// 计算两个向量之间的欧氏距离
function euclideanDistance(v1, v2) {
    let sum = 0;
    for (let i = 0; i < v1.length; i++) {
        sum += (v1[i] - v2[i]) ** 2;
    }
    return Math.sqrt(sum);
}

// ==================== K-Means 聚类算法 ====================
function kMeansClustering(pixels, k, colorSpace, maxIterations = 50) {
    if (pixels.length === 0) return null;
    
    // 将像素转换为向量
    const vectors = pixels.map(p => colorToVector(p.r, p.g, p.b, colorSpace));
    
    // 初始化质心：随机选择k个不同的像素
    const centroids = [];
    const selectedIndices = new Set();
    while (centroids.length < k && selectedIndices.size < vectors.length) {
        const idx = Math.floor(Math.random() * vectors.length);
        if (!selectedIndices.has(idx)) {
            selectedIndices.add(idx);
            centroids.push([...vectors[idx]]);
        }
    }
    
    let labels = new Array(vectors.length).fill(0);
    let changed = true;
    let iteration = 0;
    
    while (changed && iteration < maxIterations) {
        changed = false;
        
        // 分配每个点到最近的质心
        for (let i = 0; i < vectors.length; i++) {
            let minDist = Infinity;
            let bestCluster = 0;
            for (let j = 0; j < centroids.length; j++) {
                const dist = euclideanDistance(vectors[i], centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }
            if (labels[i] !== bestCluster) {
                changed = true;
                labels[i] = bestCluster;
            }
        }
        
        // 更新质心
        const newCentroids = Array(k).fill().map(() => Array(vectors[0].length).fill(0));
        const counts = new Array(k).fill(0);
        
        for (let i = 0; i < vectors.length; i++) {
            const cluster = labels[i];
            counts[cluster]++;
            for (let d = 0; d < vectors[i].length; d++) {
                newCentroids[cluster][d] += vectors[i][d];
            }
        }
        
        for (let j = 0; j < k; j++) {
            if (counts[j] > 0) {
                for (let d = 0; d < newCentroids[j].length; d++) {
                    newCentroids[j][d] /= counts[j];
                }
                centroids[j] = newCentroids[j];
            }
        }
        
        iteration++;
    }
    
    // 计算每个聚类的像素数量
    const counts = new Array(k).fill(0);
    for (let i = 0; i < labels.length; i++) {
        counts[labels[i]]++;
    }
    
    // 将质心转换为RGB颜色用于显示
    const centersRgb = centroids.map(c => vectorToRgb(c, colorSpace));
    
    return {
        centers: centersRgb,
        counts: counts,
        labels: labels,
        centroids: centroids
    };
}

// ==================== 图片处理 ====================

// 从图片中提取像素数据
function extractPixelsFromImage(img, maxPixels = 10000) {
    return new Promise((resolve, reject) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        let width = img.width;
        let height = img.height;
        let sampleRate = 1;
        
        // 如果像素太多，进行采样
        let totalPixels = width * height;
        if (totalPixels > maxPixels) {
            sampleRate = Math.sqrt(totalPixels / maxPixels);
            width = Math.floor(width / sampleRate);
            height = Math.floor(height / sampleRate);
        }
        
        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(img, 0, 0, width, height);
        
        const imageData = ctx.getImageData(0, 0, width, height);
        const data = imageData.data;
        
        const pixels = [];
        for (let i = 0; i < data.length; i += 4) {
            pixels.push({
                r: data[i],
                g: data[i+1],
                b: data[i+2]
            });
        }
        
        resolve(pixels);
    });
}

// 加载图片文件
function loadImageFromFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = e.target.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// ==================== 可视化 ====================

// 显示聚类颜色
function displayClusterColors(centers, counts) {
    const container = document.getElementById('clusterColors');
    const statsContainer = document.getElementById('clusterStats');
    
    if (!centers || centers.length === 0) {
        container.innerHTML = '<div class="placeholder">暂无数据</div>';
        statsContainer.innerHTML = '';
        return;
    }
    
    const total = counts.reduce((a, b) => a + b, 0);
    
    let colorsHtml = '';
    let statsHtml = '<div class="cluster-stats-title">📊 各类占比</div><div class="stats-bars">';
    
    for (let i = 0; i < centers.length; i++) {
        const [r, g, b] = centers[i];
        const hex = `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
        const percentage = ((counts[i] / total) * 100).toFixed(1);
        
        colorsHtml += `
            <div class="color-item">
                <div class="color-circle" style="background-color: ${hex};"></div>
                <div class="color-label">${hex}</div>
                <div class="color-label">${percentage}%</div>
            </div>
        `;
        
        statsHtml += `
            <div class="stat-bar-item">
                <span class="stat-color" style="background: ${hex};"></span>
                <span class="stat-label">类 ${i+1}</span>
                <span class="stat-value">${counts[i]} 像素 (${percentage}%)</span>
            </div>
        `;
    }
    statsHtml += '</div>';
    
    container.innerHTML = colorsHtml;
    statsContainer.innerHTML = statsHtml;
}

// 绘制ECharts图表
function drawChart(centers, counts, chartType) {
    const chartDom = document.getElementById('mainChart');
    if (!chartDom) return;
    
    if (chartInstance) {
        chartInstance.dispose();
    }
    
    chartInstance = echarts.init(chartDom);
    
    const labels = centers.map((_, i) => `聚类 ${i+1}`);
    const colors = centers.map(c => {
        const [r, g, b] = c;
        return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
    });
    
    let option = {};
    
    if (chartType === 'bar') {
        option = {
            tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
            xAxis: { type: 'category', data: labels, name: '聚类类别' },
            yAxis: { type: 'value', name: '像素数量' },
            series: [{
                name: '像素数量',
                type: 'bar',
                data: counts,
                itemStyle: { color: (params) => colors[params.dataIndex], borderRadius: [4,4,0,0] },
                label: { show: true, position: 'top', formatter: '{c}' }
            }],
            title: { text: '各聚类像素数量分布', left: 'center', top: 0, textStyle: { color: '#fff' } },
            backgroundColor: 'transparent',
            grid: { containLabel: true, top: 50, bottom: 20 }
        };
    } else {
        option = {
            tooltip: { trigger: 'item', formatter: '{b}: {d}% ({c} 像素)' },
            series: [{
                name: '颜色聚类',
                type: 'pie',
                radius: '55%',
                center: ['50%', '50%'],
                data: labels.map((label, i) => ({
                    name: label,
                    value: counts[i],
                    itemStyle: { color: colors[i] }
                })),
                label: { show: true, formatter: '{b}\n{d}%', color: '#fff' },
                emphasis: { scale: true }
            }],
            title: { text: '各聚类像素占比', left: 'center', top: 0, textStyle: { color: '#fff' } },
            backgroundColor: 'transparent'
        };
    }
    
    chartInstance.setOption(option);
    window.addEventListener('resize', () => chartInstance?.resize());
}

// ==================== AI 颜色和谐度分析 ====================

async function analyzeColorHarmony(colors) {
    const harmonyTextElem = document.getElementById('harmonyText');
    const harmonyDetailElem = document.getElementById('harmonyDetail');
    const statusIcon = document.querySelector('#harmonyStatus .status-icon');
    
    if (!colors || colors.length < 2) {
        harmonyTextElem.innerHTML = '⚠️ 至少需要2种颜色才能进行和谐度分析';
        statusIcon.textContent = '⚠️';
        return;
    }
    
    harmonyTextElem.innerHTML = '<span class="loading"></span> AI 分析中...';
    statusIcon.textContent = '⏳';
    harmonyDetailElem.style.display = 'none';
    
    // 将颜色转换为十六进制
    const hexColors = colors.map(c => {
        const [r, g, b] = c;
        return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
    });
    
    // 调用 AI API (使用免费的 CloseAI 或模拟接口)
    // 注意：需要替换为你的 API Key，这里提供一个模拟版本和真实API调用示例
    
    const useRealAPI = false; // 设置为 true 并填入 API Key 以使用真实 API
    
    if (useRealAPI) {
        try {
            const response = await fetch('https://api.closeai-asia.com/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer YOUR_API_KEY_HERE'  // 替换为你的API Key
                },
                body: JSON.stringify({
                    model: 'gpt-3.5-turbo',
                    messages: [
                        {
                            role: 'system',
                            content: '你是一位色彩设计专家。请分析给定的颜色组合是否和谐，并给出评分和简短建议。'
                        },
                        {
                            role: 'user',
                            content: `请分析以下颜色组合的色彩和谐度：${hexColors.join(', ')}。请以JSON格式返回：{"isHarmonious": true/false, "score": 0-100, "reason": "分析理由", "suggestion": "改进建议"}`
                        }
                    ],
                    temperature: 0.7
                })
            });
            
            const data = await response.json();
            const result = JSON.parse(data.choices[0].message.content);
            
            displayHarmonyResult(result.isHarmonious, result.score, result.reason, result.suggestion);
            return;
        } catch (error) {
            console.error('AI API调用失败:', error);
        }
    }
    
    // 模拟 AI 分析（基于色彩理论的简单规则）
    const mockAnalysis = simulateColorHarmony(hexColors);
    displayHarmonyResult(mockAnalysis.isHarmonious, mockAnalysis.score, mockAnalysis.reason, mockAnalysis.suggestion);
}

function simulateColorHarmony(hexColors) {
    // 简单的色彩和谐度模拟算法
    // 将颜色转换为HSL并分析色相关系
    
    function hexToHsl(hex) {
        let r = parseInt(hex.slice(1,3), 16) / 255;
        let g = parseInt(hex.slice(3,5), 16) / 255;
        let b = parseInt(hex.slice(5,7), 16) / 255;
        
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        let h, s, l = (max + min) / 2;
        
        if (max === min) {
            h = s = 0;
        } else {
            const d = max - min;
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
            switch (max) {
                case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                case g: h = (b - r) / d + 2; break;
                case b: h = (r - g) / d + 4; break;
            }
            h /= 6;
        }
        return { h: h * 360, s: s * 100, l: l * 100 };
    }
    
    const hslColors = hexColors.map(hex => hexToHsl(hex));
    
    // 检查色相关系
    let harmonyScore = 70; // 基础分
    let reasons = [];
    
    if (hslColors.length === 2) {
        const hueDiff = Math.abs(hslColors[0].h - hslColors[1].h);
        if (hueDiff > 30 && hueDiff < 60) {
            harmonyScore += 15;
            reasons.push('✓ 类似色搭配，和谐统一');
        } else if (hueDiff > 150 && hueDiff < 210) {
            harmonyScore += 20;
            reasons.push('✓ 互补色搭配，对比鲜明且和谐');
        } else if (hueDiff < 15) {
            harmonyScore -= 10;
            reasons.push('⚠️ 颜色过于相似，缺乏层次感');
        } else {
            harmonyScore -= 5;
            reasons.push('⚠️ 色相差值不够理想');
        }
    } else if (hslColors.length >= 3) {
        // 检查是否形成三角形或类似搭配
        const hues = hslColors.map(c => c.h).sort((a,b) => a-b);
        let maxGap = 0;
        for (let i = 0; i < hues.length - 1; i++) {
            maxGap = Math.max(maxGap, hues[i+1] - hues[i]);
        }
        const lastGap = 360 - (hues[hues.length-1] - hues[0]);
        maxGap = Math.max(maxGap, lastGap);
        
        if (maxGap > 120 && maxGap < 180) {
            harmonyScore += 15;
            reasons.push('✓ 色彩分布均匀，可能形成三角形配色');
        } else if (maxGap < 60) {
            harmonyScore -= 15;
            reasons.push('⚠️ 颜色过于集中，缺乏变化');
        } else {
            harmonyScore += 5;
            reasons.push('✓ 色彩有一定变化');
        }
    }
    
    // 检查饱和度
    const avgSat = hslColors.reduce((sum, c) => sum + c.s, 0) / hslColors.length;
    if (avgSat > 70) {
        harmonyScore -= 10;
        reasons.push('⚠️ 饱和度整体偏高，可能过于刺眼');
    } else if (avgSat < 20) {
        harmonyScore -= 5;
        reasons.push('⚠️ 饱和度偏低，可能显得沉闷');
    } else {
        harmonyScore += 5;
        reasons.push('✓ 饱和度适中');
    }
    
    // 检查明度
    const avgLight = hslColors.reduce((sum, c) => sum + c.l, 0) / hslColors.length;
    if (avgLight > 80) {
        reasons.push('⚠️ 整体偏亮，缺乏对比');
    } else if (avgLight < 20) {
        reasons.push('⚠️ 整体偏暗，不够鲜明');
    }
    
    harmonyScore = Math.min(100, Math.max(0, harmonyScore));
    const isHarmonious = harmonyScore >= 60;
    
    let suggestion = '';
    if (isHarmonious) {
        suggestion = '这组颜色搭配不错！可以继续保持这种配色风格。';
    } else {
        suggestion = '建议调整部分颜色的饱和度或明度，或者尝试使用色轮上相距更远/更近的颜色。';
    }
    
    return {
        isHarmonious: isHarmonious,
        score: harmonyScore,
        reason: reasons.join('；') || '颜色组合基本可用',
        suggestion: suggestion
    };
}

function displayHarmonyResult(isHarmonious, score, reason, suggestion) {
    const harmonyTextElem = document.getElementById('harmonyText');
    const harmonyDetailElem = document.getElementById('harmonyDetail');
    const statusIcon = document.querySelector('#harmonyStatus .status-icon');
    
    statusIcon.textContent = isHarmonious ? '✅' : '⚠️';
    harmonyTextElem.innerHTML = `${isHarmonious ? '和谐 ✓' : '不够和谐 ⚠️'} | 和谐度评分: ${score}/100`;
    
    harmonyDetailElem.style.display = 'block';
    harmonyDetailElem.innerHTML = `
        <div><strong>📝 分析理由：</strong> ${reason}</div>
        <div style="margin-top: 8px;"><strong>💡 改进建议：</strong> ${suggestion}</div>
        <div style="margin-top: 8px; font-size: 11px; color: #888;">🤖 AI 分析基于色彩理论模型</div>
    `;
}

// ==================== 主流程 ====================

async function performAnalysis() {
    if (!currentImage) {
        alert('请先选择一张图片！');
        return;
    }
    
    const k = parseInt(document.getElementById('kSlider').value);
    const colorSpace = document.getElementById('colorSpaceSelect').value;
    const chartType = document.getElementById('chartTypeSelect').value;
    
    // 更新图表类型标签
    document.getElementById('chartTypeLabel').innerHTML = chartType === 'bar' ? '(柱状图)' : '(饼图)';
    
    // 显示加载状态
    const analyzeBtn = document.getElementById('analyzeBtn');
    const originalText = analyzeBtn.textContent;
    analyzeBtn.textContent = '⏳ 聚类分析中...';
    analyzeBtn.disabled = true;
    
    try {
        // 提取像素
        const pixels = await extractPixelsFromImage(currentImage);
        
        // 执行K-Means聚类
        const result = kMeansClustering(pixels, k, colorSpace);
        
        if (!result) {
            throw new Error('聚类失败');
        }
        
        currentClusters = result;
        
        // 显示聚类颜色
        displayClusterColors(result.centers, result.counts);
        
        // 绘制图表
        drawChart(result.centers, result.counts, chartType);
        
        // AI 和谐度分析
        await analyzeColorHarmony(result.centers);
        
    } catch (error) {
        console.error('分析出错:', error);
        alert('分析失败：' + error.message);
    } finally {
        analyzeBtn.textContent = originalText;
        analyzeBtn.disabled = false;
    }
}

// 更新K值显示
function updateKDisplay() {
    const kSlider = document.getElementById('kSlider');
    const kDisplay = document.getElementById('kValueDisplay');
    kDisplay.textContent = kSlider.value;
}

// 加载示例图片
function loadSampleImage(type) {
    const img = new Image();
    img.onload = () => {
        currentImage = img;
        const previewImg = document.getElementById('previewImg');
        previewImg.src = img.src;
        previewImg.style.display = 'block';
        document.getElementById('imagePlaceholder').style.display = 'none';
        document.getElementById('imageInfo').innerHTML = `📷 示例图片: ${type} | 尺寸: ${img.width} x ${img.height}`;
        
        // 自动执行分析
        setTimeout(() => performAnalysis(), 100);
    };
    img.src = sampleImages[type];
}

// 处理图片上传
async function handleImageUpload(file) {
    if (!file) return;
    
    try {
        const img = await loadImageFromFile(file);
        currentImage = img;
        const previewImg = document.getElementById('previewImg');
        previewImg.src = URL.createObjectURL(file);
        previewImg.style.display = 'block';
        document.getElementById('imagePlaceholder').style.display = 'none';
        document.getElementById('imageInfo').innerHTML = `📷 用户图片 | 尺寸: ${img.width} x ${img.height} | 大小: ${(file.size / 1024).toFixed(1)} KB`;
        
        // 自动执行分析
        setTimeout(() => performAnalysis(), 100);
    } catch (error) {
        console.error('图片加载失败:', error);
        alert('图片加载失败，请重试');
    }
}

// ==================== 事件绑定 ====================

function bindEvents() {
    // K值滑块
    const kSlider = document.getElementById('kSlider');
    kSlider.addEventListener('input', updateKDisplay);
    
    // 分析按钮
    document.getElementById('analyzeBtn').addEventListener('click', performAnalysis);
    
    // 图片上传
    document.getElementById('imageUpload').addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleImageUpload(e.target.files[0]);
        }
    });
    
    // 示例图片按钮
    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const imgType = btn.dataset.img;
            loadSampleImage(imgType);
        });
    });
    
    // 图表类型切换时重新绘制（如果已有数据）
    document.getElementById('chartTypeSelect').addEventListener('change', () => {
        if (currentClusters) {
            const chartType = document.getElementById('chartTypeSelect').value;
            document.getElementById('chartTypeLabel').innerHTML = chartType === 'bar' ? '(柱状图)' : '(饼图)';
            drawChart(currentClusters.centers, currentClusters.counts, chartType);
        }
    });
}

// ==================== 初始化 ====================

function init() {
    bindEvents();
    updateKDisplay();
    
    // 加载默认示例图片
    loadSampleImage('landscape');
}

// 启动应用
init();