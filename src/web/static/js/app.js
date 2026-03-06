/**
 * ML Dashboard 頁面互動邏輯。
 *
 * 負責股票搜尋、日線載入、指標切換、ML 預測觸發與結果渲染。
 */

(function () {
    'use strict';

    /* ---------- 狀態 ---------- */
    var state = {
        selectedCode: '',
        selectedName: '',
        dailyData: null,
        indicatorSeries: null,
        searchTimer: null,
    };

    /* ---------- DOM 元素 ---------- */
    var els = {};

    function initElements() {
        els.searchInput = document.getElementById('stock-search');
        els.searchDropdown = document.getElementById('search-dropdown');
        els.dateStart = document.getElementById('date-start');
        els.dateEnd = document.getElementById('date-end');
        els.btnLoad = document.getElementById('btn-load');
        els.loadStatus = document.getElementById('stock-load-status');
        els.chartSection = document.getElementById('chart-section');
        els.chartMeta = document.getElementById('chart-meta');
        els.predictSection = document.getElementById('predict-section');
        els.btnPredict = document.getElementById('btn-predict');
        els.predictStatus = document.getElementById('predict-status');
        els.resultSection = document.getElementById('result-section');
        els.resultMeta = document.getElementById('result-meta');
        els.metricsGrid = document.getElementById('metrics-grid');
        els.featureImportance = document.getElementById('feature-importance');
        els.indicatorToggles = document.getElementById('indicator-toggles');
    }

    /* ---------- 預設日期 ---------- */
    function setDefaultDates() {
        var today = new Date();
        var oneYearAgo = new Date();
        oneYearAgo.setFullYear(today.getFullYear() - 1);

        els.dateEnd.value = formatDate(today);
        els.dateStart.value = formatDate(oneYearAgo);
    }

    function formatDate(d) {
        var y = d.getFullYear();
        var m = ('0' + (d.getMonth() + 1)).slice(-2);
        var day = ('0' + d.getDate()).slice(-2);
        return y + '-' + m + '-' + day;
    }

    /* ---------- 股票搜尋 ---------- */
    function handleSearchInput() {
        var keyword = els.searchInput.value.trim();

        if (state.searchTimer) {
            clearTimeout(state.searchTimer);
        }

        if (!keyword) {
            hideDropdown();
            return;
        }

        state.searchTimer = setTimeout(function () {
            fetch('/api/stocks/search?q=' + encodeURIComponent(keyword))
                .then(function (res) { return res.json(); })
                .then(function (results) {
                    renderDropdown(results);
                })
                .catch(function () {
                    renderDropdown([]);
                });
        }, 300);
    }

    function renderDropdown(results) {
        if (!results || results.length === 0) {
            els.searchDropdown.innerHTML =
                '<div class="dropdown-empty">查無結果</div>';
            els.searchDropdown.classList.add('show');
            return;
        }

        var html = '';
        for (var i = 0; i < results.length; i++) {
            var r = results[i];
            html += '<div class="dropdown-item" data-code="' + r.code +
                '" data-name="' + r.name + '">' +
                '<span class="dropdown-code">' + r.code + '</span>' +
                '<span class="dropdown-name">' + r.name + '</span>' +
                '</div>';
        }

        els.searchDropdown.innerHTML = html;
        els.searchDropdown.classList.add('show');

        // 綁定點擊
        var items = els.searchDropdown.querySelectorAll('.dropdown-item');
        for (var i = 0; i < items.length; i++) {
            items[i].addEventListener('click', handleSelectStock);
        }
    }

    function handleSelectStock(e) {
        var item = e.currentTarget;
        var code = item.getAttribute('data-code');
        var name = item.getAttribute('data-name');

        state.selectedCode = code;
        state.selectedName = name;

        els.searchInput.value = code + ' ' + name;
        els.btnLoad.disabled = false;
        hideDropdown();
    }

    function hideDropdown() {
        els.searchDropdown.classList.remove('show');
    }

    /* ---------- 載入日線 ---------- */
    function handleLoad() {
        if (!state.selectedCode) return;

        var start = els.dateStart.value;
        var end = els.dateEnd.value;

        setStatus(els.loadStatus, 'info', '載入中...');
        els.btnLoad.disabled = true;

        var url = '/api/stocks/' + state.selectedCode + '/daily';
        var params = [];
        if (start) params.push('start=' + start);
        if (end) params.push('end=' + end);
        if (params.length > 0) url += '?' + params.join('&');

        // 同時載入日線和技術指標
        var indicatorUrl = '/api/stocks/' + state.selectedCode + '/indicators';
        if (params.length > 0) indicatorUrl += '?' + params.join('&');

        Promise.all([
            fetch(url).then(function (res) { return res.json(); }),
            fetch(indicatorUrl).then(function (res) { return res.json(); }),
        ])
            .then(function (results) {
                var dailyData = results[0];
                var indicators = results[1];

                if (dailyData.error) {
                    setStatus(els.loadStatus, 'error', dailyData.error);
                    els.btnLoad.disabled = false;
                    return;
                }

                if (!dailyData || dailyData.length === 0) {
                    setStatus(els.loadStatus, 'error', '查無資料');
                    els.btnLoad.disabled = false;
                    return;
                }

                state.dailyData = dailyData;
                state.indicatorSeries = indicators.error ? null : indicators;

                setStatus(
                    els.loadStatus, 'success',
                    state.selectedCode + ' ' + state.selectedName +
                    ' - 共 ' + dailyData.length + ' 筆資料'
                );
                els.btnLoad.disabled = false;

                // 顯示圖表
                renderChart();
                els.chartSection.style.display = '';
                els.chartSection.classList.add('fade-in');
                els.chartMeta.textContent =
                    state.selectedCode + ' ' + state.selectedName;

                // 顯示預測區塊
                els.predictSection.style.display = '';

                // 隱藏先前的預測結果
                els.resultSection.style.display = 'none';
            })
            .catch(function (err) {
                setStatus(els.loadStatus, 'error', '載入失敗：' + err.message);
                els.btnLoad.disabled = false;
            });
    }

    /* ---------- 圖表渲染 ---------- */
    function getToggles() {
        var toggles = { sma: false, ema: false, bb: false, rsi: false, macd: false };
        var checkboxes = els.indicatorToggles.querySelectorAll('input[type="checkbox"]');
        for (var i = 0; i < checkboxes.length; i++) {
            toggles[checkboxes[i].value] = checkboxes[i].checked;
        }
        return toggles;
    }

    function renderChart() {
        if (!state.dailyData) return;
        var toggles = getToggles();
        window.StockChart.render(state.dailyData, state.indicatorSeries, toggles);
    }

    /* ---------- ML 預測 ---------- */
    function handlePredict() {
        if (!state.selectedCode) return;

        setStatus(els.predictStatus, 'info',
            '<span class="spinner"></span>模型訓練中，請稍候...');
        els.btnPredict.classList.add('btn-calculating');

        fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ stock_code: state.selectedCode }),
        })
            .then(function (res) { return res.json(); })
            .then(function (result) {
                els.btnPredict.classList.remove('btn-calculating');

                if (result.error) {
                    setStatus(els.predictStatus, 'error', result.error);
                    return;
                }

                setStatus(els.predictStatus, 'success',
                    '預測完成 - 訓練 ' + result.train_samples +
                    ' 筆，測試 ' + result.test_samples +
                    ' 筆，' + result.n_features + ' 個特徵');

                renderPredictionResults(result);
            })
            .catch(function (err) {
                els.btnPredict.classList.remove('btn-calculating');
                setStatus(els.predictStatus, 'error', '預測失敗：' + err.message);
            });
    }

    /* ---------- 預測結果渲染 ---------- */
    function renderPredictionResults(result) {
        // 顯示結果區塊
        els.resultSection.style.display = '';
        els.resultSection.classList.add('fade-in');
        els.resultMeta.textContent =
            state.selectedCode + ' ' + state.selectedName;

        // 渲染指標卡片
        renderMetricsCards(result.metrics);

        // 渲染預測圖表
        window.StockChart.destroyPrediction();
        var predContainer = document.getElementById('prediction-chart');
        if (predContainer) {
            window.StockChart.renderPrediction(predContainer, result.predictions);
        }

        // 渲染特徵重要度
        renderFeatureImportance(result.feature_importance);
    }

    function renderMetricsCards(metrics) {
        var cards = [
            {
                name: '價格 MAE',
                value: metrics.price_MAE.toFixed(2),
                desc: '平均絕對誤差（元）',
            },
            {
                name: '價格 RMSE',
                value: metrics.price_RMSE.toFixed(2),
                desc: '均方根誤差（元）',
            },
            {
                name: '價格 MAPE',
                value: metrics.price_MAPE.toFixed(2) + '%',
                desc: '平均絕對百分比誤差',
            },
            {
                name: '方向正確率',
                value: metrics.directional_accuracy.toFixed(1) + '%',
                desc: '預測漲跌方向的準確率',
            },
        ];

        var html = '';
        for (var i = 0; i < cards.length; i++) {
            var c = cards[i];
            html += '<div class="indicator-card">' +
                '<div class="indicator-name">' + c.name + '</div>' +
                '<div class="indicator-value">' + c.value + '</div>' +
                '<div class="indicator-desc">' + c.desc + '</div>' +
                '</div>';
        }
        els.metricsGrid.innerHTML = html;
    }

    function renderFeatureImportance(features) {
        if (!features || features.length === 0) {
            els.featureImportance.innerHTML =
                '<div class="dropdown-empty">無特徵重要度資料</div>';
            return;
        }

        var maxImp = features[0].importance;
        var count = Math.min(features.length, 15);

        var html = '';
        for (var i = 0; i < count; i++) {
            var f = features[i];
            var pct = maxImp > 0 ? (f.importance / maxImp * 100) : 0;

            html += '<div class="fi-row">' +
                '<span class="fi-rank">' + (i + 1) + '</span>' +
                '<span class="fi-name">' + f.name + '</span>' +
                '<div class="fi-bar-wrapper">' +
                '<div class="fi-bar" style="width: ' + pct.toFixed(1) + '%"></div>' +
                '</div>' +
                '<span class="fi-value">' + (f.importance * 100).toFixed(2) + '%</span>' +
                '</div>';
        }
        els.featureImportance.innerHTML = html;
    }

    /* ---------- 工具函式 ---------- */
    function setStatus(el, type, message) {
        el.className = 'stock-load-status';
        if (type === 'info') el.classList.add('status-info');
        else if (type === 'success') el.classList.add('status-success');
        else if (type === 'error') el.classList.add('status-error');

        // 若是 predict-status，替換 class
        if (el.id === 'predict-status') {
            el.className = 'predict-status';
            if (type === 'info') el.classList.add('status-info');
            else if (type === 'success') el.classList.add('status-success');
            else if (type === 'error') el.classList.add('status-error');
        }

        el.innerHTML = message;
    }

    /* ---------- 事件綁定與初始化 ---------- */
    function init() {
        initElements();
        setDefaultDates();

        // 搜尋
        els.searchInput.addEventListener('input', handleSearchInput);

        // 點擊其他區域關閉下拉選單
        document.addEventListener('click', function (e) {
            if (!e.target.closest('.search-wrapper')) {
                hideDropdown();
            }
        });

        // 載入按鈕
        els.btnLoad.addEventListener('click', handleLoad);

        // 預測按鈕
        els.btnPredict.addEventListener('click', handlePredict);

        // 指標切換
        els.indicatorToggles.addEventListener('change', function () {
            renderChart();
        });
    }

    // DOM 載入完成後初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
