/**
 * K 線圖渲染模組。
 *
 * 使用 TradingView Lightweight Charts v4 繪製主圖（K 線 + 成交量 + 指標疊加）
 * 與子圖（RSI、MACD）及預測比較圖。
 * 台股慣例：紅漲綠跌。
 */

(function () {
    'use strict';

    /** 指標線顏色對照 */
    var LINE_COLORS = {
        MA5: '#f0b90b',
        MA10: '#e491ff',
        MA20: '#58a6ff',
        EMA12: '#ff7b72',
        EMA26: '#3fb950',
        BB_Upper: '#8b949e',
        BB_Lower: '#8b949e',
        RSI14: '#f0b90b',
        DIF: '#58a6ff',
        MACD: '#f0b90b',
        OSC: '#3fb950',
    };

    /** 深色主題共用設定 */
    var DARK_THEME = {
        layout: {
            background: { color: '#161b22' },
            textColor: '#8b949e',
        },
        grid: {
            vertLines: { color: '#21262d' },
            horzLines: { color: '#21262d' },
        },
        crosshair: {
            mode: 0,
        },
        timeScale: {
            borderColor: '#30363d',
            timeVisible: false,
        },
        rightPriceScale: {
            borderColor: '#30363d',
        },
    };

    /** 圖表實例與 ResizeObserver 暫存 */
    var charts = [];
    var resizeObservers = [];

    /** 建立圖表並掛載 ResizeObserver */
    function createChart(container, opts) {
        var merged = Object.assign({}, DARK_THEME, opts || {}, {
            width: container.clientWidth,
            height: container.clientHeight,
        });
        var chart = LightweightCharts.createChart(container, merged);

        var ro = new ResizeObserver(function (entries) {
            for (var i = 0; i < entries.length; i++) {
                var cr = entries[i].contentRect;
                chart.applyOptions({ width: cr.width });
            }
        });
        ro.observe(container);
        resizeObservers.push(ro);
        charts.push(chart);
        return chart;
    }

    /**
     * 判斷指標 key 的繪製位置。
     *
     * @param {string} key - 指標 key
     * @returns {string} 'main' | 'rsi' | 'macd'
     */
    function classifyIndicator(key) {
        if (/^MA\d+$/.test(key) || /^EMA\d+$/.test(key) ||
            key === 'BB_Upper' || key === 'BB_Lower') {
            return 'main';
        }
        if (/^RSI\d+$/.test(key)) return 'rsi';
        if (key === 'DIF' || key === 'MACD' || key === 'OSC') return 'macd';
        return 'main';
    }

    /**
     * 渲染主圖（K 線 + 成交量 + 指標疊加）。
     *
     * @param {HTMLElement} container - 主圖容器
     * @param {Array} dailyData - 日線資料
     * @param {Object} indicatorSeries - 指標序列
     * @param {Object} toggles - 指標顯示切換設定
     */
    function renderMainChart(container, dailyData, indicatorSeries, toggles) {
        var chart = createChart(container, {
            localization: {
                priceFormatter: function (price) {
                    return price.toFixed(2);
                },
            },
        });

        // K 線（台股：紅漲綠跌）
        var candleSeries = chart.addCandlestickSeries({
            upColor: '#f85149',
            downColor: '#3fb950',
            borderUpColor: '#f85149',
            borderDownColor: '#3fb950',
            wickUpColor: '#f85149',
            wickDownColor: '#3fb950',
        });

        var candleData = dailyData.map(function (d) {
            return {
                time: d.date,
                open: d.open,
                high: d.high,
                low: d.low,
                close: d.close,
            };
        });
        candleSeries.setData(candleData);

        // 成交量（獨立 priceScale，佔底部 20%）
        var volumeSeries = chart.addHistogramSeries({
            priceFormat: { type: 'volume' },
            priceScaleId: 'volume',
        });
        chart.priceScale('volume').applyOptions({
            scaleMargins: { top: 0.8, bottom: 0 },
        });

        var volumeData = dailyData.map(function (d) {
            return {
                time: d.date,
                value: d.volume,
                color: d.close >= d.open
                    ? 'rgba(248, 81, 73, 0.3)'
                    : 'rgba(63, 185, 80, 0.3)',
            };
        });
        volumeSeries.setData(volumeData);

        // 主圖疊加指標線
        if (indicatorSeries) {
            var mainSeries = {};
            var keys = Object.keys(indicatorSeries);
            for (var i = 0; i < keys.length; i++) {
                var key = keys[i];
                if (classifyIndicator(key) === 'main' && _shouldShow(key, toggles)) {
                    mainSeries[key] = indicatorSeries[key];
                }
            }

            var mainKeys = Object.keys(mainSeries);
            for (var i = 0; i < mainKeys.length; i++) {
                var mk = mainKeys[i];
                var isBB = (mk === 'BB_Upper' || mk === 'BB_Lower');
                var lineStyle = isBB
                    ? LightweightCharts.LineStyle.Dashed
                    : LightweightCharts.LineStyle.Solid;

                var lineSeries = chart.addLineSeries({
                    color: LINE_COLORS[mk] || '#8b949e',
                    lineWidth: isBB ? 1 : 2,
                    lineStyle: lineStyle,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false,
                });

                var lineData = [];
                var values = mainSeries[mk];
                for (var j = 0; j < dailyData.length; j++) {
                    if (j < values.length && values[j] !== null && values[j] !== undefined) {
                        lineData.push({
                            time: dailyData[j].date,
                            value: values[j],
                        });
                    }
                }
                lineSeries.setData(lineData);
            }
        }

        chart.timeScale().fitContent();
        return chart;
    }

    /** 建立子圖 DOM 容器 */
    function createSubChartDiv(parentContainer, label) {
        var wrapper = document.createElement('div');
        wrapper.className = 'chart-sub-wrapper';

        var labelEl = document.createElement('div');
        labelEl.className = 'chart-sub-label';
        labelEl.textContent = label;
        wrapper.appendChild(labelEl);

        var chartDiv = document.createElement('div');
        chartDiv.className = 'chart-sub';
        wrapper.appendChild(chartDiv);

        parentContainer.appendChild(wrapper);
        return chartDiv;
    }

    /** 渲染子圖（RSI / MACD） */
    function renderSubChart(container, dailyData, seriesGroup) {
        var chart = createChart(container);
        var keys = Object.keys(seriesGroup);

        for (var i = 0; i < keys.length; i++) {
            var key = keys[i];
            var values = seriesGroup[key];

            // OSC 使用柱狀圖
            if (key === 'OSC') {
                var histSeries = chart.addHistogramSeries({
                    priceLineVisible: false,
                    lastValueVisible: false,
                });
                var histData = [];
                for (var j = 0; j < dailyData.length; j++) {
                    if (j < values.length && values[j] !== null && values[j] !== undefined) {
                        histData.push({
                            time: dailyData[j].date,
                            value: values[j],
                            color: values[j] >= 0
                                ? 'rgba(248, 81, 73, 0.6)'
                                : 'rgba(63, 185, 80, 0.6)',
                        });
                    }
                }
                histSeries.setData(histData);
            } else {
                var lineSeries = chart.addLineSeries({
                    color: LINE_COLORS[key] || '#8b949e',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false,
                });
                var lineData = [];
                for (var j = 0; j < dailyData.length; j++) {
                    if (j < values.length && values[j] !== null && values[j] !== undefined) {
                        lineData.push({
                            time: dailyData[j].date,
                            value: values[j],
                        });
                    }
                }
                lineSeries.setData(lineData);
            }
        }

        chart.timeScale().fitContent();
        return chart;
    }

    /**
     * 渲染預測 vs 實際的比較圖。
     *
     * @param {HTMLElement} container - 圖表容器
     * @param {Array} predictions - 預測結果陣列 [{date, actual, predicted}]
     */
    function renderPredictionChart(container, predictions) {
        if (!predictions || predictions.length === 0) return;

        var chart = createChart(container, {
            localization: {
                priceFormatter: function (price) {
                    return price.toFixed(2);
                },
            },
        });

        // 實際值（藍線）
        var actualSeries = chart.addLineSeries({
            color: '#58a6ff',
            lineWidth: 2,
            priceLineVisible: false,
            lastValueVisible: true,
            title: 'actual',
        });

        var actualData = predictions.map(function (p) {
            return { time: p.date, value: p.actual };
        });
        actualSeries.setData(actualData);

        // 預測值（黃線）
        var predictedSeries = chart.addLineSeries({
            color: '#f0b90b',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            priceLineVisible: false,
            lastValueVisible: true,
            title: 'predicted',
        });

        var predictedData = predictions.map(function (p) {
            return { time: p.date, value: p.predicted };
        });
        predictedSeries.setData(predictedData);

        chart.timeScale().fitContent();
        return chart;
    }

    /** 判斷指標是否應顯示 */
    function _shouldShow(key, toggles) {
        if (!toggles) return true;
        if (/^MA\d+$/.test(key)) return !!toggles.sma;
        if (/^EMA\d+$/.test(key)) return !!toggles.ema;
        if (key === 'BB_Upper' || key === 'BB_Lower') return !!toggles.bb;
        return true;
    }

    /** 銷毀所有圖表實例 */
    function destroy() {
        for (var i = 0; i < resizeObservers.length; i++) {
            resizeObservers[i].disconnect();
        }
        resizeObservers = [];

        for (var i = 0; i < charts.length; i++) {
            charts[i].remove();
        }
        charts = [];

        var subContainer = document.getElementById('sub-charts-container');
        if (subContainer) {
            subContainer.innerHTML = '';
        }
    }

    /** 銷毀預測圖表 */
    function destroyPrediction() {
        var container = document.getElementById('prediction-chart');
        if (container) {
            container.innerHTML = '';
        }
    }

    /**
     * 渲染完整圖表（主圖 + 子圖）。
     *
     * @param {Array} dailyData - 日線資料
     * @param {Object} indicatorSeries - 技術指標序列
     * @param {Object} toggles - 指標顯示切換設定
     */
    function render(dailyData, indicatorSeries, toggles) {
        destroy();

        if (!dailyData || dailyData.length === 0) return;

        // 主圖
        var mainContainer = document.getElementById('main-chart');
        if (mainContainer) {
            renderMainChart(mainContainer, dailyData, indicatorSeries, toggles);
        }

        // 分類子圖指標
        var rsiSeries = {};
        var macdSeries = {};

        if (indicatorSeries) {
            var seriesKeys = Object.keys(indicatorSeries);
            for (var i = 0; i < seriesKeys.length; i++) {
                var key = seriesKeys[i];
                var target = classifyIndicator(key);
                if (target === 'rsi' && toggles && toggles.rsi) {
                    rsiSeries[key] = indicatorSeries[key];
                } else if (target === 'macd' && toggles && toggles.macd) {
                    macdSeries[key] = indicatorSeries[key];
                }
            }
        }

        // 子圖
        var subContainer = document.getElementById('sub-charts-container');
        if (!subContainer) return;

        if (Object.keys(rsiSeries).length > 0) {
            var rsiDiv = createSubChartDiv(subContainer, 'RSI');
            renderSubChart(rsiDiv, dailyData, rsiSeries);
        }
        if (Object.keys(macdSeries).length > 0) {
            var macdDiv = createSubChartDiv(subContainer, 'MACD');
            renderSubChart(macdDiv, dailyData, macdSeries);
        }
    }

    window.StockChart = {
        render: render,
        renderPrediction: renderPredictionChart,
        destroy: destroy,
        destroyPrediction: destroyPrediction,
    };
})();
