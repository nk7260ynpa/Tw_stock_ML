/**
 * 預測比較圖渲染模組。
 *
 * 使用 TradingView Lightweight Charts v4 繪製預測 vs 實際比較圖。
 */

(function () {
    'use strict';

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
            title: '實際',
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
            title: '預測',
        });

        var predictedData = predictions.map(function (p) {
            return { time: p.date, value: p.predicted };
        });
        predictedSeries.setData(predictedData);

        chart.timeScale().fitContent();
        return chart;
    }

    /** 銷毀預測圖表 */
    function destroyPrediction() {
        for (var i = 0; i < resizeObservers.length; i++) {
            resizeObservers[i].disconnect();
        }
        resizeObservers = [];

        for (var i = 0; i < charts.length; i++) {
            charts[i].remove();
        }
        charts = [];

        var container = document.getElementById('prediction-chart');
        if (container) {
            container.innerHTML = '';
        }
    }

    window.StockChart = {
        renderPrediction: renderPredictionChart,
        destroyPrediction: destroyPrediction,
    };
})();
