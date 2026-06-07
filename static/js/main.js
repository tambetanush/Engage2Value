const sampleInput = {
    "trafficSource_isTrueDirect": 1.0,
    "browser": "Chrome",
    "device_screenResolution": "1920x1080",
    "trafficSource_adContent": "None",
    "trafficSource_keyword": "organic",
    "screenSize": "large",
    "geoCluster": "cluster_1",
    "trafficSource_adwordsClickInfo_slot": "Top",
    "device_mobileDeviceBranding": "Samsung",
    "device_mobileInputSelector": "Touch",
    "userId": 100001,
    "trafficSource_campaign": "campaign_1",
    "device_mobileDeviceMarketingName": "Galaxy S21",
    "geoNetwork_networkDomain": "com",
    "gclIdPresent": 0,
    "device_operatingSystemVersion": "11.0",
    "sessionNumber": 5,
    "device_flashVersion": "32.0",
    "geoNetwork_region": "Maharashtra",
    "trafficSource": "Google",
    "totals_visits": 2,
    "geoNetwork_networkLocation": "ISP",
    "sessionId": 200001,
    "os": "Android",
    "geoNetwork_subContinent": "Southern Asia",
    "trafficSource_medium": "organic",
    "trafficSource_adwordsClickInfo_isVideoAd": "False",
    "browserMajor": "Chrome 95",
    "locationCountry": "India",
    "device_browserSize": "fullscreen",
    "trafficSource_adwordsClickInfo_adNetworkType": "Search Network",
    "socialEngagementType": "Not Socially Engaged",
    "geoNetwork_city": "Mumbai",
    "trafficSource_adwordsClickInfo_page": 1.0,
    "geoNetwork_metro": "Mumbai",
    "pageViews": 10.0,
    "locationZone": 1,
    "device_mobileDeviceModel": "SM-G991B",
    "trafficSource_referralPath": "/",
    "totals_bounces": 0.0,
    "date": 20251011,
    "device_language": "en-us",
    "deviceType": "mobile",
    "userChannel": "organic",
    "device_browserVersion": "95.0",
    "totalHits": 50,
    "device_screenColors": "24-bit",
    "sessionStart": 10,
    "geoNetwork_continent": "Asia",
    "device_isMobile": true,
    "new_visits": 0
};

document.addEventListener('DOMContentLoaded', () => {
    // Navigation
    const navBtns = document.querySelectorAll('.nav-btn');
    const pages = document.querySelectorAll('.page');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            navBtns.forEach(b => b.classList.remove('active'));
            pages.forEach(p => p.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(btn.dataset.target).classList.add('active');
        });
    });

    // Populate Prediction Form with HerdPulse classes
    const formGrid = document.getElementById('form-fields');
    Object.keys(sampleInput).forEach(key => {
        const div = document.createElement('div');
        div.className = 'flex flex-col gap-1';
        div.innerHTML = `
            <label for="${key}" class="text-[10px] font-bold text-slate-500 uppercase tracking-widest font-mono">${key}</label>
            <input type="text" id="${key}" name="${key}" required class="px-4 py-3 rounded-xl bg-slate-50 text-slate-800 border border-slate-200 focus:outline-none focus:ring-2 focus:ring-rose-500/20 focus:border-rose-500 transition-all font-medium">
        `;
        formGrid.appendChild(div);
    });

    document.getElementById('btn-randomize').addEventListener('click', () => {
        Object.entries(sampleInput).forEach(([key, value]) => {
            let randVal = value;
            if (typeof value === 'number') {
                randVal = (value * (Math.random() * 0.5 + 0.75)).toFixed(2);
                if (Number.isInteger(value)) randVal = Math.round(randVal);
            }
            document.getElementById(key).value = randVal;
        });
    });

    // Handle Prediction Submission
    document.getElementById('predict-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = document.getElementById('btn-submit');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<span>⏳</span> Predicting...';
        
        const data = {};
        Object.keys(sampleInput).forEach(key => {
            let val = document.getElementById(key).value;
            // parse numbers
            if (!isNaN(val) && val.trim() !== '') {
                val = Number(val);
            } else if (val.toLowerCase() === 'true') {
                val = true;
            } else if (val.toLowerCase() === 'false') {
                val = false;
            }
            data[key] = val;
        });

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await res.json();
            
            if (result.error) throw new Error(result.error);
            
            document.getElementById('prediction-result').classList.remove('hidden');
            document.getElementById('predicted-amount').textContent = result.predicted_purchase_value.toFixed(2);
        } catch (err) {
            alert('Prediction failed: ' + err.message);
        } finally {
            btn.innerHTML = originalText;
        }
    });

    // Load Features for Exploration
    fetch('/api/features')
        .then(res => res.json())
        .then(data => {
            const select = document.getElementById('feature-select');
            const secSelect = document.getElementById('secondary-feature-select');
            select.innerHTML = '<option value="">-- Choose a Feature --</option>';
            if(secSelect) secSelect.innerHTML = '<option value="">-- Select for Bivariate Analysis --</option>';
            data.features.forEach(f => {
                const opt = document.createElement('option');
                opt.value = f;
                opt.textContent = f;
                select.appendChild(opt);
                
                if(secSelect) {
                    const optSec = document.createElement('option');
                    optSec.value = f;
                    optSec.textContent = f;
                    secSelect.appendChild(optSec);
                }
            });
            document.getElementById('dash-feature-count').textContent = data.features.length;
        });

    // Global SHAP fetching
    let shapLoaded = false;
    const exploreBtn = document.querySelector('[data-target="exploration"]');
    if (exploreBtn) {
        exploreBtn.addEventListener('click', async () => {
            if (!shapLoaded) {
                try {
                    const res = await fetch('/api/shap-summary');
                    const data = await res.json();
                    if (data.error) throw new Error(data.error);

                    const isDark = document.body.classList.contains('dark-mode');
                    const textColor = isDark ? '#f1f5f9' : '#1e293b';
                    const layoutTheme = {
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: textColor, family: "'Lora', serif" },
                        margin: { t: 20, r: 20, l: 200, b: 40 },
                        autosize: true
                    };

                    const trace = {
                        x: data.importances,
                        y: data.features,
                        type: 'bar',
                        orientation: 'h',
                        marker: { color: '#e11d48' }
                    };

                    Plotly.newPlot('plot-global-shap', [trace], layoutTheme, { responsive: true });
                    shapLoaded = true;
                } catch (e) {
                    document.getElementById('plot-global-shap').innerHTML = `<div class="flex items-center justify-center h-full text-red-500 font-mono text-sm">Error: ${e.message}</div>`;
                }
            }
        });
    }

    // Handle Feature Selection
    document.getElementById('feature-select').addEventListener('change', async (e) => {
        const feature = e.target.value;
        
        if (!feature) {
            document.getElementById('exploration-results').classList.add('hidden');
            document.getElementById('secondary-feature-container').classList.add('hidden');
            document.getElementById('bivariate-container').classList.add('hidden');
            return;
        }

        const resultsDiv = document.getElementById('exploration-results');
        resultsDiv.classList.remove('hidden');
        document.getElementById('secondary-feature-container').classList.remove('hidden');
        document.getElementById('secondary-feature-select').value = '';
        document.getElementById('bivariate-container').classList.add('hidden');
        
        try {
            const res = await fetch(`/api/explore/${feature}`);
            const data = await res.json();

            // Populate Stats
            document.getElementById('stat-type').textContent = data.stats.type;
            document.getElementById('stat-missing').textContent = data.stats.missing_pct + '%';
            document.getElementById('stat-unique').textContent = data.stats.unique_vals;
            
            const isNum = data.stats.type === 'Numeric';
            document.getElementById('corr-box').style.display = isNum ? 'block' : 'none';
            document.getElementById('shap-box').style.display = isNum && data.stats.isolated_shap_importance ? 'block' : 'none';
            
            if (isNum) {
                document.getElementById('stat-corr').textContent = data.stats.correlation;
                if (data.stats.isolated_shap_importance) {
                    document.getElementById('stat-shap').textContent = data.stats.isolated_shap_importance;
                }
            }
            
            // Inference
            document.getElementById('inference-text').textContent = data.inference;

            // Handle Subgroup Discovery
            if (data.stats.subgroup_insight) {
                document.getElementById('subgroup-insight').textContent = data.stats.subgroup_insight;
                document.getElementById('subgroup-discovery').classList.remove('hidden');
            } else {
                document.getElementById('subgroup-discovery').classList.add('hidden');
            }

            // Generate Plots
            const isDark = document.body.classList.contains('dark-mode');
            const textColor = isDark ? '#f1f5f9' : '#1e293b';
            
            const layoutTheme = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: textColor, family: "'Lora', serif" },
                margin: { t: 40, r: 20, l: 40, b: 40 },
                title: { font: { family: "'Playfair Display', serif", size: 16 } },
                autosize: true
            };

            const config = { responsive: true };

            const chartsGrid = document.getElementById('charts-grid-container');
            chartsGrid.innerHTML = ''; // clear

            const addPlot = (id, trace, title) => {
                const div = document.createElement('div');
                div.className = 'bg-white rounded-2xl p-6 border border-slate-200/80 shadow-sm min-h-[400px] flex flex-col min-w-0 overflow-hidden';
                div.id = id;
                chartsGrid.appendChild(div);
                Plotly.newPlot(id, Array.isArray(trace) ? trace : [trace], { ...layoutTheme, title }, config);
            };

            if (isNum) {
                const fData = data.plots.target_scatter.x;
                const tData = data.plots.target_scatter.y;

                addPlot('plot1', data.plots.distribution, 'Feature Distribution');
                
                if (data.plots.target_scatter.is_anomaly) {
                    const isAnom = data.plots.target_scatter.is_anomaly;
                    const normX = [], normY = [], anomX = [], anomY = [];
                    for (let i = 0; i < isAnom.length; i++) {
                        if (isAnom[i]) {
                            anomX.push(fData[i]);
                            anomY.push(tData[i]);
                        } else {
                            normX.push(fData[i]);
                            normY.push(tData[i]);
                        }
                    }
                    addPlot('plot2', [
                        { x: normX, y: normY, mode: 'markers', type: 'scatter', name: 'Normal', marker: { color: '#e11d48', size: 6, opacity: 0.6 } },
                        { x: anomX, y: anomY, mode: 'markers', type: 'scatter', name: 'Anomaly', marker: { color: '#ef4444', size: 8, opacity: 0.9, symbol: 'x' } }
                    ], 'Feature vs Target (Outliers)');
                } else {
                    addPlot('plot2', {
                        x: fData, y: tData, mode: 'markers', type: 'scatter', name: 'Feature vs Target',
                        marker: { color: '#e11d48', size: 6, opacity: 0.6 }
                    }, 'Feature vs Target');
                }

                addPlot('plot3', {
                    x: fData, y: tData, type: 'histogram2dcontour', colorscale: 'Roses'
                }, '2D Density');

                addPlot('plot4', {
                    y: fData, type: 'box', name: feature, marker: { color: '#e11d48' }
                }, 'Boxplot (Outliers)');

                addPlot('plot5', {
                    x: data.plots.distribution.x, y: data.plots.distribution.y,
                    type: 'scatter', fill: 'tozeroy', marker: { color: '#8b5cf6' }
                }, 'Cumulative Distribution');

                addPlot('plot6', {
                    y: fData, type: 'violin', marker: { color: '#10b981' }
                }, 'Violin Density');

                if (data.plots.pdp) {
                    addPlot('plot7', data.plots.pdp, 'Marginal Effect (PDP)');
                }

            } else {
                addPlot('plot1', data.plots.distribution, 'Top Categories');
                addPlot('plot2', data.plots.target_scatter, 'Mean Target per Category');
                
                addPlot('plot3', {
                    values: data.plots.distribution.y,
                    labels: data.plots.distribution.x,
                    type: 'pie', hole: 0.4
                }, 'Category Share');
            }

            resultsDiv.classList.remove('hidden');

        } catch (err) {
            console.error(err);
            alert("Failed to load feature exploration data.");
        }
    });

    // Handle Secondary Feature Selection (Bivariate)
    const secSelect = document.getElementById('secondary-feature-select');
    if (secSelect) {
        secSelect.addEventListener('change', async (e) => {
            const secFeature = e.target.value;
            const primFeature = document.getElementById('feature-select').value;
            const container = document.getElementById('bivariate-container');
            
            if (!secFeature || !primFeature) {
                container.classList.add('hidden');
                return;
            }
            
            container.classList.remove('hidden');
            document.getElementById('plot-bivariate').innerHTML = '<div class="flex items-center justify-center h-full text-slate-400 font-mono text-sm">Loading interaction...</div>';
            
            try {
                const res = await fetch(`/api/bivariate/${primFeature}/${secFeature}`);
                const data = await res.json();
                if (data.error) throw new Error(data.error);

                const isDark = document.body.classList.contains('dark-mode');
                const textColor = isDark ? '#f1f5f9' : '#1e293b';
                const layoutTheme = {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: textColor, family: "'Lora', serif" },
                    margin: { t: 40, r: 20, l: 60, b: 80 },
                    title: { font: { family: "'Playfair Display', serif", size: 16 } },
                    autosize: true
                };

                document.getElementById('plot-bivariate').innerHTML = '';
                Plotly.newPlot('plot-bivariate', Array.isArray(data.plot) ? data.plot : [data.plot], layoutTheme, { responsive: true });
            } catch (err) {
                document.getElementById('plot-bivariate').innerHTML = `<div class="flex items-center justify-center h-full text-red-500 font-mono text-sm">Error: ${err.message}</div>`;
            }
        });
    }

    // Theme Toggle Logic (HerdPulse Style)
    const themeToggleBtn = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    
    // Check saved theme or system preference
    const savedTheme = localStorage.getItem('themePreference');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        document.body.classList.add('dark-mode');
        themeIcon.textContent = '☀️';
    } else {
        themeIcon.textContent = '🌙';
    }

    themeToggleBtn.addEventListener('click', () => {
        const isDark = document.body.classList.toggle('dark-mode');
        themeIcon.textContent = isDark ? '☀️' : '🌙';
        localStorage.setItem('themePreference', isDark ? 'dark' : 'light');
        
        // Trigger a fake window resize or custom event to prompt Plotly to update if needed
        window.dispatchEvent(new Event('themechanged'));
    });

    async function loadDashboardKeyInsights() {
        const isDark = document.body.classList.contains('dark-mode');
        const textColor = isDark ? '#f1f5f9' : '#1e293b';
        const layoutTheme = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: textColor, family: "'Lora', serif" },
            margin: { t: 30, r: 20, l: 40, b: 40 },
            autosize: true
        };

        try {
            const res1 = await fetch('/api/explore/pageViews');
            const data1 = await res1.json();
            const plot1 = document.getElementById('dash-plot-pageviews');
            if (data1.plots && data1.plots.pdp) {
                plot1.classList.remove('flex', 'items-center', 'justify-center', 'text-center');
                plot1.classList.add('block');
                plot1.innerHTML = '';
                Plotly.newPlot('dash-plot-pageviews', [data1.plots.pdp], layoutTheme, { responsive: true });
            } else {
                plot1.innerHTML = 'PDP data not available.';
            }

            const res2 = await fetch('/api/explore/totalHits');
            const data2 = await res2.json();
            const plot2 = document.getElementById('dash-plot-totalhits');
            if (data2.plots && data2.plots.target_scatter) {
                plot2.classList.remove('flex', 'items-center', 'justify-center', 'text-center');
                plot2.classList.add('block');
                plot2.innerHTML = '';
                let traces = [];
                if (data2.plots.target_scatter.is_anomaly) {
                    const isAnom = data2.plots.target_scatter.is_anomaly;
                    const fData = data2.plots.target_scatter.x;
                    const tData = data2.plots.target_scatter.y;
                    const normX = [], normY = [], anomX = [], anomY = [];
                    for (let i = 0; i < isAnom.length; i++) {
                        if (isAnom[i]) {
                            anomX.push(fData[i]);
                            anomY.push(tData[i]);
                        } else {
                            normX.push(fData[i]);
                            normY.push(tData[i]);
                        }
                    }
                    traces = [
                        { x: normX, y: normY, mode: 'markers', type: 'scatter', name: 'Normal', marker: { color: '#e11d48', size: 6, opacity: 0.6 } },
                        { x: anomX, y: anomY, mode: 'markers', type: 'scatter', name: 'Anomaly', marker: { color: '#ef4444', size: 8, opacity: 0.9, symbol: 'x' } }
                    ];
                } else {
                    traces = [{ x: data2.plots.target_scatter.x, y: data2.plots.target_scatter.y, mode: 'markers', type: 'scatter', marker: { color: '#e11d48', size: 6, opacity: 0.6 } }];
                }
                Plotly.newPlot('dash-plot-totalhits', traces, layoutTheme, { responsive: true });
            } else {
                document.getElementById('dash-plot-totalhits').innerHTML = 'Outlier data not available.';
            }

        } catch (err) {
            console.error("Failed to load dashboard insights", err);
            document.getElementById('dash-plot-pageviews').innerHTML = 'Error loading data.';
            document.getElementById('dash-plot-totalhits').innerHTML = 'Error loading data.';
        }
    }
    
    // Load dashboard graphs immediately
    loadDashboardKeyInsights();
});
