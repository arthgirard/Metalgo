let isGlobalOpen = document.body.getAttribute('data-ouvert') === 'true';
let salesChart = null;
let dayCharts = {}; // one Chart.js instance per expanded Historique day

// initialize view
document.addEventListener('DOMContentLoaded', () => {
    updateStats();
    updateHistory();

    // automatic data update
    setInterval(() => {
        updateStats();
        updateHistory();
    }, 3000);

    setInterval(checkOpenStatus, 60000); 
});

// tab switching logic
function changerOnglet(viewId, button) {
    document.querySelectorAll('.tab-view').forEach(div => div.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(btn => btn.classList.remove('active'));

    document.getElementById(viewId).classList.add('active');
    button.classList.add('active');

    if (viewId === 'view-stats' || viewId === 'view-predict') updateStats();
}

// log new sale action
function sendLog(type, detail, btnElement = null) {
    if (navigator.vibrate) navigator.vibrate(15); 
    if (btnElement) btnElement.classList.add('loading');

    fetch('/api/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: type, detail: detail })
    })
    .then(res => res.json())
    .then(() => { updateStats(); updateHistory(); })
    .catch(err => console.error(err))
    .finally(() => { if (btnElement) btnElement.classList.remove('loading'); });
}

// revert the last recorded action
function undoLastAction(btnElement = null) {
    if (btnElement) btnElement.style.opacity = '0.5';

    fetch('/api/undo', { method: 'POST' })
    .then(res => res.json())
    .then(data => { if (data.status === 'success') { updateStats(); updateHistory(); } })
    .catch(err => console.error(err))
    .finally(() => { if (btnElement) btnElement.style.opacity = '1'; });
}

// refresh history feed with colors
function updateHistory() {
    fetch('/api/history')
    .then(res => res.json())
    .then(data => {
        const container = document.getElementById('history-list');
        container.innerHTML = '';
        if (data.length === 0) {
            container.innerHTML = '<div class="hist-empty">Aucune vente enregistrée.</div>';
            return;
        }
        data.forEach(item => {
            let colorClass = '';
            if (item.detail === '250g') colorClass = 'color-250g';
            else if (item.detail === '1kg' || item.detail === '1 kg') colorClass = 'color-1kg';
            else if (item.detail === '2kg' || item.detail === '2 kg') colorClass = 'color-2kg';

            container.innerHTML += `
                <div class="hist-row">
                    <span>Format <strong class="${colorClass}">${item.detail}</strong> ajouté</span>
                    <span class="time">${item.heure}</span>
                </div>
            `;
        });
    });
}

// fetch statistics and AI predictions
function updateStats() {
    fetch('/api/stats')
    .then(res => res.json())
    .then(data => {
        if(document.getElementById('count-250g')) {
            document.getElementById('count-250g').innerText = data.c250 || 0;
            document.getElementById('count-1kg').innerText = data.c1kg || 0;
            document.getElementById('count-2kg').innerText = data.c2kg || 0;
            document.getElementById('stat-peak').innerText = data.peak_hour || "--";
            document.getElementById('stat-mass').innerText = data.total_mass || "0 kg";
        }
        if(data.hourly_data) updateChart(data.hourly_data);
    }).catch(err => console.error(err));

    fetch('/api/prediction')
    .then(res => res.json())
    .then(data => {
        const trendContainer = document.getElementById('trend-container');
        const eventBanner = document.getElementById('event-banner');

        if(document.getElementById('time-left')) {
            document.getElementById('time-left').innerText = data.heures_restantes !== undefined ? data.heures_restantes : "--";
            document.getElementById('meteo-label').innerText = data.meteo || "--";
        }

        if (data.evenement && eventBanner) {
            eventBanner.style.display = 'flex';
            document.getElementById('event-name').innerText = data.evenement;
        } else if (eventBanner) {
            eventBanner.style.display = 'none';
        }

        if (data.previsions && document.getElementById('pred-250g')) {
            document.getElementById('pred-250g').innerText = data.previsions['250g'] || 0;
            document.getElementById('pred-1kg').innerText = data.previsions['1kg'] || 0;
            document.getElementById('pred-2kg').innerText = data.previsions['2kg'] || 0;
        }

        if(document.getElementById('prediction-info')) {
            document.getElementById('prediction-info').innerText = data.debug_info || "Modèle prêt";
        }

        if(trendContainer) {
            const match = (data.debug_info || "").match(/(\d+)%/);
            if (match) {
                let percent = parseInt(match[1]);
                trendContainer.style.display = 'block';
                const badgeText = document.getElementById('trend-badge-text');
                
                if (percent > 110) {
                    trendContainer.className = 'trend-badge up';
                    badgeText.innerText = `Affluence supérieure prévue (+${percent-100}%)`;
                } else if (percent < 90) {
                    trendContainer.className = 'trend-badge down';
                    badgeText.innerText = `Achalandage faible (${percent}%)`;
                } else {
                    trendContainer.className = 'trend-badge stable';
                    badgeText.innerText = `Tendance stable (${percent}%)`;
                }
            } else {
                trendContainer.style.display = 'none'; 
            }
        }
    }).catch(err => console.error(err));
}

// build and update line chart for hourly sales
function updateChart(hourlyData) {
    const ctx = document.getElementById('salesChart');
    if(!ctx) return;

    const labels = Object.keys(hourlyData).map(h => h + 'h');
    const data250 = Object.values(hourlyData).map(d => d['250g']);
    const data1kg = Object.values(hourlyData).map(d => d['1kg']);
    const data2kg = Object.values(hourlyData).map(d => d['2kg']);

    if(salesChart) {
        salesChart.data.datasets[0].data = data250;
        salesChart.data.datasets[1].data = data1kg;
        salesChart.data.datasets[2].data = data2kg;
        salesChart.update();
    } else {
        salesChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    { label: '250g', data: data250, borderColor: '#B45309', backgroundColor: '#B45309', tension: 0.3, borderWidth: 2, pointRadius: 1 },
                    { label: '1 kg', data: data1kg, borderColor: '#475569', backgroundColor: '#475569', tension: 0.3, borderWidth: 2, pointRadius: 1 },
                    { label: '2 kg', data: data2kg, borderColor: '#1E3A8A', backgroundColor: '#1E3A8A', tension: 0.3, borderWidth: 3, pointRadius: 2 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: { legend: { display: false }, tooltip: { cornerRadius: 8 } },
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#8C8175', font: {family: 'Inter'} } },
                    y: { grid: { color: '#E8E2D9' }, ticks: { color: '#8C8175', stepSize: 5, font: {family: 'Inter'} }, border: {display: false} }
                }
            }
        });
    }
}

// toggle prediction sub-views
function switchPredict(mode, btn) {
    document.querySelectorAll('.tab-pill').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    if (mode === 'day') {
        document.getElementById('predict-day-content').style.display = 'block';
        document.getElementById('predict-week-content').style.display = 'none';
    } else {
        document.getElementById('predict-day-content').style.display = 'none';
        document.getElementById('predict-week-content').style.display = 'block';
        loadWeeklyForecast();
    }
}

// pull 7-day model projections
function loadWeeklyForecast() {
    const container = document.getElementById('week-list');
    const loading = document.getElementById('loading-week');
    if(container.innerHTML.trim() !== "") return;

    fetch('/api/forecast_week')
    .then(res => res.json())
    .then(data => {
        loading.style.display = 'none';
        container.innerHTML = '';

        if(data.error || data.length === 0) {
            container.innerHTML = `<div class="model-status-text" style="text-align:center;">Données indisponibles.</div>`;
            return;
        }

        data.forEach(jour => {
            if (jour.ferme) return; 
            container.innerHTML += `
                <div class="week-card">
                    <div class="week-header">
                        <span class="week-date">${jour.date_affichee}</span>
                        <span class="week-meteo">${jour.meteo}</span>
                    </div>
                    <div class="week-data">
                        <div class="w-col"><span class="v color-250g">${jour.totals['250g']}</span><span class="l">250g</span></div>
                        <div class="w-col"><span class="v color-1kg">${jour.totals['1kg']}</span><span class="l">1 kg</span></div>
                        <div class="w-col"><span class="v color-2kg">${jour.totals['2kg']}</span><span class="l">2 kg</span></div>
                    </div>
                </div>
            `;
        });
    });
}

// force server retraining and db refresh
function retrainModel() {
    const btn = document.getElementById('btn-retrain');
    const msg = document.getElementById('retrain-msg');
    
    btn.innerText = "Calcul...";
    btn.classList.add('loading');
    msg.style.display = 'none';

    fetch('/api/retrain', { method: 'POST' })
    .then(res => res.json())
    .then(() => {
        btn.innerText = "Recalibrer";
        btn.classList.remove('loading');
        msg.innerText = "Terminé avec succès.";
        msg.style.display = 'block';
        msg.style.color = "#059669"; 
        setTimeout(() => { msg.style.display = 'none'; updateStats(); }, 3000);
    })
    .catch(err => {
        btn.innerText = "Recalibrer";
        btn.classList.remove('loading');
        msg.innerText = "Erreur de connexion.";
        msg.style.display = 'block';
        msg.style.color = "#DC2626"; 
    });
}

// verify if app should be locked
function checkOpenStatus() {
    fetch('/api/status')
    .then(res => res.json())
    .then(data => {
        isGlobalOpen = data.ouvert;
        document.body.setAttribute('data-ouvert', isGlobalOpen);
    });
}

// open the Historique page (kept separate from changerOnglet so the
// bottom nav stays on "Rapport", since Historique is a sub-page of it)
function openHistory() {
    document.querySelectorAll('.tab-view').forEach(div => div.classList.remove('active'));
    document.getElementById('view-history').classList.add('active');
    loadHistoryDays();
}

// return from Historique back to the Rapport tab
function closeHistory() {
    document.getElementById('view-history').classList.remove('active');
    document.getElementById('view-stats').classList.add('active');
}

// reformat an ISO date (YYYY-MM-DD) to DD/MM/YYYY for display
function formatDayLabel(dateStr) {
    const [y, m, d] = dateStr.split('-');
    return `${d}/${m}/${y}`;
}

// list every day that has reported sales, newest first
function loadHistoryDays() {
    const container = document.getElementById('history-days-list');
    if (container.innerHTML.trim() !== "") return;

    fetch('/api/history_days')
    .then(res => res.json())
    .then(days => {
        container.innerHTML = '';
        if (days.length === 0) {
            container.innerHTML = '<div class="model-status-text" style="text-align:center;">Aucune donnée.</div>';
            return;
        }
        days.forEach(day => {
            container.innerHTML += `
                <div class="day-card">
                    <div class="day-header" onclick="toggleDay('${day.date}')">
                        <span class="day-date">${formatDayLabel(day.date)}</span>
                        <div class="day-totals">
                            <div class="day-total-col"><span class="stat-val color-250g">${day.c250}</span><span class="stat-lbl">250g</span></div>
                            <div class="day-total-col"><span class="stat-val color-1kg">${day.c1kg}</span><span class="stat-lbl">1kg</span></div>
                            <div class="day-total-col"><span class="stat-val color-2kg">${day.c2kg}</span><span class="stat-lbl">2kg</span></div>
                        </div>
                    </div>
                    <div class="day-expand chart-box" id="day-expand-${day.date}" style="display:none;">
                        <canvas id="day-chart-${day.date}"></canvas>
                    </div>
                </div>
            `;
        });
    });
}

// expand/collapse a day and draw its hourly chart the first time it's opened
function toggleDay(dateStr) {
    const expandDiv = document.getElementById(`day-expand-${dateStr}`);
    const isOpen = expandDiv.style.display === 'block';

    if (isOpen) {
        expandDiv.style.display = 'none';
        return;
    }
    // unhide before drawing, or Chart.js sizes the canvas to zero
    expandDiv.style.display = 'block';
    if (dayCharts[dateStr]) return;

    fetch(`/api/history_day/${dateStr}`)
    .then(res => res.json())
    .then(hourlyData => {
        const ctx = document.getElementById(`day-chart-${dateStr}`);
        const labels = Object.keys(hourlyData).map(h => h + 'h');
        const data250 = Object.values(hourlyData).map(d => d['250g']);
        const data1kg = Object.values(hourlyData).map(d => d['1kg']);
        const data2kg = Object.values(hourlyData).map(d => d['2kg']);

        dayCharts[dateStr] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    { label: '250g', data: data250, backgroundColor: '#B45309' },
                    { label: '1 kg', data: data1kg, backgroundColor: '#475569' },
                    { label: '2 kg', data: data2kg, backgroundColor: '#1E3A8A' }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { cornerRadius: 8 } },
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#8C8175', font: {family: 'Inter'} } },
                    y: { grid: { color: '#E8E2D9' }, ticks: { color: '#8C8175', stepSize: 5, font: {family: 'Inter'} }, border: {display: false}, beginAtZero: true }
                }
            }
        });
    });
}
