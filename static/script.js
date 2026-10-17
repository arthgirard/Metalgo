'use strict';

const FORMATS = ['250g', '1kg', '2kg'];
const FORMAT_LABELS = { '250g': '250g', '1kg': '1 kg', '2kg': '2 kg' };
const FORMAT_CLASSES = { '250g': 'color-250g', '1kg': 'color-1kg', '2kg': 'color-2kg' };

const REFRESH_MS = 5000;        // stats + history poll
const STATUS_MS = 60000;        // open/closed poll
const FLUSH_MS = 15000;         // offline queue retry
const PENDING_KEY = 'metalgo.pending';

let salesChart = null;
let dayCharts = {};             // one Chart.js instance per expanded Historique day
let flushing = false;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Everything rendered through innerHTML goes through this first. /api/log is
// unauthenticated, so anything that reaches the database can also reach every
// tablet's history feed — a stored payload would otherwise execute there.
const HTML_ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
function esc(value) {
    return String(value === null || value === undefined ? '' : value)
        .replace(/[&<>"']/g, ch => HTML_ESCAPES[ch]);
}

function el(id) { return document.getElementById(id); }

function setText(id, value) {
    const node = el(id);
    if (node) node.innerText = value;
}

// Local wall-clock stamp in the exact format the API expects. Deliberately
// not toISOString(), which would convert to UTC and shift every queued sale
// by the timezone offset.
function localStamp(date) {
    const p = n => String(n).padStart(2, '0');
    return `${date.getFullYear()}-${p(date.getMonth() + 1)}-${p(date.getDate())} ` +
           `${p(date.getHours())}:${p(date.getMinutes())}:${p(date.getSeconds())}`;
}

// Parse "YYYY-MM-DD" as a LOCAL date. new Date("2026-08-14") parses as UTC
// midnight, which in Quebec lands on the previous evening — the old weekday
// lookup was only correct because that error and a Monday-indexed array
// cancelled each other out, and would have broken at any positive UTC offset.
function parseLocalDate(dateStr) {
    const [y, m, d] = dateStr.split('-').map(Number);
    return new Date(y, m - 1, d);
}

const WEEKDAYS_FR = ['Dimanche', 'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi'];

function formatDayLabel(dateStr) {
    const date = parseLocalDate(dateStr);
    const [y, m, d] = dateStr.split('-');
    return `${WEEKDAYS_FR[date.getDay()]} ${d}/${m}/${y}`;
}

// ---------------------------------------------------------------------------
// Offline queue
// ---------------------------------------------------------------------------
// A tap that fails used to be swallowed by console.error: the spinner cleared
// and staff assumed the sale registered. Behind a counter on shop wifi that
// silently loses real sales, so failures are now both visible and retried.

function loadQueue() {
    try {
        const raw = JSON.parse(localStorage.getItem(PENDING_KEY));
        return Array.isArray(raw) ? raw : [];
    } catch (err) {
        return [];
    }
}

function saveQueue(queue) {
    try {
        localStorage.setItem(PENDING_KEY, JSON.stringify(queue));
    } catch (err) {
        console.error('Could not persist the pending queue', err);
    }
}

function enqueue(entry) {
    const queue = loadQueue();
    queue.push(entry);
    saveQueue(queue);
}

// Replay queued sales oldest first, stopping at the first failure so order
// is preserved and nothing is dropped.
function flushQueue() {
    if (flushing) return Promise.resolve();
    const queue = loadQueue();
    if (queue.length === 0) return Promise.resolve();

    flushing = true;
    const entry = queue[0];

    return fetch('/api/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(entry)
    })
        .then(res => {
            // A 400 means the server will never accept this entry. Retrying
            // forever would block every later sale behind it.
            if (res.ok || res.status === 400) {
                const rest = loadQueue().slice(1);
                saveQueue(rest);
                return rest.length > 0;
            }
            return false;
        })
        .catch(() => false)
        .then(more => {
            flushing = false;
            if (more) return flushQueue();
            updateStats();
            updateHistory();
        })
        .catch(err => { flushing = false; console.error(err); });
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

function sendLog(type, detail, btnElement = null) {
    if (navigator.vibrate) navigator.vibrate(15);
    if (btnElement) btnElement.classList.add('loading');

    const entry = { type: type, detail: detail, client_time: localStamp(new Date()) };

    fetch('/api/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(entry)
    })
        .then(res => {
            // A 400 is a rejected payload, not a lost connection; queueing it
            // would retry something the server will never accept.
            if (res.status === 400) return;
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            updateStats();
            updateHistory();
        })
        .catch(() => {
            // Keep the sale rather than lose it. Replayed automatically with
            // its original timestamp once the connection returns.
            enqueue(entry);
        })
        .finally(() => { if (btnElement) btnElement.classList.remove('loading'); });
}

function undoLastAction(btnElement = null) {
    if (btnElement) btnElement.style.opacity = '0.5';

    // Drop the newest queued sale first: it hasn't reached the server, so
    // asking the server to undo would delete a different, older sale.
    const queue = loadQueue();
    if (queue.length > 0) {
        queue.pop();
        saveQueue(queue);
        if (btnElement) btnElement.style.opacity = '1';
        return;
    }

    fetch('/api/undo', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                updateStats();
                updateHistory();
            }
        })
        .catch(err => console.error(err))
        .finally(() => { if (btnElement) btnElement.style.opacity = '1'; });
}

// ---------------------------------------------------------------------------
// History feed
// ---------------------------------------------------------------------------

function updateHistory() {
    fetch('/api/history')
        .then(res => res.json())
        .then(data => {
            const container = el('history-list');
            if (!container) return;
            if (data.length === 0) {
                container.innerHTML = '<div class="hist-empty">Aucune vente enregistrée.</div>';
                return;
            }
            container.innerHTML = data.map(item => {
                const colorClass = FORMAT_CLASSES[item.detail] || '';
                // Historical CONVERSION rows are still in the database and
                // must not be described as sales. The feature itself is gone.
                const label = item.type === 'VENTE'
                    ? `Format <strong class="${colorClass}">${esc(item.detail)}</strong> ajouté`
                    : `<span class="hist-other">${esc(item.type)} — ${esc(item.detail)}</span>`;
                return `<div class="hist-row"><span>${label}</span>` +
                       `<span class="time">${esc(item.heure)}</span></div>`;
            }).join('');
        })
        .catch(err => console.error(err));
}

// ---------------------------------------------------------------------------
// Stats + predictions
// ---------------------------------------------------------------------------

function updateStats() {
    fetch('/api/stats')
        .then(res => res.json())
        .then(data => {
            if (el('count-250g')) {
                setText('count-250g', data.c250 || 0);
                setText('count-1kg', data.c1kg || 0);
                setText('count-2kg', data.c2kg || 0);
                setText('stat-peak', data.peak_hour || '--');
                setText('stat-mass', data.total_mass || '0 kg');
            }
            if (data.hourly_data) updateChart(data.hourly_data);
        })
        .catch(err => console.error(err));

    fetch('/api/prediction')
        .then(res => res.json())
        .then(renderPrediction)
        .catch(err => console.error(err));
}

function renderPrediction(data) {
    const eventBanner = el('event-banner');

    if (el('time-left')) {
        const closed = data.debug_info === 'Fermé';
        setText('time-left', closed ? 'Fermé' :
            (data.heures_restantes !== undefined ? `encore ${data.heures_restantes} h` : ''));
    }

    if (eventBanner) {
        if (data.evenement) {
            eventBanner.style.display = 'flex';
            setText('event-name', data.evenement);
        } else {
            eventBanner.style.display = 'none';
        }
    }

    if (data.previsions && el('pred-250g')) {
        // One line under each number: what should be left at closing when
        // there is a plan for today, otherwise the likely range.
        FORMATS.forEach(fmt => {
            setText(`pred-${fmt}`, data.previsions[fmt] || 0);
            const sub = el(`sub-${fmt}`);
            const end = (data.stock_fin_prevu || {})[fmt];
            const low = (data.previsions_min || {})[fmt];
            const high = (data.previsions_max || {})[fmt];
            sub.classList.toggle('stock-out', end !== undefined && end < 0);
            if (end !== undefined) {
                sub.textContent = end < 0 ? `manque ${-end}` : `reste ≈ ${end}`;
            } else {
                sub.textContent = (low === undefined || high === undefined) ? '' : `${low} – ${high}`;
            }
        });
    }

    renderTrend(data.debug_info);
}

function renderTrend(debugInfo) {
    const trendContainer = el('trend-container');
    if (!trendContainer) return;

    const match = (debugInfo || '').match(/(\d+)%/);
    if (!match) {
        trendContainer.style.display = 'none';
        return;
    }
    const percent = parseInt(match[1], 10);
    trendContainer.style.display = 'block';
    const badgeText = el('trend-badge-text');
    if (!badgeText) return;

    if (percent > 110) {
        trendContainer.className = 'trend-badge up';
        badgeText.innerText = `Affluence supérieure prévue (+${percent - 100}%)`;
    } else if (percent < 90) {
        trendContainer.className = 'trend-badge down';
        badgeText.innerText = `Achalandage faible (${percent}%)`;
    } else {
        trendContainer.className = 'trend-badge stable';
        badgeText.innerText = `Tendance stable (${percent}%)`;
    }
}

// ---------------------------------------------------------------------------
// Stock and closing
// ---------------------------------------------------------------------------

let stockData = null;

function stockCell(fmt, entry) {
    let value;
    if (entry.status === 'incoherent') {
        value = '<span class="stock-warn">⚠ stock incohérent</span>' +
                '<span class="stock-hint">Vérifier le plan du jour</span>';
    } else if (entry.remaining <= 0 && entry.status === 'sold_out') {
        value = '<span class="stock-val stock-out">Épuisé</span>';
    } else {
        const cls = entry.remaining <= 0 ? 'stock-out' : FORMAT_CLASSES[fmt];
        value = `<span class="stock-val ${cls}">${esc(entry.remaining)}</span>`;
    }
    return `<div class="w-col">${value}<span class="l">${esc(FORMAT_LABELS[fmt])}</span></div>`;
}

function updateStock() {
    const strip = el('stock-strip');
    if (!strip) return;
    fetch('/api/stock')
        .then(res => res.json())
        .then(data => {
            stockData = data;
            // The sheet first: repacks and the closing are recorded whether
            // or not anyone entered a plan this morning.
            if (el('closing-sheet').style.display !== 'none') renderClosing();
            strip.innerHTML = data.plan
                ? `<div class="week-data">${FORMATS.map(fmt => stockCell(fmt, data.formats[fmt] || {})).join('')}</div>`
                : '<div class="hist-empty">Aucun plan pour aujourd\'hui</div>';
        })
        .catch(err => console.error(err));
}

function renderClosing() {
    const data = stockData || {};
    const list = el('conversion-list');
    const conversions = data.conversions || [];
    list.innerHTML = conversions.length ? conversions.map(c =>
        `<div class="hist-row"><span>${esc(c.source_bags)} × ${esc(FORMAT_LABELS[c.source] || c.source)} → ` +
        `${esc(c.produced_bags)} × ${esc(FORMAT_LABELS[c.target] || c.target)}</span>` +
        `<button class="btn-cancel" aria-label="Supprimer" onclick="deleteConversion(${Number(c.id)})">✕</button></div>`
    ).join('') : '<div class="hist-empty">Aucun sac reconditionné</div>';
    setText('closing-status', data.cloture ? `Clôture enregistrée ${data.cloture.saved_at.slice(11, 16)}` : '');
}

function openClosing() {
    el('closing-sheet').style.display = 'flex';
    el('closing-error').style.display = 'none';
    // Prefilled once on open, not on every refresh, so a refresh never
    // overwrites a number someone is typing.
    const leftovers = (stockData && stockData.cloture && stockData.cloture.leftovers) || {};
    FORMATS.forEach(fmt => {
        const v = leftovers[fmt];
        el(`left-${fmt}`).value = v === null || v === undefined ? '' : v;
    });
    renderClosing();
    updateStock();
}

function closeClosing() {
    el('closing-sheet').style.display = 'none';
}

function closingError(message) {
    const node = el('closing-error');
    node.textContent = message;
    node.style.display = 'block';
}

function postJson(url, method, body) {
    return fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body),
    }).then(res => res.json().then(data => ({ ok: res.ok, data })));
}

function addConversion() {
    const [source, target] = el('conv-pair').value.split('>');
    const bags = Number(el('conv-bags').value);
    postJson('/api/conversion', 'POST', { source, target, source_bags: bags })
        .then(({ ok, data }) => {
            if (!ok) return closingError(data.message || 'Erreur');
            el('closing-error').style.display = 'none';
            updateStock();
        })
        .catch(() => closingError('Connexion impossible, réessayez.'));
}

function deleteConversion(id) {
    postJson(`/api/conversion/${id}`, 'DELETE')
        .then(({ ok, data }) => {
            if (!ok) return closingError(data.message || 'Erreur');
            updateStock();
        })
        .catch(() => closingError('Connexion impossible, réessayez.'));
}

function saveClosing() {
    const values = {};
    let any = false;
    FORMATS.forEach(fmt => {
        const raw = el(`left-${fmt}`).value.trim();
        if (raw !== '') { values[fmt] = Number(raw); any = true; }
    });
    postJson('/api/closing', 'POST', { leftovers: any ? values : null })
        .then(({ ok, data }) => {
            if (!ok) return closingError(data.message || 'Erreur');
            el('closing-error').style.display = 'none';
            updateStock();
        })
        .catch(() => closingError('Connexion impossible, réessayez.'));
}

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------

function chartDatasets(hourlyData, type) {
    const values = fmt => Object.values(hourlyData).map(d => d[fmt] || 0);
    const style = {
        '250g': '#B45309',
        '1kg': '#475569',
        '2kg': '#1E3A8A'
    };
    return FORMATS.map(fmt => {
        const base = { label: FORMAT_LABELS[fmt], data: values(fmt) };
        if (type === 'bar') return Object.assign(base, { backgroundColor: style[fmt] });
        return Object.assign(base, {
            borderColor: style[fmt],
            backgroundColor: style[fmt],
            tension: 0.3,
            borderWidth: fmt === '2kg' ? 3 : 2,
            pointRadius: fmt === '2kg' ? 2 : 1
        });
    });
}

const CHART_SCALES = {
    x: { grid: { display: false }, ticks: { color: '#8C8175', font: { family: 'Inter' } } },
    y: {
        grid: { color: '#E8E2D9' },
        ticks: { color: '#8C8175', stepSize: 5, font: { family: 'Inter' } },
        border: { display: false },
        beginAtZero: true
    }
};

function updateChart(hourlyData) {
    const ctx = el('salesChart');
    if (!ctx) return;

    const labels = Object.keys(hourlyData).map(h => h + 'h');
    const datasets = chartDatasets(hourlyData, 'line');

    // The hour buckets are no longer a fixed 10h-18h window — they follow the
    // day's real opening hours plus any hour that actually has sales — so the
    // labels have to be refreshed too, not just the data.
    if (salesChart) {
        salesChart.data.labels = labels;
        datasets.forEach((ds, i) => { salesChart.data.datasets[i].data = ds.data; });
        salesChart.update();
        return;
    }

    salesChart = new Chart(ctx, {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { display: false }, tooltip: { cornerRadius: 8 } },
            scales: CHART_SCALES
        }
    });
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

function changerOnglet(viewId, button) {
    document.querySelectorAll('.tab-view').forEach(div => div.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(btn => btn.classList.remove('active'));

    el(viewId).classList.add('active');
    button.classList.add('active');

    if (viewId === 'view-stats' || viewId === 'view-predict') updateStats();
}

function switchPredict(mode, btn) {
    document.querySelectorAll('.tab-pill').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    if (mode === 'day') {
        el('predict-day-content').style.display = 'block';
        el('predict-week-content').style.display = 'none';
    } else {
        el('predict-day-content').style.display = 'none';
        el('predict-week-content').style.display = 'block';
        loadPlan();
    }
}

// Refetched on every open. The old guard returned early whenever the
// container already had content, so the view was frozen at whatever it
// showed the first time until someone reloaded the whole page.
function loadPlan() {
    const container = el('plan-list');
    const loading = el('loading-week');
    if (!container) return;
    if (loading) loading.style.display = 'block';

    fetch('/api/plan')
        .then(res => res.json())
        .then(data => {
            if (loading) loading.style.display = 'none';
            if (!Array.isArray(data) || data.length === 0) {
                container.innerHTML =
                    '<div class="model-status-text" style="text-align:center;">Données indisponibles.</div>';
                return;
            }
            // 0.10 reads as "1 jour sur 10": the planner's terms, not a percentage.
            const target = Number(data[0].cible_rupture);
            const intro = target > 0
                ? `<div class="plan-intro">Recommandé pour manquer au plus 1 jour sur ${esc(Math.round(1 / target))}</div>`
                : '';
            const heads = FORMATS.map(fmt =>
                `<span class="${FORMAT_CLASSES[fmt]}">${esc(FORMAT_LABELS[fmt])}</span>`).join('');
            container.innerHTML = intro + `<div class="plan-heads"><span></span>${heads}</div>` +
                data.map(planCard).join('');
            container.querySelectorAll('.plan-input').forEach(recHint);
            container.oninput = event => {
                if (event.target.classList.contains('plan-input')) markEdited(event.target);
            };
        })
        .catch(() => {
            if (loading) loading.style.display = 'none';
            container.innerHTML =
                '<div class="model-status-text" style="text-align:center;">Données indisponibles.</div>';
        });
}

function savedLabel(stamp) {
    return stamp ? `✓ ${esc(stamp.slice(11, 16))}` : '✓';
}

// Last week's same day, as the planner used to read it off paper: only what
// happened that the numbers alone would not show.
function lastWeekLine(week) {
    if (!week) return '';
    const parts = (week.conversions || []).map(c =>
        `${esc(c.source_bags)}×${esc(c.source)}→${esc(c.target)}`);
    (week.ruptures || []).forEach(fmt => parts.push(`rupture ${esc(fmt)}`));
    (week.incoherent || []).forEach(fmt => parts.push(`${esc(fmt)} incohérent`));
    return parts.length ? `<div class="plan-note">Sem. dern. : ${parts.join(' · ')}</div>` : '';
}

// The recommendation only shows when the box says something else: typed
// over, or saved before the recommendation moved (then in orange).
function recHint(input) {
    const rec = input.dataset.rec;
    const hint = input.nextElementSibling;
    if (rec === '' || String(input.value).trim() === rec) {
        hint.textContent = '';
        return;
    }
    hint.textContent = `reco ${rec}`;
    hint.classList.toggle('moved', input.dataset.moved === '1');
}

function markEdited(input) {
    input.closest('.plan-card').classList.add('edited');
    recHint(input);
}

function planCard(day) {
    const saved = day.plan !== null && day.plan !== undefined;
    const moved = day.changement || {};
    const cells = FORMATS.map(fmt => {
        const rec = (day.recommande || {})[fmt];
        const value = saved ? day.plan[fmt] : (rec === undefined ? '' : rec);
        return `<div class="plan-cell">
            <input class="plan-input ${FORMAT_CLASSES[fmt]}" type="number" inputmode="numeric"
                   min="0" max="999" step="1" aria-label="${esc(FORMAT_LABELS[fmt])}"
                   data-format="${esc(fmt)}" data-rec="${esc(rec === undefined ? '' : rec)}"
                   data-moved="${fmt in moved ? 1 : 0}" value="${esc(value)}">
            <span class="plan-hint"></span>
        </div>`;
    }).join('');
    const event = day.evenement ? `<span class="plan-event">${esc(day.evenement)}</span>` : '';

    return `<div class="week-card plan-card${saved ? ' saved' : ''}" data-date="${esc(day.date)}">
        <div class="plan-row">
            <div class="plan-day">
                <span class="week-date">${esc(day.date_affichee)}</span>${event}
                <span class="plan-status">${saved ? savedLabel(day.plan_maj) : ''}</span>
                <button class="btn-utility plan-save" onclick="savePlan('${esc(day.date)}')">Enregistrer</button>
            </div>
            ${cells}
        </div>
        ${lastWeekLine(day.semaine_derniere)}
        <div class="plan-error" style="display: none;"></div>
    </div>`;
}

function savePlan(date) {
    const card = document.querySelector(`.plan-card[data-date="${date}"]`);
    if (!card) return;
    const error = card.querySelector('.plan-error');
    const bags = {};
    card.querySelectorAll('.plan-input').forEach(input => {
        // Sent as typed; the server is the one place that decides what counts
        // as a valid number of bags, and says why when it refuses.
        const n = Number(input.value);
        bags[input.dataset.format] = input.value.trim() === '' ? null : n;
    });

    fetch('/api/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date, bags }),
    })
        .then(res => res.json().then(body => ({ ok: res.ok, body })))
        .then(({ ok, body }) => {
            if (!ok) {
                error.textContent = body.message || 'Erreur';
                error.style.display = 'block';
                return;
            }
            error.style.display = 'none';
            card.classList.add('saved');
            card.classList.remove('edited');
            card.querySelector('.plan-status').innerHTML = savedLabel(body.plan_maj || localStamp(new Date()));
            // Saved now against the current recommendation: no longer "moved".
            card.querySelectorAll('.plan-input').forEach(input => {
                input.dataset.moved = '0';
                recHint(input);
            });
        })
        .catch(() => {
            error.textContent = 'Connexion impossible, réessayez.';
            error.style.display = 'block';
        });
}

function retrainModel() {
    const btn = el('btn-retrain');
    const msg = el('retrain-msg');

    btn.innerText = 'Calcul...';
    btn.classList.add('loading');
    if (msg) msg.style.display = 'none';

    fetch('/api/retrain', { method: 'POST' })
        .then(res => res.json().then(body => ({ ok: res.ok, body: body })))
        .then(({ ok, body }) => {
            btn.innerText = 'Recalibrer';
            btn.classList.remove('loading');
            if (!msg) return;
            msg.innerText = ok ? 'Terminé avec succès.' : (body.message || 'Erreur.');
            msg.style.display = 'block';
            msg.style.color = ok ? '#059669' : '#DC2626';
            setTimeout(() => { msg.style.display = 'none'; updateStats(); }, 3000);
        })
        .catch(() => {
            btn.innerText = 'Recalibrer';
            btn.classList.remove('loading');
            if (!msg) return;
            msg.innerText = 'Erreur de connexion.';
            msg.style.display = 'block';
            msg.style.color = '#DC2626';
        });
}

function checkOpenStatus() {
    fetch('/api/status')
        .then(res => res.json())
        .then(data => {
            // data-ouvert drives the pill's colour from CSS; the text comes
            // straight from the server so the two can't disagree.
            document.body.setAttribute('data-ouvert', data.ouvert ? 'true' : 'false');
            setText('status-pill', data.message);
        })
        .catch(err => console.error(err));
}

// Historique is a sub-page of Rapport, so it is opened separately from
// changerOnglet to leave the bottom nav highlighted on "Rapport".
function openHistory() {
    document.querySelectorAll('.tab-view').forEach(div => div.classList.remove('active'));
    el('view-history').classList.add('active');
    loadHistoryDays();
}

function closeHistory() {
    el('view-history').classList.remove('active');
    el('view-stats').classList.add('active');
}

// Refetched on every open, for the same reason as the weekly forecast.
function loadHistoryDays() {
    const container = el('history-days-list');
    if (!container) return;

    fetch('/api/history_days')
        .then(res => res.json())
        .then(days => {
            if (days.length === 0) {
                container.innerHTML =
                    '<div class="model-status-text" style="text-align:center;">Aucune donnée.</div>';
                return;
            }
            // Existing charts belong to DOM nodes about to be replaced.
            Object.values(dayCharts).forEach(chart => chart.destroy());
            dayCharts = {};

            container.innerHTML = days.map(day => `
                <div class="day-card">
                    <div class="day-header" onclick="toggleDay('${esc(day.date)}')">
                        <span class="day-date">${esc(formatDayLabel(day.date))}</span>
                        <div class="day-totals">
                            <div class="day-total-col"><span class="stat-val color-250g">${esc(day.c250)}</span><span class="stat-lbl">250g</span></div>
                            <div class="day-total-col"><span class="stat-val color-1kg">${esc(day.c1kg)}</span><span class="stat-lbl">1kg</span></div>
                            <div class="day-total-col"><span class="stat-val color-2kg">${esc(day.c2kg)}</span><span class="stat-lbl">2kg</span></div>
                        </div>
                    </div>
                    <div class="day-expand chart-box" id="day-expand-${esc(day.date)}" style="display:none;">
                        <canvas id="day-chart-${esc(day.date)}"></canvas>
                    </div>
                </div>`).join('');
        })
        .catch(err => console.error(err));
}

function toggleDay(dateStr) {
    const expandDiv = el(`day-expand-${dateStr}`);
    if (!expandDiv) return;

    if (expandDiv.style.display === 'block') {
        expandDiv.style.display = 'none';
        return;
    }
    // Unhide before drawing, or Chart.js sizes the canvas to zero.
    expandDiv.style.display = 'block';
    if (dayCharts[dateStr]) return;

    fetch(`/api/history_day/${encodeURIComponent(dateStr)}`)
        .then(res => res.json())
        .then(hourlyData => {
            const ctx = el(`day-chart-${dateStr}`);
            if (!ctx || hourlyData.error) return;
            dayCharts[dateStr] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: Object.keys(hourlyData).map(h => h + 'h'),
                    datasets: chartDatasets(hourlyData, 'bar')
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false }, tooltip: { cornerRadius: 8 } },
                    scales: CHART_SCALES
                }
            });
        })
        .catch(err => console.error(err));
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    flushQueue();
    updateStats();
    updateHistory();
    updateStock();
    checkOpenStatus();

    setInterval(() => { updateStats(); updateHistory(); updateStock(); }, REFRESH_MS);
    setInterval(checkOpenStatus, STATUS_MS);
    setInterval(flushQueue, FLUSH_MS);

    // Retry the moment connectivity returns rather than waiting out the timer.
    window.addEventListener('online', flushQueue);
});
