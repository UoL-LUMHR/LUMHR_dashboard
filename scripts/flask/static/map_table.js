/* Shared table view of the LSOA data already loaded by each map. */
(() => {
  const shell = document.querySelector('.map-shell');
  if (!shell) return;

  const controls = document.createElement('div');
  controls.className = 'map-view-switch';
  controls.setAttribute('aria-label', 'Choose view');
  controls.innerHTML = '<button type="button" aria-pressed="true">Map</button><button type="button" aria-pressed="false">Table</button>';
  const panel = document.createElement('section');
  panel.className = 'lsoa-table-panel';
  panel.hidden = true;
  panel.innerHTML = '<div class="lsoa-table-heading"><div><h2>LSOA data</h2><p>Data for the current map selection. Select a column heading to sort. Scroll across to see all columns.</p></div><label>Search areas <input type="search" placeholder="Name or code" aria-label="Search LSOA name or code"></label><button type="button" class="lsoa-csv-export">Download CSV</button></div><p class="lsoa-table-count" role="status"></p><div class="lsoa-table-scroll" tabindex="0" aria-label="Scrollable LSOA data table"><table><thead></thead><tbody></tbody></table></div>';
  shell.append(controls, panel);
  const [mapButton, tableButton] = controls.querySelectorAll('button');
  const search = panel.querySelector('input');
  const head = panel.querySelector('thead');
  const body = panel.querySelector('tbody');
  const count = panel.querySelector('.lsoa-table-count');
  let last, rows = [], columns = [];
  let sortKey = "name", sortDirection = 1;
  // Popup content is the source of truth for the columns shown on these maps.
  function popupDetails(layer) {
    const content = layer.getPopup()?.getContent();
    if (typeof content !== 'string') return {};
    const container = document.createElement('div');
    container.innerHTML = content;
    const fields = {};
    container.querySelectorAll('strong').forEach(label => {
      const name = label.textContent.trim().replace(/:\s*$/, '');
      if (!label.textContent.trim().endsWith(':') || ['LSOA', 'LSOA Name'].includes(name)) return;
      let value = '';
      for (let node = label.nextSibling; node; node = node.nextSibling) {
        if (node.nodeType === Node.ELEMENT_NODE && ['BR', 'DETAILS', 'STRONG', 'HR'].includes(node.tagName)) break;
        value += node.textContent;
      }
      fields[name] = value.trim() || null;
    });
    return fields;
  }
  const format = value => value == null || value === '' ? 'N/A' : typeof value === 'number' ? (Number.isInteger(value) ? value.toLocaleString() : Number(value.toFixed(4)).toLocaleString(undefined, { maximumFractionDigits: 4 })) : typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value);
  const getValue = (row, key) => key === 'name' || key === 'code' || key === 'value'
    ? row[key] : row.details[key.slice(7)];
  const missing = value => value == null || value === '' || value === 'N/A' || value === 'Not available' || (typeof value === 'number' && !Number.isFinite(value));
  const numeric = value => {
    if (typeof value === 'number') return Number.isFinite(value) ? value : null;
    if (typeof value !== 'string') return null;
    const match = value.trim().match(/^([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?:\s*(?:%|min|Mbps))?(?:\s*\([^)]*\))?$/i);
    return match ? Number(match[1].replace(/,/g, '')) : null;
  };
  function draw() {
    const query = search.value.trim().toLowerCase();
    const filtered = rows.filter(row => !query || `${row.name} ${row.code}`.toLowerCase().includes(query));
    filtered.sort((a, b) => {
      const left = getValue(a, sortKey), right = getValue(b, sortKey);
      if (missing(left) || missing(right)) return missing(left) === missing(right) ? 0 : missing(left) ? 1 : -1;
      const leftNumber = numeric(left), rightNumber = numeric(right);
      const result = leftNumber !== null && rightNumber !== null
        ? leftNumber - rightNumber
        : String(left).localeCompare(String(right), undefined, { numeric: true, sensitivity: 'base' });
      return sortDirection * result || a.name.localeCompare(b.name, undefined, { numeric: true });
    });
    head.querySelectorAll('th').forEach(th => {
      const active = th.dataset.key === sortKey;
      th.setAttribute('aria-sort', active ? (sortDirection === 1 ? 'ascending' : 'descending') : 'none');
      th.querySelector('.sort-indicator').textContent = active ? (sortDirection === 1 ? '▲' : '▼') : '↕';
    });
    count.textContent = `${filtered.length.toLocaleString()} of ${rows.length.toLocaleString()} LSOAs`;
    const fragment = document.createDocumentFragment();
    for (const row of filtered) {
      const tr = document.createElement('tr');
      for (const value of [row.name, row.code, row.value, ...columns.map(key => row.details[key])]) {
        const td = document.createElement('td'); td.textContent = format(value); tr.appendChild(td);
      }
      fragment.appendChild(tr);
    }
    body.replaceChildren(fragment);
  }
  function refresh(force = false) {
    if (panel.hidden || !window.lumhrTableSnapshot) return;
    const snapshot = window.lumhrTableSnapshot();
    if (!snapshot.layer || !snapshot.data) return;
    if (!force && last && last.layer === snapshot.layer && last.data === snapshot.data && last.scores === snapshot.scores && last.version === snapshot.version && last.metric === snapshot.metric) return;
    last = snapshot;
    rows = [];
    snapshot.layer.eachLayer(layer => {
      const props = layer.feature?.properties || {};
      const code = props.CODE ?? props.LSOA_CODE ?? props.LSOA21CD ?? props.lsoa21cd;
      if (!code) return;
      const source = snapshot.data[code] || {};
      const name = source.lsoa_name || source.name || props.LSOA21NM || props.NAME || code;
      const details = snapshot.details ? snapshot.details(code) : popupDetails(layer);
      rows.push({ code, name, value: snapshot.value ? snapshot.value(code) : snapshot.scores?.[code], details });
    });
    columns = [...new Set(rows.flatMap(row => Object.keys(row.details)))];
    const definitions = [
      ['name', 'LSOA area name'], ['code', 'LSOA code'], ['value', snapshot.metric || 'Selected map value'],
      ...columns.map(key => [`detail:${key}`, key])
    ];
    if (!definitions.some(([key]) => key === sortKey)) { sortKey = 'name'; sortDirection = 1; }
    const tr = document.createElement('tr');
    for (const [key, title] of definitions) {
      const th = document.createElement('th'); th.scope = 'col'; th.dataset.key = key;
      const button = document.createElement('button'); button.type = 'button'; button.className = 'lsoa-sort';
      button.append(document.createTextNode(title + ' '));
      const indicator = document.createElement('span'); indicator.className = 'sort-indicator'; indicator.setAttribute('aria-hidden', 'true'); button.appendChild(indicator);
      button.addEventListener('click', () => {
        if (sortKey === key) sortDirection *= -1;
        else {
          sortKey = key;
          const values = rows.map(row => getValue(row, key)).filter(value => !missing(value));
          sortDirection = values.length && values.every(value => numeric(value) !== null) ? -1 : 1;
        }
        draw();
      });
      th.appendChild(button); tr.appendChild(th);
    }
    head.replaceChildren(tr); draw();
  }
  function setView(table) {
    panel.hidden = !table;
    const exportControls = document.getElementById('lumhr_map_export_controls');
    if (exportControls) exportControls.hidden = table;
    mapButton.setAttribute('aria-pressed', String(!table));
    tableButton.setAttribute('aria-pressed', String(table));
    if (table) refresh(true);
    else if (window.lumhrExportMap) window.lumhrExportMap()?.invalidateSize();
  }
  mapButton.addEventListener('click', () => setView(false));
  tableButton.addEventListener('click', () => setView(true));
  panel.querySelector('.lsoa-csv-export').addEventListener('click', () => {
    // Read the rendered cells so the download matches the visible filter and sort.
    const headings = [...head.querySelectorAll('th button')].map(button => button.firstChild.textContent.trim());
    if (!headings.length) return;
    const data = [...body.querySelectorAll('tr')].map(row => [...row.cells].map(cell => cell.textContent));
    const escapeCell = value => {
      let text = String(value ?? '');
      // Prevent spreadsheet applications interpreting text as a formula.
      if (/^[\s]*[=+@-]/.test(text) && numeric(text) === null) text = "'" + text;
      return `"${text.replace(/"/g, '""')}"`;
    };
    const csv = [headings, ...data].map(row => row.map(escapeCell).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob(['\ufeff', csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `${location.pathname.replace(/^\/+|\/+$/g, '').replace(/[^a-z0-9_-]+/gi, '-') || 'lsoa'}-lsoa-data.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  search.addEventListener('input', draw);
  window.setInterval(refresh, 1000);
})();
