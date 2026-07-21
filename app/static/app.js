document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('dropZone');
  const pdfFileInput = document.getElementById('pdfFileInput');
  const selectFileBtn = document.getElementById('selectFileBtn');
  const fileInfo = document.getElementById('fileInfo');
  const progressWrap = document.getElementById('progressWrap');
  const progressBarFill = document.getElementById('progressBarFill');
  const statusText = document.getElementById('statusText');
  const resultsSection = document.getElementById('resultsSection');
  const downloadJsonBtn = document.getElementById('downloadJsonBtn');

  let selectedFile = null;
  let lastExtractionData = null;

  // File selection
  selectFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    pdfFileInput.click();
  });

  pdfFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Drag and drop handlers
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert('Please select a valid .pdf file.');
      return;
    }
    selectedFile = file;
    fileInfo.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    startUpload(file);
  }

  async function startUpload(file) {
    progressWrap.style.display = 'block';
    resultsSection.style.display = 'none';
    progressBarFill.style.width = '15%';
    statusText.textContent = 'Rendering PDF & running Multithreaded Vision OCR...';

    const formData = new FormData();
    formData.append('file', file);

    try {
      progressBarFill.style.width = '45%';
      statusText.textContent = 'Indexing chunks in Qdrant Vector DB & running LangGraph Agent...';

      const response = await fetch('/process-pdf', {
        method: 'POST',
        body: formData
      });

      progressBarFill.style.width = '85%';
      statusText.textContent = 'Computing math ratios & formatting output...';

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned status ${response.status}`);
      }

      const result = await response.json();
      progressBarFill.style.width = '100%';
      statusText.textContent = 'Extraction Completed Successfully!';

      setTimeout(() => {
        progressWrap.style.display = 'none';
        displayResults(result);
      }, 600);

    } catch (err) {
      progressBarFill.style.width = '100%';
      progressBarFill.style.background = '#ff4b4b';
      statusText.textContent = `Error: ${err.message}`;
      alert(`Extraction Error: ${err.message}`);
    }
  }

  function displayResults(result) {
    const data = result.data || {};
    lastExtractionData = data;
    resultsSection.style.display = 'block';

    // Meta
    document.getElementById('metaCompany').textContent = data.company_name || 'N/A';
    document.getElementById('metaTicker').textContent = data.ticker || 'N/A';
    document.getElementById('metaCountry').textContent = data.country || 'N/A';
    document.getElementById('metaCurrency').textContent = data.currency || 'N/A';

    // Ratios Grid
    const ratiosGrid = document.getElementById('ratiosGrid');
    ratiosGrid.innerHTML = '';

    const ratios = data.ratios || {};
    for (const [ratioName, periodObj] of Object.entries(ratios)) {
      if (!periodObj || typeof periodObj !== 'object') continue;

      const card = document.createElement('div');
      card.className = 'ratio-card';

      let valsHtml = '';
      for (const [period, val] of Object.entries(periodObj)) {
        if (period === 'source') continue;
        if (val !== null && val !== undefined) {
          valsHtml += `
            <div class="ratio-val-row">
              <span style="color: var(--text-muted); font-size: 0.9rem;">${period}</span>
              <span class="ratio-val">${val}</span>
            </div>
          `;
        }
      }

      const formula = periodObj.source || '';

      card.innerHTML = `
        <div class="ratio-title">${ratioName}</div>
        ${valsHtml || '<div style="color: var(--text-muted); font-size: 0.9rem;">N/A</div>'}
        <div class="ratio-formula">${formula}</div>
      `;
      ratiosGrid.appendChild(card);
    }

    // Statement Table
    const tbody = document.querySelector('#statementTable tbody');
    tbody.innerHTML = '';

    const stmtData = data.statement_data || {};
    for (const [field, periodObj] of Object.entries(stmtData)) {
      const tr = document.createElement('tr');
      
      let valsStr = 'N/A';
      if (periodObj && typeof periodObj === 'object') {
        valsStr = Object.entries(periodObj)
          .map(([p, v]) => `<strong>${p}:</strong> ${v}`)
          .join(' &nbsp;|&nbsp; ');
      } else if (periodObj !== null && periodObj !== undefined) {
        valsStr = periodObj;
      }

      tr.innerHTML = `
        <td style="font-weight: 600;">${field}</td>
        <td>${valsStr}</td>
      `;
      tbody.appendChild(tr);
    }
  }

  // JSON Download
  downloadJsonBtn.addEventListener('click', () => {
    if (!lastExtractionData) return;
    const jsonStr = JSON.stringify(lastExtractionData, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${(lastExtractionData.company_name || 'report').replace(/\s+/g, '_')}_extraction.json`;
    a.click();
    URL.revokeObjectURL(url);
  });
});
