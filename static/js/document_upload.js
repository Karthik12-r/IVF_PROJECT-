(() => {
    const fieldIds = {
        age: 'Age', amh: 'AMH', fsh: 'FSH',
        lh: null, afc: null, uterus: null, ovaries: null
    };
    const labelFor = {
        Age: 'Age', BMI: 'BMI', AMH: 'AMH', FSH: 'FSH',
        Previous_IVF_Attempts: 'Previous IVF Attempts', Stress_Level: 'Stress Level',
        Sleep_Hours: 'Sleep Hours', Exercise_Min_per_Day: 'Exercise'
    };

    function renderMessage(target, message, style = 'info') {
        if (!target) return;
        const box = document.createElement('div');
        box.className = `alert alert-${style} mb-0`;
        box.textContent = message;
        target.replaceChildren(box);
    }

    document.querySelectorAll('form[data-document-upload]').forEach(form => {
        form.addEventListener('submit', async event => {
            event.preventDefault();
            const target = form.parentElement.querySelector('[data-document-upload-status]');
            const button = form.querySelector('button[type="submit"]');
            const originalText = button ? button.textContent : '';
            if (button) { button.disabled = true; button.textContent = 'Checking report…'; }
            renderMessage(target, 'Checking report, then processing a new document if needed…');
            try {
                const response = await fetch(form.action, {
                    method: 'POST',
                    body: new FormData(form),
                    credentials: 'same-origin',
                    headers: { 'X-Requested-With': 'XMLHttpRequest', 'Accept': 'application/json' }
                });
                const result = await response.json();
                if (!response.ok) throw new Error(result.message || result.error || 'Document processing failed.');

                const predictor = document.getElementById('prediction-form');
                const outcomeText = result.status === 'DUPLICATE'
                    ? 'Duplicate report detected. OCR processing skipped. Existing extracted data reused.'
                    : result.status === 'OLD_REPORT'
                        ? 'Older report detected. Current dashboard values were not overwritten; the file remains in history.'
                        : result.status === 'VALIDATED'
                            ? 'New report processed. Validated extracted features have been saved and dashboard values updated.'
                            : result.message;
                renderMessage(target, outcomeText, result.status === 'VALIDATED' || result.status === 'DUPLICATE' ? 'success' : result.status === 'FAILED' ? 'danger' : 'warning');

                if (predictor && (result.status === 'VALIDATED' || result.status === 'DUPLICATE')) {
                    Object.entries(result.dashboard_values || {}).forEach(([key, value]) => {
                        const input = document.getElementById(key);
                        if (input && value !== null && value !== undefined) input.value = value;
                        const source = document.getElementById(`source-${key}`);
                        const provenance = (result.feature_sources || {})[key];
                        if (source && provenance) {
                            const details = [provenance.source, provenance.document, provenance.report_date].filter(Boolean);
                            source.textContent = details.join(' · ');
                        }
                    });
                    const sourceId = document.getElementById('source_document_id');
                    if (sourceId && result.latest_report_id) sourceId.value = result.latest_report_id;
                    const missing = result.missing_prediction_fields || [];
                    const status = document.getElementById('prediction-input-status');
                    if (missing.length) {
                        status.textContent = `Prediction requires additional validated information: ${missing.map(v => labelFor[v] || v).join(', ')}.`;
                        status.className = 'small text-warning mb-2';
                    } else if (result.status !== 'VALIDATED' && result.status !== 'DUPLICATE') {
                        status.textContent = 'Review the report values before running a prediction.';
                    } else {
                        status.textContent = 'Validated values are ready. Preparing the existing prediction endpoint…';
                    }

                    const candidates = result.validation || {};
                    const reviewFields = Object.entries(candidates).filter(([, item]) => item.status === 'REQUIRES_REVIEW');
                    if (reviewFields.length) {
                        const review = document.createElement('div');
                        review.className = 'alert alert-warning mt-3';
                        const title = document.createElement('strong');
                        title.textContent = 'Some OCR values require review:';
                        review.append(title);
                        const list = document.createElement('ul');
                        reviewFields.forEach(([field, item]) => {
                            const row = document.createElement('li');
                            row.textContent = `${field.toUpperCase()}: ${item.value ?? 'unavailable'} — ${item.reason}`;
                            list.append(row);
                        });
                        review.append(list);
                        target.append(review);
                    }
                    if (result.prediction_ready && !missing.length) {
                        window.setTimeout(() => predictor.requestSubmit(), 250);
                    }
                } else if (predictor && result.status === 'VALIDATION_REQUIRED') {
                    const candidates = result.extracted_features || {};
                    const review = Object.entries(fieldIds).filter(([field, input]) => input && candidates[field] !== null && candidates[field] !== undefined);
                    if (review.length) {
                        const detail = review.map(([field, input]) => `${labelFor[input]} candidate: ${candidates[field]}`).join('; ');
                        target.lastElementChild.textContent += ` Review before entering these values: ${detail}`;
                    }
                } else if (!predictor && !['DUPLICATE', 'OLD_REPORT'].includes(result.status)) {
                    window.location.reload();
                }
                form.reset();
            } catch (error) {
                renderMessage(target, error.message || 'Unable to process this document.', 'danger');
            } finally {
                if (button) { button.disabled = false; button.textContent = originalText; }
            }
        });
    });
})();
