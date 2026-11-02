import { createPercentileControls } from '../percentileControls.js';
import { getSpeedSettings, setSpeedSettings } from '../speedSettings.js';
/** Barra compacta: CSV estático; processamento exclusivamente no navegador. */
const DATASET_URL = './data/visem_all_trajectory.csv';

export function initUploadControls(onSubmit) {
  const host = document.querySelector('.container');
  const bar = document.createElement('form');
  bar.id = 'dataset-controls';
  bar.setAttribute('aria-label', 'BehaviourRug settings');
  bar.innerHTML = `
    <strong class="app-brand">BehaviourRug</strong>
    <span class="dataset-name" title="Dataset loaded automatically">visem_all_trajectory.csv</span>
    <label for="symbolic-clusters">Symbolic clusters</label>
    <input id="symbolic-clusters" aria-label="Symbolic clusters" type="number" min="1" max="100" step="1" value="5" required>
    <label for="metric-clusters">Metric clusters</label>
    <input id="metric-clusters" aria-label="Metric clusters" type="number" min="1" max="100" step="1" value="5" required>
    <label for="sampling-k">Sampling k</label>
    <input id="sampling-k" aria-label="k amostragem" type="number" min="1" max="1000" step="1" value="10" required>
    <button id="process-dataset" type="submit">Process</button>
    <span id="dataset-status" role="status" aria-live="polite">Preparing dataset…</span>`;
  host.prepend(bar);

  const button = bar.querySelector('#process-dataset');
  const status = bar.querySelector('#dataset-status');
  let datasetFile = null;
  let busy = false;

  async function run() {
    if (!datasetFile || busy) return false;
    const clusters = Number(bar.querySelector('#symbolic-clusters').value);
    const metricClusters = Number(bar.querySelector('#metric-clusters').value);
    const k = Number(bar.querySelector('#sampling-k').value);
    if (!Number.isInteger(clusters) || clusters < 1 || !Number.isInteger(k) || k < 1 || !Number.isInteger(metricClusters) || metricClusters < 1 || metricClusters > 100) {
      status.textContent = 'Enter valid positive integer parameters.';
      return false;
    }
    busy = true;
    button.disabled = true;
    try {
      await onSubmit(datasetFile, {clusters, metricClusters, k, speedSettings: getSpeedSettings()}, message => { status.textContent = message; });
      status.textContent = 'Processing complete';
      return true;
    } catch (error) {
      console.error(error);
      status.textContent = `Error: ${error.message}`;
      return false;
    } finally {
      busy = false;
      button.disabled = false;
    }
  }
  bar.addEventListener('submit', event => {event.preventDefault(); void run();});
  createPercentileControls(bar, async settings => {
    if (busy || !datasetFile) return false;
    const previous = getSpeedSettings();
    setSpeedSettings(settings);
    const success = await run();
    if (!success) setSpeedSettings(previous);
    return success;
  });

  async function loadDefault() {
    try {
      status.textContent = 'Loading CSV…';
      const response = await fetch(DATASET_URL);
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${DATASET_URL}`);
      const blob = await response.blob();
      datasetFile = new File([blob], 'visem_all_trajectory.csv', {type: 'text/csv'});
      status.textContent = 'CSV loaded';
      await run();
    } catch (error) {
      console.error(error);
      status.textContent = `Failed to load CSV: ${error.message}`;
    }
  }
  void loadDefault();
  return {run, bar};
}
