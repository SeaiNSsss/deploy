/* Minimal ONNX runtime web loader & runner */
(function () {
  const modelFile = document.getElementById('modelFile');
  const inputShapeEl = document.getElementById('inputShape');
  const inputDataEl = document.getElementById('inputData');
  const inputNameEl = document.getElementById('inputName');
  const dtypeEl = document.getElementById('dtype');
  const runBtn = document.getElementById('runBtn');
  const statusEl = document.getElementById('status');
  const outputEl = document.getElementById('output');

  /** @type {ort.InferenceSession | null} */
  let session = null;
  /** @type {string | null} */
  let defaultInputName = null;

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function parseShape(shapeStr) {
    if (!shapeStr) return null;
    const parts = shapeStr.split(',').map((s) => s.trim()).filter(Boolean);
    if (!parts.length) return null;
    return parts.map((p) => {
      const n = Number(p);
      if (!Number.isFinite(n) || n <= 0) throw new Error('shape ต้องเป็นจำนวนบวก เช่น 1,3,224,224');
      return n;
    });
  }

  function parseArray(dataStr) {
    if (!dataStr) return [];
    let arr = [];
    try {
      // try parse JSON array
      const maybe = JSON.parse(dataStr);
      if (Array.isArray(maybe)) return maybe;
    } catch (_) {}
    // fallback to comma-separated
    arr = dataStr.split(',').map((s) => s.trim()).filter((s) => s.length > 0).map(Number);
    if (arr.some((v) => !Number.isFinite(v))) {
      throw new Error('ข้อมูลอินพุตต้องเป็นตัวเลข คั่นด้วยคอมมา หรือ JSON array');
    }
    return arr;
  }

  async function loadModelFromFile(file) {
    setStatus('กำลังโหลดโมเดล...');
    const buf = await file.arrayBuffer();
    // Prefer WebGL if available; fallback to wasm
    /** @type {ort.InferenceSession.SessionOptions} */
    const options = { executionProviders: ['webgl', 'wasm'] };
    session = await ort.InferenceSession.create(buf, options);
    const inputs = session.inputNames;
    defaultInputName = inputs && inputs.length ? inputs[0] : null;
    setStatus(`โหลดโมเดลสำเร็จ • inputs: ${inputs.join(', ')}`);
  }

  function makeTypedArray(dtype, data) {
    switch (dtype) {
      case 'float32':
        return new Float32Array(data);
      case 'int32':
        return new Int32Array(data);
      default:
        throw new Error('ไม่รองรับชนิดข้อมูลนี้');
    }
  }

  function product(nums) { return nums.reduce((a, b) => a * b, 1); }

  async function runInference() {
    try {
      if (!modelFile.files || !modelFile.files[0]) {
        alert('กรุณาเลือกไฟล์โมเดล .onnx ก่อน');
        return;
      }
      if (!session) {
        await loadModelFromFile(modelFile.files[0]);
      }

      const dtype = dtypeEl.value;
      const shape = parseShape(inputShapeEl.value);
      const data = parseArray(inputDataEl.value);
      const inputName = inputNameEl.value && inputNameEl.value.trim().length ? inputNameEl.value.trim() : (defaultInputName || session.inputNames[0]);

      if (!shape) throw new Error('กรุณาระบุ shape ของอินพุต เช่น 1,3,224,224');
      const needed = product(shape);
      if (data.length && data.length !== needed) {
        throw new Error(`จำนวนข้อมูล (${data.length}) ต้องเท่ากับผลคูณ shape (${needed})`);
      }

      const feedData = data.length ? data : new Array(needed).fill(0);
      const typed = makeTypedArray(dtype, feedData);

      const tensor = new ort.Tensor(dtype, typed, shape);
      const feeds = { [inputName]: tensor };

      setStatus('กำลังรันโมเดล...');
      const start = performance.now();
      const results = await session.run(feeds);
      const ms = (performance.now() - start).toFixed(1);
      setStatus(`รันสำเร็จใน ${ms} ms`);

      const outNames = Object.keys(results);
      const summary = outNames.map((name) => {
        const t = results[name];
        return {
          name,
          dtype: t.type,
          dims: t.dims,
          size: t.data.length,
          sample: Array.from(t.data.slice(0, Math.min(16, t.data.length))),
        };
      });
      outputEl.textContent = JSON.stringify(summary, null, 2);
    } catch (err) {
      console.error(err);
      setStatus('เกิดข้อผิดพลาด');
      outputEl.textContent = (err && err.message) ? err.message : String(err);
    }
  }

  runBtn?.addEventListener('click', runInference);
})();


