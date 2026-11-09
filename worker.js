// Runs the Python painting code (pure numpy) in the browser with Pyodide, off
// the main thread so the page stays responsive while a stamp is painting.
importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js');

const ready = (async () => {
  const py = await loadPyodide();
  await py.loadPackage('numpy');
  for (const f of ['stamp_core_v3.py', 'stamp_core_v4.py']) {
    const r = await fetch('py/' + f);
    if (!r.ok) throw new Error('Could not load ' + f);
    py.FS.writeFile(f, await r.text());
  }
  py.runPython(`
import sys
sys.path.insert(0, '.')
import numpy as np
import stamp_core_v3 as sc
import stamp_core_v4 as poster

class _Inline:
    """The browser has no threads here, so the knife styles' thread pool
    runs its work in order instead."""
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def map(self, fn, *its): return list(map(fn, *its))
sc.ThreadPoolExecutor = _Inline

def paint(buf, w, h, q, report):
    q = q.to_py()
    rgb = np.ascontiguousarray(np.asarray(buf.to_py(), np.uint8).reshape(h, w, 4)[..., :3])
    seed = int(q['seed'])
    if q['style'] == 'poster':
        art = poster.stylize(rgb, seed=seed, progress=report,
                             colors=int(q['colors']), detail=float(q['detail']),
                             simplify=float(q['simplify']), lines=float(q['lines']),
                             warmth=float(q['warmth']), saturation=float(q['saturation']),
                             grain=float(q['grain']))
    else:
        art = sc.stylize(rgb, style=q['style'], seed=seed, progress=report,
                         brush=int(q['brush']), detail=float(q['detail']),
                         texture=float(q['texture']))
    return np.dstack([art, np.full((h, w), 255, np.uint8)]).tobytes()
`);
  return py.globals.get('paint');
})();

ready.then(() => postMessage({ type: 'ready' }),
           e => postMessage({ type: 'error', error: 'Could not start the painter: ' + e.message }));

onmessage = async ({ data }) => {
  try {
    const paint = await ready;
    const report = f => postMessage({ type: 'progress', value: f });
    const out = paint(data.pixels, data.w, data.h, data.q, report);
    const bytes = out.toJs();
    out.destroy();
    postMessage({ type: 'done', pixels: bytes, w: data.w, h: data.h }, [bytes.buffer]);
  } catch (e) {
    postMessage({ type: 'error', error: 'Painting failed: ' + e.message });
  }
};
