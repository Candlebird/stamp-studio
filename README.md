# Stamp Studio

Arrange photos and turn them into painted postage stamps, in the browser.
Served at https://candlebird.github.io/stamp-studio/.

This is the web version of `postage-stamp-gimp-plugin/stamp-studio`. Instead of a
local Python server, the painting code (`py/stamp_core_v3.py`, `py/stamp_core_v4.py`,
pure numpy) runs in a Web Worker with [Pyodide](https://pyodide.org). Photos never
leave the device.

- `index.html`: the app (arranging, text and border are plain canvas)
- `worker.js`: loads Pyodide and numpy and runs the painting code
- `py/`: copies of the painting code. One change from the originals: in
  `stamp_core_v3.py`, `_knife_footprints` casts its repeat counts to `np.intp`,
  because numpy in the browser is 32-bit.
- `photos/`, `thumbs/`: the stamp photos (longest side 2000 px) and 240 px
  thumbnails, listed in `photos/photos.json`. Credits are in `photos/CREDITS.md`.

To run it locally, serve the folder (for example `python -m http.server`) and open
it. Opening `index.html` directly from disk won't work, because workers and
`fetch` need a web server.
