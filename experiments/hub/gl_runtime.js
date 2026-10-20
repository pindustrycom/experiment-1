/* Shared WebGL runtime for the 3D pages (exp4, exp6, exp6 live).

   Pages load it with <script src="../hub/gl_runtime.js"></script>, which works when the page is opened from the
   repository; build_hub.py inlines it into the published copy, because lock_site.py encrypts every file that is not a page.

   Why it exists: Chrome turns WebGL off for a whole site after its pages lose the GPU context a few times, and then
   only a full browser restart brings it back. Each page used to keep its own context alive forever: it rendered every
   frame even when the 3D view was scrolled away, and pages kept in the back/forward cache held their context while the
   next 3D page opened another. This runtime
     - renders only while the tab is visible and the canvas is on screen,
     - caps the pixel ratio so a large canvas on a retina screen does not exhaust GPU memory,
     - always releases the context when the page is left (back/forward cache included) and reloads on return,
     - stops on a lost context and explains what happened instead of showing a blank canvas.

   const gl = LabGL.start(canvas, { antialias: true });   // null when WebGL is unavailable (a message is shown)
   gl.renderer                                             // the THREE.WebGLRenderer
   gl.loop(now => { ...; gl.renderer.render(scene, camera); });   // called once per frame while visible; do not reschedule
   gl.lost                                                 // true after the context was lost
   LabGL.dispose(object3d)                                 // free the geometries, materials and textures under an object
*/
(function () {
  "use strict";
  if (window.LabGL) return;
  const BLOCKED = "<b>The 3D view could not start.</b> Your browser has turned off WebGL for this site, usually after a page " +
    "lost its graphics context. Quit the browser completely (every window) and open this page again. If it still fails, " +
    "check that hardware acceleration is on in the browser settings.";
  const LOST = "<b>The 3D view lost its graphics context.</b> It comes back on its own if the browser restores it; " +
    "otherwise reload the page.";

  function fail(canvas, msg) {
    const next = canvas.nextElementSibling;
    if (next && next.classList.contains("gl-fail")) { next.innerHTML = msg; return; }
    const d = document.createElement("div"); d.className = "gl-fail"; d.setAttribute("role", "status"); d.innerHTML = msg;
    canvas.style.visibility = "hidden"; canvas.insertAdjacentElement("afterend", d);
  }

  function start(canvas, opts) {
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer(Object.assign({ canvas, powerPreference: "default", preserveDrawingBuffer: false }, opts));
    } catch (e) { fail(canvas, BLOCKED); return null; }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));

    const gl = { renderer, lost: false, loop };
    let fn = null, raf = 0, onScreen = true;
    const active = () => fn && !gl.lost && onScreen && document.visibilityState === "visible";
    const tick = now => { raf = 0; if (!active()) return; fn(now); if (!raf && active()) raf = requestAnimationFrame(tick); };
    const wake = () => { if (!raf && active()) raf = requestAnimationFrame(tick); };
    function loop(f) { fn = f; wake(); }

    if ("IntersectionObserver" in window) {
      new IntersectionObserver(es => { onScreen = es[es.length - 1].isIntersecting; wake(); }, { rootMargin: "100px" }).observe(canvas);
    }
    document.addEventListener("visibilitychange", wake);

    canvas.addEventListener("webglcontextlost", e => { e.preventDefault(); gl.lost = true; fail(canvas, LOST); });
    canvas.addEventListener("webglcontextrestored", () => location.reload());
    // Release the GPU on every exit. A page kept in the back/forward cache would otherwise hold its context while the next
    // 3D page opens another; it is rebuilt by a reload when it is shown again.
    window.addEventListener("pagehide", () => { fn = null; if (raf) cancelAnimationFrame(raf); renderer.dispose(); renderer.forceContextLoss(); });
    window.addEventListener("pageshow", e => { if (e.persisted) location.reload(); });
    return gl;
  }

  function dispose(root) {
    root.traverse(o => {
      if (o.geometry) o.geometry.dispose();
      for (const m of [].concat(o.material || [])) {
        for (const k in m) if (m[k] && m[k].isTexture) m[k].dispose();
        m.dispose();
      }
    });
  }

  window.LabGL = { start, dispose };
})();
