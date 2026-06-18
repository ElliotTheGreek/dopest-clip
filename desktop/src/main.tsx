import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

// Surface any fatal boot error into the DOM (production has no Vite overlay; a silent
// throw leaves a blank window). Keep this — it's cheap insurance, not just for QA.
function showFatal(msg: string): void {
  const el = document.getElementById("root");
  if (el) {
    el.innerHTML =
      '<pre id="fatal-error" style="color:#ff8080;background:#1a1a1e;padding:16px;' +
      'white-space:pre-wrap;font-family:monospace;font-size:13px">' +
      msg.replace(/[<>&]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[c] as string)) +
      "</pre>";
  }
}

window.addEventListener("error", (e) => showFatal("Uncaught error:\n" + (e.error?.stack || e.message)));
window.addEventListener("unhandledrejection", (e) =>
  showFatal("Unhandled rejection:\n" + String((e.reason && (e.reason.stack || e.reason)) ?? e.reason)),
);

try {
  const el = document.getElementById("root");
  if (!el) throw new Error("no #root element");
  createRoot(el).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
} catch (err) {
  showFatal("Mount failed:\n" + String((err as Error)?.stack || err));
}
