import { useEffect, useState } from "react";
import { useEditor } from "./state/useEditor";
import { Projects } from "./components/Projects";
import { Transcript } from "./components/Transcript";
import { Timeline } from "./components/Timeline";
import { Inspector } from "./components/Inspector";
import { Preview } from "./components/Preview";
import { Providers } from "./components/Providers";

type Tab = "edit" | "providers";

export function App() {
  const ed = useEditor();
  const [tab, setTab] = useState<Tab>("edit");

  // auto-dismiss toast
  useEffect(() => {
    if (!ed.toast) return;
    const t = setTimeout(() => ed.setToast(null), 4000);
    return () => clearTimeout(t);
  }, [ed.toast, ed]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">dopest-clip</div>
        <nav className="tabs">
          <button className={tab === "edit" ? "active" : ""} onClick={() => setTab("edit")}>Edit</button>
          <button className={tab === "providers" ? "active" : ""} onClick={() => setTab("providers")}>Providers</button>
        </nav>
        <div className="conn">
          <span className={"dot " + (ed.conn.connected ? "configured" : "unconfigured")} />
          {ed.conn.connected ? `sidecar v${ed.conn.version ?? "?"}` : "sidecar offline"}
        </div>
      </header>

      {!ed.conn.connected && (
        <div className="banner error">
          Cannot reach the dopest-clip sidecar. {ed.conn.error ? `(${ed.conn.error}) ` : ""}
          Launch it with: <code>python -m dopest_clip --serve --port 8765</code> (or let <code>npm run dev</code> spawn it).
        </div>
      )}

      {tab === "edit" ? (
        <div className="layout">
          <aside className="col col-left">
            <Projects ed={ed} />
          </aside>
          <section className="col col-mid">
            <Transcript ed={ed} />
            <Timeline ed={ed} />
          </section>
          <aside className="col col-right">
            <Inspector ed={ed} />
            <Preview ed={ed} />
          </aside>
        </div>
      ) : (
        <div className="layout single">
          <Providers ed={ed} />
        </div>
      )}

      <footer className="statusbar">
        {ed.busy && <span className="busy">⏳ {ed.busy}…</span>}
        {ed.error && <span className="err" onClick={() => ed.setError(null)} title="click to dismiss">⚠ {ed.error}</span>}
        {ed.toast && <span className="toast" onClick={() => ed.setToast(null)}>{ed.toast}</span>}
        {!ed.busy && !ed.error && !ed.toast && <span className="idle">ready</span>}
      </footer>
    </div>
  );
}
