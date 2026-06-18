import { useCallback, useEffect, useState } from "react";
import type { EditorApi } from "../state/useEditor";
import * as sc from "../api/sidecar";
import type { ListProvidersResult } from "../api/types";

// Renders list_providers(): per capability, the providers with configured/active state,
// and a dropdown to set_provider().
export function Providers({ ed }: { ed: EditorApi }) {
  const [data, setData] = useState<ListProvidersResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    setErr(null);
    try {
      setData(await sc.listProviders());
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    if (ed.conn.connected) load();
  }, [ed.conn.connected, load]);

  async function choose(capability: string, provider: string) {
    setErr(null);
    try {
      await sc.setProvider(capability, provider);
      await load();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="panel providers-panel">
      <div className="panel-head">
        <h2>Providers</h2>
        <button className="ghost" onClick={load}>&#x21bb;</button>
      </div>
      {err && <div className="error-box">{err}</div>}
      {!data && !err && <p className="empty">Loading…</p>}
      {data && Object.keys(data).length === 0 && <p className="empty">No capabilities reported.</p>}
      <div className="cap-grid">
        {data &&
          Object.entries(data).map(([cap, info]) => {
            const providers = info.providers ?? {};
            const active = info.active ?? Object.entries(providers).find(([, p]) => p.active)?.[0] ?? "";
            return (
              <div className="cap" key={cap}>
                <div className="cap-head">
                  <strong>{cap}</strong>
                  <select value={active} onChange={(e) => choose(cap, e.target.value)}>
                    {Object.keys(providers).map((name) => (
                      <option key={name} value={name}>{name}</option>
                    ))}
                  </select>
                </div>
                <ul className="prov-list">
                  {Object.entries(providers).map(([name, p]) => (
                    <li key={name} className={p.active ? "active" : ""}>
                      <span className={"dot " + (p.configured ? "configured" : "unconfigured")} />
                      <span className="prov-name">{name}</span>
                      <span className="prov-detail">{p.configured ? (p.detail ?? "configured") : "not configured"}</span>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
      </div>
    </div>
  );
}
