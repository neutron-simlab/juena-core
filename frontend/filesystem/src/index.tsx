import { useEffect, useRef, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import type { FrontendRendererArgs } from '@streamlit/component-v2-lib';
import { Editor } from './Editor';
import { ApiError, basename, cache, dirty, fileUrl, getStore, parent, request, type Data, type Draft, type Page, type Preview, type View } from './state';
import './style.css';

type Bridge = { publish: (view: View) => void; ask: (path: string) => void; authLost: () => void };
function Explorer({ data, publish, ask, authLost }: { data: Data } & Bridge) {
  const store = getStore(data);
  const [view, setView] = useState(store.view);
  const [page, setPage] = useState<Page | null>(null);
  const [filePath, setFilePath] = useState(view.draft?.preview.path || '');
  const [notice, setNotice] = useState('');
  const [fileNotice, setFileNotice] = useState('');
  const [loading, setLoading] = useState(false);
  const [fileLoading, setFileLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [empty, setEmpty] = useState(false);
  const [denied, setDenied] = useState(window.juenaFilesystem!.expired);
  const [reload, setReload] = useState(0);
  const [confirm, setConfirm] = useState<(() => void) | null>(null);
  const [imageUrl, setImageUrl] = useState('');
  const [queued, setQueued] = useState('');
  const alive = useRef(true);
  const update = (change: Partial<View>, editing = false) => {
    const next = { ...store.view, ...change };
    store.view = next;
    setView(next);
    // Drafts stay in the session-scoped browser store across component remounts.
    // Navigation bookkeeping crosses the bridge; keystrokes and file bytes do not.
    if (!editing) publish({ ...next, draft: null });
  };
  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);
  const fail = (error: unknown, target: (message: string) => void) => {
    if (error instanceof DOMException && error.name === 'AbortError') return;
    if (error instanceof ApiError && (error.status === 401 || error.code === 'authentication_required')) {
      const discarded = [...window.juenaFilesystem!.stores.values()].some(item => dirty(item.view.draft));
      window.juenaFilesystem!.stores.clear();
      window.juenaFilesystem!.expired = true;
      store.directories.clear(); store.previews.clear(); store.images.clear();
      store.view.draft = null;
      setPage(null); setFilePath(''); setImageUrl(''); setDenied(true);
      authLost();
      const message = 'Your session expired. Sign in again.' + (discarded ? ' Your unsaved edits were discarded.' : '');
      setNotice(message); setFileNotice(message);
    } else if (error instanceof ApiError && error.code === 'conversation_not_found') {
      store.directories.clear(); store.previews.clear(); store.images.clear();
      update({ draft: null }); setPage(null); setFilePath(''); setEmpty(true);
    } else if (error instanceof ApiError && error.code === 'workspace_busy') {
      target('Files are busy. Loaded views are still available; refresh after execution finishes.');
    } else target(error instanceof Error ? error.message : 'Could not load files. Try Refresh.');
  };
  const guard = (action: () => void) => {
    if (dirty(store.view.draft)) setConfirm(() => action);
    else action();
  };
  const navigate = (cwd: string, history = [...store.view.history, store.view.cwd]) => guard(() => {
    setFilePath(''); setFileNotice(''); setQueued('');
    update({ cwd, history: history.slice(-50), offset: 0, draft: null });
  });
  useEffect(() => {
    if (store.generation !== data.refreshGeneration) {
      store.generation = data.refreshGeneration;
      setReload(value => value + 1);
    }
  }, [data.refreshGeneration]);
  useEffect(() => {
    if (denied || view.collapsed) return;
    const controller = new AbortController();
    const key = `${view.cwd}:${view.offset}`;
    const cached = store.directories.get(key);
    setPage(cached?.value || null); setLoading(true); setNotice('');
    (async () => {
      try {
        // A missing chat is an unsent conversation, not a filesystem error.
        const exists = await fetch(`${data.apiUrl}/chats/${encodeURIComponent(data.threadId)}?include_messages=false`,
          { credentials: 'include', cache: 'no-store', signal: controller.signal });
        if (controller.signal.aborted) return;
        if (exists.status === 404) {
          store.directories.clear(); store.previews.clear(); store.images.clear();
          setEmpty(true); setPage(null); setFilePath(''); update({ draft: null }); return;
        }
        if (!exists.ok) {
          const detail = (await exists.json().catch(() => ({}))).detail;
          throw new ApiError(exists.status, detail?.code || '', detail?.message || 'Could not access this conversation.');
        }
        setEmpty(false);
        const params = new URLSearchParams({ path: view.cwd, hidden: 'true', offset: String(view.offset) });
        const result: Page = await (await request(data, `${fileUrl(data)}?${params}`, { signal: controller.signal })).json();
        if (controller.signal.aborted) return;
        cache(store.directories, key, result, 20); setPage(result);
      } catch (error) { if (!controller.signal.aborted) fail(error, setNotice); }
      finally { if (!controller.signal.aborted) setLoading(false); }
    })();
    return () => controller.abort();
  }, [view.cwd, view.offset, view.collapsed, reload, denied]);
  useEffect(() => {
    if (!filePath || denied || dirty(store.view.draft)) return;
    const controller = new AbortController();
    const cached = store.previews.get(filePath);
    if (cached) update({ draft: { preview: cached.value, text: cached.value.text } });
    setFileNotice(''); setFileLoading(true);
    (async () => {
      try {
        const preview: Preview = await (await request(data, `${fileUrl(data, '/file')}?${new URLSearchParams({ path: filePath })}`, { signal: controller.signal })).json();
        if (controller.signal.aborted) return;
        cache(store.previews, filePath, preview, 8);
        // Typing can begin while a refresh is in flight. Never replace it.
        if (!dirty(store.view.draft)) update({ draft: { preview, text: preview.text } });
      } catch (error) { if (!controller.signal.aborted) fail(error, setFileNotice); }
      finally { if (!controller.signal.aborted) setFileLoading(false); }
    })();
    return () => controller.abort();
  }, [filePath, reload, denied]);
  const preview = view.draft?.preview;
  useEffect(() => {
    setImageUrl('');
    if (!preview || preview.is_text || !preview.path.toLowerCase().endsWith('.png') || preview.size > 5 * 1024 * 1024) return;
    const controller = new AbortController();
    let url = '';
    const cached = store.images.get(preview.path);
    if (cached) { url = URL.createObjectURL(cached.value); setImageUrl(url); }
    request(data, `${fileUrl(data, '/download')}?${new URLSearchParams({ path: preview.path })}`, { signal: controller.signal })
      .then(response => response.blob()).then(blob => {
        if (!controller.signal.aborted) {
          cache(store.images, preview.path, blob, 4);
          if (url) URL.revokeObjectURL(url);
          url = URL.createObjectURL(blob); setImageUrl(url);
        }
      }).catch(error => { if (!controller.signal.aborted) fail(error, setFileNotice); });
    return () => { controller.abort(); if (url) URL.revokeObjectURL(url); };
  }, [preview?.path, preview?.revision, reload]);
  const save = async () => {
    const draft = store.view.draft;
    if (!draft || !draft.preview.editable || !draft.preview.revision || saving) return false;
    const text = draft.text;
    if (new TextEncoder().encode(text).length > 1024 * 1024) { setFileNotice('Editing is limited to 1 MiB.'); return false; }
    setSaving(true); setFileNotice('');
    try {
      const result = await (await request(data, fileUrl(data, '/file'), { method: 'PUT',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: draft.preview.path, text, revision: draft.preview.revision }) })).json();
      if (!alive.current || window.juenaFilesystem!.expired) return false;
      const updated = { ...draft.preview, text, size: new TextEncoder().encode(text).length, revision: result.revision };
      cache(store.previews, updated.path, updated, 8);
      update({ draft: { preview: updated, text: store.view.draft!.text } });
      setFileNotice('Saved. The agent will read this version.'); return !dirty(store.view.draft);
    } catch (error) { if (alive.current) fail(error, setFileNotice); return false; }
    finally { if (alive.current) setSaving(false); }
  };
  const download = async () => {
    try {
      const blob = await (await request(data, `${fileUrl(data, '/download')}?${new URLSearchParams({ path: filePath })}`)).blob();
      if (!alive.current) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = basename(filePath); a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { if (alive.current) fail(error, setFileNotice); }
  };
  const openFile = (path: string) => guard(() => {
    const cached = store.previews.get(path);
    update({ draft: cached ? { preview: cached.value, text: cached.value.text } : null });
    setFilePath(path); setQueued(''); setFileNotice('');
  });
  const paths = view.cwd.split('/').filter(Boolean);
  return <section className="explorer" aria-label="Conversation files" data-conversation={data.threadId}>
    <button className="heading" aria-expanded={!view.collapsed} onClick={() => update({ collapsed: !view.collapsed })}>
      <span>Files</span><span aria-hidden="true">{view.collapsed ? '▸' : '▾'}</span>
    </button>
    {!view.collapsed && <>
      <p className="caption">The same files and paths your agent uses.</p>
      {denied ? <p role="alert">{notice || fileNotice || 'Your session expired. Sign in again.'}</p> : <>
        <nav aria-label="Folder path" className="breadcrumbs">
          <button onClick={() => navigate('/')}>/</button>
          {paths.map((part, index) => <span key={index}><span aria-hidden="true">{index ? ' / ' : ' '}</span><button onClick={() => navigate('/' + paths.slice(0, index + 1).join('/'))}>{part}</button></span>)}
        </nav>
        <div className="toolbar">
          <button disabled={!view.history.length} onClick={() => navigate(view.history.at(-1)!, view.history.slice(0, -1))}>← Back</button>
          <button disabled={view.cwd === '/'} onClick={() => navigate(parent(view.cwd))}>↑ Up</button>
          <button onClick={() => setReload(value => value + 1)}>Refresh</button>
        </div>
        {view.cwd.startsWith('/findings') && <p className="caption">Specialist observations and intermediate conclusions, not verified execution results.</p>}
        {notice && <p className="notice" role="status">{notice}</p>}
        {empty ? <p className="notice">Send your first message to create this conversation's workspace.</p> : <>
          <div className="directory" aria-label="Folder contents" aria-busy={loading}>
            {loading && <p className="caption" role="status">Loading folder…</p>}
            {page?.entries.map(entry => <button className="entry" key={entry.path} title={entry.path} onClick={() => entry.is_dir ? navigate(entry.path) : openFile(entry.path)}>
              <span aria-hidden="true">{entry.is_dir ? '📁' : '📄'}</span><span className="filename">{basename(entry.path)}</span>
              <small className="metadata">{entry.is_dir ? 'Folder' : `${entry.size.toLocaleString()} B`}{entry.modified_at && <span>{new Date(entry.modified_at).toLocaleDateString()}</span>}</small>
            </button>)}
            {!loading && page && page.entries.length === 0 && <p className="caption">This folder is empty.</p>}
          </div>
          <div className="toolbar">
            {view.offset > 0 && <button onClick={() => update({ offset: Math.max(0, view.offset - 100) })}>Previous</button>}
            {page?.next_offset != null && <button onClick={() => update({ offset: page.next_offset! })}>Next</button>}
          </div>
          {page && <p className="caption">Last loaded {store.directories.get(`${view.cwd}:${view.offset}`)?.time}</p>}
        </>}
        {filePath && <section className="preview" aria-label="File preview">
          <div className="toolbar"><strong className="filename">{basename(filePath)}</strong><button onClick={() => guard(() => { setFilePath(''); update({ draft: null }); })}>Close file</button></div>
          <code className="file-path">{filePath}</code>
          {fileLoading && <p className="caption" role="status">Loading file…</p>}
          {preview && <>
            <p className="caption">{preview.size.toLocaleString()} bytes{!preview.editable && ' · Read-only'}</p>
            {preview.is_text ? <Editor path={preview.path} text={view.draft!.text} readOnly={!preview.editable || saving} onChange={text => {
              if (store.view.draft && text !== store.view.draft.text) update({ draft: { ...store.view.draft, text } }, true);
            }} /> : imageUrl ? <img className="image" src={imageUrl} alt={basename(filePath)} /> : <p className="caption">Download this file to view it.</p>}
            {preview.truncated && <p className="caption">Preview truncated. Editing is limited to 1 MiB.</p>}
            {dirty(view.draft) && <p className="caption">Unsaved changes</p>}
          </>}
          {fileNotice && <p className="notice" role="status">{fileNotice}</p>}
          <div className="toolbar">
            {preview?.editable && <button className="primary" disabled={saving || !dirty(view.draft)} onClick={() => void save()}>{saving ? 'Saving…' : 'Save'}</button>}
            {dirty(view.draft) && <button disabled={saving} onClick={() => update({ draft: { ...view.draft!, text: view.draft!.preview.text } })}>Discard</button>}
            <button disabled={saving} onClick={() => guard(() => { update({ draft: null }); store.previews.delete(filePath); setReload(value => value + 1); })}>Reload file</button>
            <button onClick={() => void download()}>Download</button>
            <button disabled={!preview || dirty(view.draft) || saving} onClick={() => { ask(filePath); setQueued('Path queued for your next message.'); }}>Ask about this file</button>
          </div>
          {queued && <p className="notice" role="status">{queued}</p>}
        </section>}
      </>}
    </>}
    {confirm && <div className="confirm" role="dialog" aria-label="Unsaved file changes">
      <p>Save or discard your changes before switching files.</p>
      <div className="toolbar">
        <button disabled={saving} onClick={async () => { if (await save()) { const action = confirm; setConfirm(null); action(); } }}>Save and continue</button>
        <button disabled={saving} onClick={() => { update({ draft: null }); const action = confirm; setConfirm(null); action(); }}>Discard and continue</button>
        <button onClick={() => setConfirm(null)}>Stay here</button>
      </div>
    </div>}
  </section>;
}

const mounts = new WeakMap<object, { root: Root; host: HTMLDivElement }>();
export default function render({ data, parentElement, setStateValue, setTriggerValue }: FrontendRendererArgs) {
  const props = data as Data;
  let mount = mounts.get(parentElement);
  if (!mount) {
    const host = document.createElement('div');
    parentElement.appendChild(host);
    mount = { root: createRoot(host), host };
    mounts.set(parentElement, mount);
  }
  const { root, host } = mount;
  root.render(<Explorer key={`${props.sessionScope}:${props.threadId}`} data={props}
    publish={view => setStateValue('view', view)}
    authLost={() => setTriggerValue('auth_lost', true)}
    ask={path => setStateValue('ask', { id: crypto.randomUUID(), thread_id: props.threadId, path })} />);
  return () => { root.unmount(); host.remove(); mounts.delete(parentElement); };
}
