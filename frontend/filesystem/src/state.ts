export type Entry = { path: string; is_dir: boolean; size: number; modified_at?: string; writable: boolean };
export type Page = { path: string; entries: Entry[]; next_offset: number | null };
export type Preview = { path: string; text: string; size: number; modified_at?: string; editable: boolean; is_text: boolean; truncated: boolean; revision: string | null };
export type Draft = { preview: Preview; text: string };
export type View = { cwd: string; history: string[]; offset: number; collapsed: boolean; draft: Draft | null };
export type Data = { apiUrl: string; threadId: string; sessionScope: string; refreshGeneration: number; initialView?: View };
export type Cached<T> = { value: T; time: string };
export type Store = { view: View; directories: Map<string, Cached<Page>>; previews: Map<string, Cached<Preview>>; images: Map<string, Cached<Blob>>; generation: number };

declare global { interface Window { juenaFilesystem?: { scope: string; stores: Map<string, Store>; expired: boolean } } }
export function getStore(data: Data): Store {
  if (window.juenaFilesystem?.scope !== data.sessionScope) {
    window.juenaFilesystem = { scope: data.sessionScope, stores: new Map(), expired: false };
  }
  const stores = window.juenaFilesystem.stores;
  let store = stores.get(data.threadId);
  if (!store) {
    store = { view: !window.juenaFilesystem.expired && data.initialView || {
      cwd: '/', history: [], offset: 0, collapsed: false, draft: null,
    }, directories: new Map(), previews: new Map(), images: new Map(), generation: data.refreshGeneration };
    stores.set(data.threadId, store);
  }
  return store;
}
export function cache<T>(map: Map<string, Cached<T>>, key: string, value: T, limit: number) {
  map.delete(key);
  map.set(key, { value, time: new Date().toLocaleTimeString() });
  while (map.size > limit) map.delete(map.keys().next().value!);
}
export function dirty(draft: Draft | null) { return !!draft && draft.text !== draft.preview.text; }
export function parent(path: string) { return path.slice(0, path.lastIndexOf('/')) || '/'; }
export function basename(path: string) { return path.slice(path.lastIndexOf('/') + 1); }
export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) { super(message); }
}
export async function request(data: Data, suffix: string, init: RequestInit = {}) {
  const response = await fetch(`${data.apiUrl}${suffix}`, { credentials: 'include', cache: 'no-store', ...init });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = body.detail;
    throw new ApiError(response.status, detail?.code || '', typeof detail === 'string' ? detail : detail?.message || `Request failed (${response.status})`);
  }
  return response;
}
export function fileUrl(data: Data, suffix = '') { return `/threads/${encodeURIComponent(data.threadId)}/filesystem${suffix}`; }
