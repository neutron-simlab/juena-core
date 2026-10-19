import { useEffect, useRef } from 'react';
import ace from 'ace-builds';
import 'ace-builds/src-noconflict/mode-python';
import 'ace-builds/src-noconflict/mode-json';
import 'ace-builds/src-noconflict/mode-markdown';
import 'ace-builds/src-noconflict/mode-javascript';
import 'ace-builds/src-noconflict/mode-typescript';
import 'ace-builds/src-noconflict/mode-sh';
import 'ace-builds/src-noconflict/mode-yaml';
import 'ace-builds/src-noconflict/mode-html';
import 'ace-builds/src-noconflict/mode-css';
import 'ace-builds/src-noconflict/mode-text';
import 'ace-builds/src-noconflict/theme-textmate';

const languages: Record<string, string> = { py: 'python', json: 'json', ipynb: 'json', md: 'markdown', js: 'javascript', ts: 'typescript', sh: 'sh', yaml: 'yaml', yml: 'yaml', html: 'html', css: 'css' };
export function Editor({ text, path, readOnly, onChange }: { text: string; path: string; readOnly: boolean; onChange: (text: string) => void }) {
  const host = useRef<HTMLDivElement>(null);
  const editor = useRef<ace.Ace.Editor | null>(null);
  const changed = useRef(onChange);
  changed.current = onChange;
  useEffect(() => {
    const instance = ace.edit(host.current!);
    instance.renderer.attachToShadowRoot();
    editor.current = instance;
    instance.setTheme('ace/theme/textmate');
    instance.setOptions({ showPrintMargin: false, fontSize: 13, tabSize: 4, useSoftTabs: true,
      showLineNumbers: true, highlightSelectedWord: true, useWorker: false, wrap: true });
    instance.session.on('change', () => changed.current(instance.getValue()));
    const resize = new ResizeObserver(() => instance.resize());
    resize.observe(host.current!);
    return () => { resize.disconnect(); instance.destroy(); editor.current = null; };
  }, []);
  useEffect(() => {
    const instance = editor.current!;
    instance.setReadOnly(readOnly);
    instance.session.setMode(`ace/mode/${languages[path.split('.').pop()!.toLowerCase()] || 'text'}`);
    if (instance.getValue() !== text) instance.setValue(text, -1);
  }, [text, path, readOnly]);
  return <div className="editor" ref={host} aria-label={`Editor for ${path}`} />;
}
