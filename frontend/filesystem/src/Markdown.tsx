import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Files come from the agent's sandbox. react-markdown never renders raw HTML
// and blanks `javascript:` URLs by default. Links open in a new tab, so a click
// does not navigate the Streamlit page away from the conversation.
export function MarkdownPreview({ text, path }: { text: string; path: string }) {
  return <div className="markdown" aria-label={`Preview of ${path}`}>
    <Markdown remarkPlugins={[remarkGfm]} components={{
      a: ({ node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
    }}>{text}</Markdown>
  </div>;
}
