import { ChatInterface } from "./components/ChatInterface";
import "./App.css";

function App() {
  return (
    <div
      className="min-h-screen flex items-center justify-center"
      style={{ background: "var(--gray-00)", color: "var(--gray-11)" }}
    >
      <div className="max-w-4xl w-full flex flex-col">
        <header
          className="border-b p-4"
          style={{ borderColor: "var(--gray-06)" }}
        >
          <h1 className="text-2xl font-normal">eme</h1>
        </header>
        <main className="flex-1 overflow-hidden">
          <ChatInterface />
        </main>
      </div>
    </div>
  );
}

export default App;
