import { useState, useRef, useEffect } from 'react';
import { apiService, type Message } from '../services/api';

interface ChatInterfaceProps {
  className?: string;
}

export function ChatInterface({ className = '' }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue.trim(),
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const stream = await apiService.sendMessage(userMessage.text);
      const reader = stream.getReader();
      const decoder = new TextDecoder();

      let botResponse = '';
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: '',
        sender: 'bot',
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, botMessage]);
      setIsLoading(false); // Hide "thinking" UI
      setIsStreaming(true); // Show streaming state

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        botResponse += chunk;

        // Update message with accumulated text
        setMessages((prev) =>
          prev.map((msg) => (msg.id === botMessage.id ? { ...msg, text: botResponse } : msg)),
        );
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'bot',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
    }
  };

  return (
    <div className={`flex flex-col h-full ${className}`}>
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center py-8" style={{ color: 'var(--gray-09)' }}>
            <p>Hi! Ask me something like "should I take CS214 and CS211 at the same time?"</p>
          </div>
        )}
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className="max-w-[80%] p-3 rounded-lg"
              style={{
                backgroundColor: 'var(--gray-06)',
                color: 'var(--gray-11)',
                border: message.sender === 'bot' ? `1px solid var(--gray-09)` : 'none',
              }}
            >
              <p className="text-sm leading-relaxed whitespace-pre-wrap text-left">
                {message.text}
              </p>
            </div>
          </div>
        ))}
        {isLoading && !isStreaming && (
          <div className="flex justify-start">
            <div
              className="p-3 rounded-lg"
              style={{
                backgroundColor: 'var(--gray-06)',
                color: 'var(--gray-11)',
                border: '1px solid var(--gray-09)',
              }}
            >
              <div className="flex items-center space-x-2">
                <div className="animate-pulse">eme is thinking</div>
                <div className="flex space-x-1">
                  <div
                    className="w-1 h-1 rounded-full animate-bounce"
                    style={{ backgroundColor: 'var(--gray-09)' }}
                  ></div>
                  <div
                    className="w-1 h-1 rounded-full animate-bounce"
                    style={{
                      backgroundColor: 'var(--gray-09)',
                      animationDelay: '0.1s',
                    }}
                  ></div>
                  <div
                    className="w-1 h-1 rounded-full animate-bounce"
                    style={{
                      backgroundColor: 'var(--gray-09)',
                      animationDelay: '0.2s',
                    }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form
        onSubmit={handleSubmit}
        className="p-4 border-t"
        style={{ borderColor: 'var(--gray-06)' }}
      >
        <div className="flex space-x-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask eme something..."
            disabled={isLoading}
            className="flex-1 px-3 py-2 rounded focus:outline-none disabled:opacity-50"
            style={{
              backgroundColor: 'var(--gray-00)',
              border: '1px solid var(--gray-06)',
              color: 'var(--gray-11)',
            }}
            onFocus={(e) => (e.target.style.borderColor = 'var(--gray-09)')}
            onBlur={(e) => (e.target.style.borderColor = 'var(--gray-06)')}
          />
          <button
            type="submit"
            disabled={!inputValue.trim() || isLoading}
            className="px-4 py-2 rounded disabled:opacity-50 disabled:cursor-not-allowed"
            style={{
              backgroundColor: 'var(--gray-06)',
              color: 'var(--gray-11)',
              border: '1px solid var(--gray-09)',
            }}
            onMouseEnter={(e) =>
              ((e.target as HTMLButtonElement).style.backgroundColor = 'var(--gray-09)')
            }
            onMouseLeave={(e) =>
              ((e.target as HTMLButtonElement).style.backgroundColor = 'var(--gray-06)')
            }
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
