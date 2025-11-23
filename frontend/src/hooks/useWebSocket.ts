import { useState, useEffect, useCallback, useRef } from 'react';
import type { WebSocketMessage, Artifact } from '@/types';

interface WebSocketState {
  isConnected: boolean;
  messages: WebSocketMessage[];
  artifacts: Artifact[];
  sendMessage: (content: string) => void;
  sendContinue: (originalInput: string, responseId: string) => void;
}

const WS_URL = 'ws://localhost:8000/ws';
const RECONNECT_DELAY = 3000;

export const useWebSocket = (): WebSocketState => {
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          console.log('Received message:', message);

          // Add message to history
          setMessages((prev) => [...prev, message]);

          // Handle artifact creation
          if (message.type === 'artifact_created') {
            setArtifacts((prev) => [...prev, message.artifact]);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        wsRef.current = null;

        // Attempt to reconnect
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, RECONNECT_DELAY);
      };
    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
    }
  }, []);

  const sendMessage = useCallback((content: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const message: WebSocketMessage = {
        type: 'user_message',
        content,
      };
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.error('WebSocket is not connected');
    }
  }, []);

  const sendContinue = useCallback((originalInput: string, responseId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const message: WebSocketMessage = {
        type: 'continue_execution',
        content: originalInput,
        previous_response_id: responseId,
      };
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.error('WebSocket is not connected');
    }
  }, []);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return {
    isConnected,
    messages,
    artifacts,
    sendMessage,
    sendContinue,
  };
};
