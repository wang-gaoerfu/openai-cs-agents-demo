"use client";

import { useEffect, useState } from "react";
import { AgentPanel } from "@/components/agent-panel";
import { Chat } from "@/components/Chat";
import type { Agent, AgentEvent, GuardrailCheck, Message } from "@/lib/types";
import { callChatAPI } from "@/lib/api";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [currentAgent, setCurrentAgent] = useState<string>("");
  const [guardrails, setGuardrails] = useState<GuardrailCheck[]>([]);
  const [context, setContext] = useState<Record<string, any>>({});
  const [conversationId, setConversationId] = useState<string | null>(null);
  // 等待助手响应时的加载状态
  const [isLoading, setIsLoading] = useState(false);

  // 启动对话
  useEffect(() => {
    (async () => {
      const data = await callChatAPI("", conversationId ?? "");
      setConversationId(data.conversation_id);
      setCurrentAgent(data.current_agent);
      setContext(data.context);
      const initialEvents = (data.events || []).map((e: any) => ({
        ...e,
        timestamp: e.timestamp ?? Date.now(),
      }));
      setEvents(initialEvents);
      setAgents(data.agents || []);
      setGuardrails(data.guardrails || []);
      if (Array.isArray(data.messages)) {
        setMessages(
          data.messages.map((m: any) => ({
            id: Date.now().toString() + Math.random().toString(),
            content: m.content,
            role: "assistant",
            agent: m.agent,
            timestamp: new Date(),
          }))
        );
      }
    })();
  }, []);

  // 发送用户消息
  const handleSendMessage = async (content: string) => {
    const userMsg: Message = {
      id: Date.now().toString(),
      content,
      role: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    const data = await callChatAPI(content, conversationId ?? "");

    if (!conversationId) setConversationId(data.conversation_id);
    setCurrentAgent(data.current_agent);
    setContext(data.context);
    if (data.events) {
      const stamped = data.events.map((e: any) => ({
        ...e,
        timestamp: e.timestamp ?? Date.now(),
      }));
      setEvents((prev) => [...prev, ...stamped]);
    }
    if (data.agents) setAgents(data.agents);
    // 更新守卫状态
    if (data.guardrails) setGuardrails(data.guardrails);

    if (data.messages) {
      const responses: Message[] = data.messages.map((m: any) => ({
        id: Date.now().toString() + Math.random().toString(),
        content: m.content,
        role: "assistant",
        agent: m.agent,
        timestamp: new Date(),
      }));
      setMessages((prev) => [...prev, ...responses]);
    }

    setIsLoading(false);
  };

  return (
    <main className="flex h-screen gap-2 bg-gray-100 p-2">
      <AgentPanel
        agents={agents}
        currentAgent={currentAgent}
        events={events}
        guardrails={guardrails}
        context={context}
      />
      <Chat
        messages={messages}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
      />
    </main>
  );
}
