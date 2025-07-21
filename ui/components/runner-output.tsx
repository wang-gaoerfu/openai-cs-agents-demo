"use client";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { AgentEvent } from "@/lib/types";
import {
  ArrowRightLeft,
  Wrench,
  WrenchIcon,
  RefreshCw,
  MessageSquareMore,
} from "lucide-react";
import { PanelSection } from "./panel-section";

interface RunnerOutputProps {
  runnerEvents: AgentEvent[];
}

function formatEventName(type: string) {
  const typeMap: Record<string, string> = {
    handoff: "转接",
    tool_call: "工具调用",
    tool_output: "工具输出",
    context_update: "上下文更新",
    message: "消息"
  };
  
  return typeMap[type] || type;
}

function EventIcon({ type }: { type: string }) {
  const className = "h-4 w-4 text-zinc-600";
  switch (type) {
    case "handoff":
      return <ArrowRightLeft className={className} />;
    case "tool_call":
      return <Wrench className={className} />;
    case "tool_output":
      return <WrenchIcon className={className} />;
    case "context_update":
      return <RefreshCw className={className} />;
    default:
      return null;
  }
}

function EventDetails({ event }: { event: AgentEvent }) {
  let details = null;
  const className =
    "border border-gray-100 text-xs p-2.5 rounded-md flex flex-col gap-2";
  switch (event.type) {
    case "handoff":
      details = event.metadata && (
        <div className={className}>
          <div className="text-gray-600">
            <span className="text-zinc-600 font-medium">从:</span>{" "}
            {event.metadata.source_agent}
          </div>
          <div className="text-gray-600">
            <span className="text-zinc-600 font-medium">到:</span>{" "}
            {event.metadata.target_agent}
          </div>
        </div>
      );
      break;
    case "tool_call":
      details = event.metadata && event.metadata.tool_args && (
        <div className={className}>
          <div className="text-xs text-zinc-600 mb-1 font-medium">
            参数
          </div>
          <pre className="text-xs text-gray-600 bg-gray-50 p-2 rounded overflow-x-auto">
            {JSON.stringify(event.metadata.tool_args, null, 2)}
          </pre>
        </div>
      );
      break;
    case "tool_output":
      details = event.metadata && event.metadata.tool_result && (
        <div className={className}>
          <div className="text-xs text-zinc-600 mb-1 font-medium">结果</div>
          <pre className="text-xs text-gray-600 bg-gray-50 p-2 rounded overflow-x-auto">
            {JSON.stringify(event.metadata.tool_result, null, 2)}
          </pre>
        </div>
      );
      break;
    case "context_update":
      details = event.metadata?.changes && (
        <div className={className}>
          {Object.entries(event.metadata.changes).map(([key, value]) => (
            <div key={key} className="text-xs">
              <div className="text-gray-600">
                <span className="text-zinc-600 font-medium">{key}:</span>{" "}
                {value ?? "空"}
              </div>
            </div>
          ))}
        </div>
      );
      break;
    default:
      return null;
  }

  return (
    <div className="mt-1 text-sm">
      {event.content && (
        <div className="text-gray-700 font-mono mb-2">{event.content}</div>
      )}
      {details}
    </div>
  );
}

function TimeBadge({ timestamp }: { timestamp: Date }) {
  const date =
    timestamp && typeof (timestamp as any)?.toDate === "function"
      ? (timestamp as any).toDate()
      : timestamp;
  const formattedDate = new Date(date).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  return (
    <Badge
      variant="outline"
      className="text-[10px] h-5 bg-white text-zinc-500 border-gray-200"
    >
      {formattedDate}
    </Badge>
  );
}

export function RunnerOutput({ runnerEvents }: RunnerOutputProps) {
  return (
    <div className="flex-1 overflow-hidden">
      <PanelSection title="运行输出" icon={<MessageSquareMore className="h-4 w-4 text-blue-600" />}>
        <ScrollArea className="h-[calc(100%-2rem)] rounded-md border border-gray-200 bg-gray-100 shadow-sm">
        <div className="p-4 space-y-3">
          {runnerEvents.length === 0 ? (
            <p className="text-center text-zinc-500 p-4">
              暂无运行事件
            </p>
          ) : (
            runnerEvents.map((event) => (
              <Card
                key={event.id}
                className="border border-gray-200 bg-white shadow-sm rounded-lg"
              >
                <CardHeader className="flex flex-row justify-between items-center p-4">
                  <span className="font-medium text-gray-800 text-sm">
                    {event.agent}
                  </span>
                  <TimeBadge timestamp={event.timestamp} />
                </CardHeader>

                <CardContent className="flex items-start gap-3 p-4">
                  <div className="rounded-full p-2 bg-gray-100 flex items-center gap-2">
                    <EventIcon type={event.type} />
                    <div className="text-xs text-gray-600">
                      {formatEventName(event.type)}
                    </div>
                  </div>

                  <div className="flex-1">
                    <EventDetails event={event} />
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </div>
        </ScrollArea>
      </PanelSection>
    </div>
  );
}
