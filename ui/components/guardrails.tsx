"use client";

import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Shield, CheckCircle, XCircle } from "lucide-react";
import { PanelSection } from "./panel-section";
import type { GuardrailCheck } from "@/lib/types";

interface GuardrailsProps {
  guardrails: GuardrailCheck[];
  inputGuardrails: string[];
}

export function Guardrails({ guardrails, inputGuardrails }: GuardrailsProps) {
  const guardrailNameMap: Record<string, string> = {
    relevance_guardrail: "相关性守卫",
    jailbreak_guardrail: "越狱守卫",
    "Relevance Guardrail": "相关性守卫",
    "Jailbreak Guardrail": "越狱守卫",
    "相关性守卫": "相关性守卫",
    "越狱守卫": "越狱守卫",
  };

  const guardrailDescriptionMap: Record<string, string> = {
    "相关性守卫": "确保消息与航空支持相关",
    "越狱守卫": "检测并阻止绕过或覆盖系统指令的尝试",
  };

  const extractGuardrailName = (rawName: string): string =>
    guardrailNameMap[rawName] ?? rawName;

  const guardrailsToShow: GuardrailCheck[] = inputGuardrails.map((rawName) => {
    const existing = guardrails.find((gr) => gr.name === rawName);
    if (existing) {
      return existing;
    }
    return {
      id: rawName,
      name: rawName,
      input: "",
      reasoning: "",
      passed: false,
      timestamp: new Date(),
    };
  });

  return (
    <PanelSection
      title="守卫机制"
      icon={<Shield className="h-4 w-4 text-blue-600" />}
    >
      <div className="grid grid-cols-3 gap-3">
        {guardrailsToShow.map((gr) => (
          <Card
            key={gr.id}
            className={`bg-white border-gray-200 transition-all ${
              !gr.input ? "opacity-60" : ""
            }`}
          >
            <CardHeader className="p-3 pb-1">
              <CardTitle className="text-sm flex items-center text-zinc-900">
                {extractGuardrailName(gr.name)}
              </CardTitle>
            </CardHeader>
            <CardContent className="p-3 pt-1">
              <p className="text-xs font-light text-zinc-500 mb-1">
                {(() => {
                  const title = extractGuardrailName(gr.name);
                  return guardrailDescriptionMap[title] ?? gr.input;
                })()}
              </p>
              <div className="flex text-xs">
                {!gr.input || gr.passed ? (
                  <Badge className="mt-2 px-2 py-1 bg-emerald-500 hover:bg-emerald-600 flex items-center text-white">
                    <CheckCircle className="h-4 w-4 mr-1 text-white" />
                    通过
                  </Badge>
                ) : (
                  <Badge className="mt-2 px-2 py-1 bg-red-500 hover:bg-red-600 flex items-center text-white">
                    <XCircle className="h-4 w-4 mr-1 text-white" />
                    失败
                  </Badge>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </PanelSection>
  );
}
