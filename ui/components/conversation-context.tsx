"use client";

import { PanelSection } from "./panel-section";
import { Card, CardContent } from "@/components/ui/card";
import { BookText } from "lucide-react";

interface ConversationContextProps {
  context: {
    passenger_name?: string;
    confirmation_number?: string;
    seat_number?: string;
    flight_number?: string;
    account_number?: string;
  };
}

// 上下文键名的中文映射
const contextKeyMap: Record<string, string> = {
  passenger_name: "乘客姓名",
  confirmation_number: "确认号",
  seat_number: "座位号",
  flight_number: "航班号",
  account_number: "账号",
};

export function ConversationContext({ context }: ConversationContextProps) {
  return (
    <PanelSection
      title="对话上下文"
      icon={<BookText className="h-4 w-4 text-blue-600" />}
    >
      <Card className="bg-gradient-to-r from-white to-gray-50 border-gray-200 shadow-sm">
        <CardContent className="p-3">
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(context).map(([key, value]) => (
              <div
                key={key}
                className="flex items-center gap-2 bg-white p-2 rounded-md border border-gray-200 shadow-sm transition-all"
              >
                <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                <div className="text-xs">
                  <span className="text-zinc-500 font-light">{contextKeyMap[key] || key}:</span>{" "}
                  <span
                    className={
                      value
                        ? "text-zinc-900 font-light"
                        : "text-gray-400 italic"
                    }
                  >
                    {value || "空"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </PanelSection>
  );
}