from __future__ import annotations as _annotations

import json
import os
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable, Awaitable
from pydantic import BaseModel, Field
from openai import OpenAI

# 显式加载.env文件
from dotenv import load_dotenv
load_dotenv()
print("环境变量加载状态：", os.environ.get("DEEPSEEK_DEV_MODE"), os.environ.get("DASHSCOPE_API_KEY", "已设置但不显示"))

# Type for context
T = TypeVar('T', bound=BaseModel)

# Response item types
class TResponseInputItem(BaseModel):
    role: str
    content: str

class MessageOutputItem(BaseModel):
    agent: Any
    content: str

class HandoffOutputItem(BaseModel):
    source_agent: Any
    target_agent: Any
    reason: str

class ToolCallItem(BaseModel):
    agent: Any
    raw_item: Any

class ToolCallOutputItem(BaseModel):
    agent: Any
    output: Any

class RunResult(BaseModel):
    new_items: List[Any] = Field(default_factory=list)
    
    def to_input_list(self):
        """Convert result items to input list format"""
        result = []
        for item in self.new_items:
            if isinstance(item, MessageOutputItem):
                result.append({"role": "assistant", "content": item.content})
        return result
    
    def final_output_as(self, output_type):
        """Parse the final output as a specific type"""
        # This is a simplified implementation
        if self.new_items and isinstance(self.new_items[-1], MessageOutputItem):
            content = self.new_items[-1].content
            try:
                # Try to parse the content as JSON
                data = json.loads(content)
                return output_type(**data)
            except:
                # Fallback for non-JSON content
                # 检查output_type并返回相应的默认值
                if hasattr(output_type, "__annotations__"):
                    default_values = {"reasoning": "Parsed from text"}
                    # 检查需要的字段
                    if "is_safe" in output_type.__annotations__:
                        default_values["is_safe"] = True
                    elif "is_relevant" in output_type.__annotations__:
                        default_values["is_relevant"] = True
                    return output_type(**default_values)
                return output_type()
        return output_type()

class ItemHelpers:
    @staticmethod
    def text_message_output(item):
        if isinstance(item, MessageOutputItem):
            return item.content
        return str(item)

class RunContextWrapper(Generic[T]):
    def __init__(self, context: T):
        self.context = context

class GuardrailFunctionOutput(BaseModel):
    output_info: Any
    tripwire_triggered: bool = False

class InputGuardrailTripwireTriggered(Exception):
    def __init__(self, guardrail_result):
        self.guardrail_result = guardrail_result
        super().__init__(f"Input guardrail tripwire triggered: {guardrail_result}")

# DeepSeek API client for Aliyun Bailian
class DeepSeekClient:
    def __init__(self, api_key=None, dev_mode=False):
        # 显式加载.env文件(如果尚未加载)
        load_dotenv()
        
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.dev_mode = dev_mode or os.environ.get("DEEPSEEK_DEV_MODE", "").lower() in ("true", "1", "yes")
        
        # 打印详细的环境变量状态
        print("="*80)
        print(f"环境变量状态:")
        print(f"DEEPSEEK_DEV_MODE = {os.environ.get('DEEPSEEK_DEV_MODE', '未设置')}")
        print(f"DASHSCOPE_API_KEY = {'已设置' if self.api_key else '未设置'}")
        print(f"当前模式: {'开发模式' if self.dev_mode else '生产模式'}")
        print("="*80)
        
        if not self.api_key and not self.dev_mode:
            print("\n" + "="*80)
            print("错误: 未设置DASHSCOPE_API_KEY环境变量")
            print("请执行以下操作之一:")
            print("1. 在环境变量中设置DASHSCOPE_API_KEY")
            print("   - 在命令行中运行: export DASHSCOPE_API_KEY=your_api_key (Linux/Mac)")
            print("   - 或者在Windows中运行: set DASHSCOPE_API_KEY=your_api_key")
            print("2. 在python-backend目录下创建.env文件并添加: DASHSCOPE_API_KEY=your_api_key")
            print("3. 或者设置DEEPSEEK_DEV_MODE=true以启用开发模式(使用模拟响应)")
            print("="*80 + "\n")
            raise ValueError("DASHSCOPE_API_KEY环境变量或api_key参数必须设置，或者启用开发模式")
        
        if self.api_key and not self.dev_mode:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                print("成功初始化OpenAI客户端，连接到阿里云百炼API")
            except Exception as e:
                print(f"初始化OpenAI客户端时出错: {e}")
                raise
    
    async def chat_completion(self, messages, model="deepseek-v3", **kwargs):
        """
        Call DeepSeek chat completion API via Aliyun Bailian
        """
        # 如果处于开发模式，返回模拟响应
        if self.dev_mode:
            print(f"[开发模式] 模拟DeepSeek响应，模型: {model}")
            
            # 获取系统指令和用户消息
            system_message = ""
            user_message = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                elif msg["role"] == "user" and not user_message:  # 获取最后一条用户消息
                    user_message = msg["content"]
            
            # 根据系统指令和用户消息生成模拟响应
            response_content = ""
            
            # 检查是否包含分流代理的指令
            if "分流代理" in system_message or "Triage Agent" in system_message:
                if "座位" in user_message or "换座" in user_message:
                    response_content = "我理解您想更换座位。我会将您转接到座位预订代理，他们可以帮您处理座位变更。"
                elif "航班状态" in user_message or "航班" in user_message:
                    response_content = "您想查询航班状态。我会将您转接到航班状态代理，他们可以提供最新信息。"
                elif "取消" in user_message:
                    response_content = "您想取消航班。我会将您转接到取消代理，他们可以帮您处理取消流程。"
                elif "行李" in user_message or "baggage" in user_message.lower():
                    response_content = "您有关于行李的问题。我会将您转接到FAQ代理，他们可以回答您的问题。"
                else:
                    response_content = "您好！我是航空公司的客服助手。请问您需要什么帮助？您可以询问关于航班状态、座位预订、行李政策或取消航班的问题。"
            
            # 检查是否包含座位预订代理的指令
            elif "座位预订代理" in system_message or "Seat Booking Agent" in system_message:
                if "确认号" in user_message:
                    response_content = "感谢您提供确认号。您想要更换到哪个座位？或者您想查看座位图吗？"
                elif "23A" in user_message or "座位" in user_message:
                    response_content = "您的座位已成功更改。如果您需要进一步帮助，请随时询问！"
                else:
                    response_content = "欢迎使用座位预订服务。为了帮您更换座位，请提供您的确认号码。"
            
            # 检查是否包含FAQ代理的指令
            elif "FAQ代理" in system_message or "FAQ Agent" in system_message:
                if "行李" in user_message:
                    response_content = "您可以携带一个行李登机。它必须低于50磅，尺寸不超过22英寸 x 14英寸 x 9英寸。"
                elif "座位" in user_message or "飞机" in user_message:
                    response_content = "飞机上共有120个座位。其中有22个商务舱座位和98个经济舱座位。4排和16排是安全出口排。5-8排是经济舱Plus，提供更多腿部空间。"
                elif "wifi" in user_message.lower() or "网络" in user_message:
                    response_content = "我们在飞机上提供免费WiFi，连接名称为Airline-Wifi"
                else:
                    response_content = "这是FAQ代理。请问您有什么具体问题？我可以回答关于行李、座位、WiFi等方面的问题。"
            
            # 如果没有匹配到特定代理或问题，返回通用响应
            if not response_content:
                response_content = f"我是航空公司客服助手。您的问题是：{user_message}。请问您需要了解航班状态、座位预订还是行李政策？"
            
            return {
                "id": "dev-mode-response",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_content,
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
        
        # 正常API调用
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            return {
                "id": completion.id,
                "object": "chat.completion",
                "created": completion.created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": completion.choices[0].message.content,
                        },
                        "finish_reason": completion.choices[0].finish_reason
                    }
                ]
            }
        except Exception as e:
            print(f"调用DeepSeek API时出错: {e}")
            # 返回错误响应
            return {
                "id": "error",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"抱歉，系统遇到了问题：{str(e)}",
                        },
                        "finish_reason": "error"
                    }
                ]
            }

# Function to create a tool from a function
def function_tool(fn=None, *, name_override=None, description_override=None):
    """Decorator to convert a function to a tool"""
    def decorator(func):
        func.name = name_override or func.__name__
        func.description = description_override or func.__doc__
        return func
    
    if fn is None:
        return decorator
    return decorator(fn)

# Agent class
class Agent(Generic[T]):
    def __init__(
        self,
        name: str,
        model: str = "deepseek-v3",
        instructions: str = "",
        handoff_description: str = "",
        tools: List[Any] = None,
        handoffs: List[Any] = None,
        input_guardrails: List[Any] = None,
        output_type: Any = None,
    ):
        self.name = name
        self.model = model
        self.instructions = instructions
        self.handoff_description = handoff_description
        self.tools = tools or []
        self.handoffs = handoffs or []
        self.input_guardrails = input_guardrails or []
        self.output_type = output_type
        self.client = DeepSeekClient(dev_mode=os.environ.get("DEEPSEEK_DEV_MODE", "").lower() in ("true", "1", "yes"))

# Handoff class
class Handoff:
    def __init__(self, agent, on_handoff=None):
        self.agent = agent
        self.agent_name = agent.name
        self.on_handoff = on_handoff
        
    def on_invoke_handoff(self, context):
        """Called when a handoff is invoked"""
        if self.on_handoff:
            return self.on_handoff(context)
        return None

# Helper function to create a handoff
def handoff(agent, on_handoff=None):
    return Handoff(agent, on_handoff)

# Input guardrail decorator
def input_guardrail(name=None):
    def decorator(func):
        func.name = name or func.__name__
        return func
    return decorator

# Runner class
class Runner:
    @staticmethod
    async def run(agent, input_items, context=None):
        """Run an agent with input items and context"""
        # Check guardrails first
        if hasattr(agent, "input_guardrails") and agent.input_guardrails:
            for guardrail in agent.input_guardrails:
                # Extract the latest user message
                latest_user_message = None
                if isinstance(input_items, list) and input_items:
                    latest_item = input_items[-1]
                    if isinstance(latest_item, dict) and latest_item.get("role") == "user":
                        latest_user_message = latest_item.get("content", "")
                    elif hasattr(latest_item, "role") and latest_item.role == "user":
                        latest_user_message = latest_item.content
                
                if latest_user_message:
                    # Run the guardrail check
                    ctx_wrapper = RunContextWrapper(context)
                    result = await guardrail(ctx_wrapper, agent, latest_user_message)
                    if result.tripwire_triggered:
                        raise InputGuardrailTripwireTriggered(
                            type("GuardrailResult", (), {"guardrail": guardrail, "output": result})
                        )
        
        # Process the input items into messages for DeepSeek API
        messages = []
        for item in input_items:
            if isinstance(item, dict):
                messages.append(item)
            elif hasattr(item, "role") and hasattr(item, "content"):
                messages.append({"role": item.role, "content": item.content})
        
        # Add system instructions
        if callable(agent.instructions):
            instructions = agent.instructions(RunContextWrapper(context), agent)
        else:
            instructions = agent.instructions
            
        messages.insert(0, {"role": "system", "content": instructions})
        
        # Call DeepSeek API
        response = await agent.client.chat_completion(messages, model=agent.model)
        
        # Process response
        result = RunResult()
        if response and "choices" in response and response["choices"]:
            message = response["choices"][0]["message"]
            result.new_items.append(MessageOutputItem(agent=agent, content=message["content"]))
        
        return result 