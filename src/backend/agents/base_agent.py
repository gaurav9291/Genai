"""Base agent class for all Agentic RAG agents with Langchain integration"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import logging
import time
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import json

logger = logging.getLogger(__name__)

class BaseAgentInput(BaseModel):
    pass

def serialize_lc_object(obj):
    if isinstance(obj, BaseMessage):
        return {"type": type(obj).__name__, "content": obj.content, "additional_kwargs": getattr(obj, "additional_kwargs", {})}
    elif isinstance(obj, list):
        return [serialize_lc_object(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: serialize_lc_object(v) for k, v in obj.items()}
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return serialize_lc_object(obj.__dict__)
    else:
        return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj

def safe_serialize_for_db(obj):
    try:
        serialized = serialize_lc_object(obj)
        json.dumps(serialized)
        return serialized
    except Exception:
        return {"serialization_error": "Failed", "type": str(type(obj))}

class BaseAgent(ABC):
    def __init__(self, name: str, description: str, version: str = "1.0.0"):
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.version = version
        self.created_at = datetime.utcnow()
        self.logger = logging.getLogger(f"agent.{name}")
        self.execution_count = 0
        self.total_execution_time = 0.0
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        pass
    
    def log_execution(self, operation: str, success: bool, details: Optional[Dict] = None):
        try:
            log_data = {
                "agent_id": self.agent_id, "agent_name": self.name, "operation": operation,
                "success": success, "timestamp": datetime.utcnow().isoformat(),
                "details": safe_serialize_for_db(details or {})
            }
            level = logging.INFO if success else logging.ERROR
            status = "completed" if success else "failed"
            self.logger.log(level, f"Agent {self.name} {operation} {status}", extra={"structured_log": log_data})
        except Exception as e:
            self.logger.error(f"Failed to log: {e}")
    
    async def execute_with_metrics(self, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        self.execution_count += 1
        try:
            result = await self.execute(**kwargs)
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            result["_metrics"] = {"execution_time": execution_time, "agent_id": self.agent_id, "execution_count": self.execution_count, "average_execution_time": self.total_execution_time / self.execution_count}
            self.log_execution("execute", True, {"execution_time": execution_time, "result_keys": list(result.keys())})
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_execution("execute", False, {"error": str(e)})
            raise
    
    def to_tool(self) -> BaseTool:
        input_schema = self._get_input_schema()
        class AgentTool(BaseTool):
            name = self.name
            description = self.description
            args_schema = input_schema
            agent_instance = self
            def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs) -> Dict[str, Any]:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        return executor.submit(asyncio.run, self.agent_instance.execute(**kwargs)).result()
                except RuntimeError:
                    return asyncio.run(self.agent_instance.execute(**kwargs))
                except Exception as e:
                    return {"error": str(e)}
            async def _arun(self, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs) -> Dict[str, Any]:
                try:
                    return await self.agent_instance.execute(**kwargs)
                except Exception as e:
                    return {"error": str(e)}
        return AgentTool()
    
    def _get_input_schema(self) -> type[BaseModel]:
        class DefaultInput(BaseAgentInput):
            kwargs: Dict[str, Any] = Field(default_factory=dict)
        return DefaultInput
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id, "name": self.name, "description": self.description, "version": self.version,
            "status": "active", "created_at": self.created_at.isoformat(),
            "metrics": {"execution_count": self.execution_count, "total_execution_time": self.total_execution_time, "average_execution_time": self.total_execution_time / self.execution_count if self.execution_count > 0 else 0}
        }
    
    def validate_input(self, **kwargs) -> bool:
        try:
            self._get_input_schema()(**kwargs)
            return True
        except Exception:
            return False
    
    def preprocess_input(self, **kwargs) -> Dict[str, Any]:
        return {k: v for k, v in kwargs.items() if v is not None and v != ""}
    
    def postprocess_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return safe_serialize_for_db(result) if isinstance(result, dict) else {"result": str(result), "type": type(result).__name__}

class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._logger = logging.getLogger("agent_registry")
    
    def register(self, agent: BaseAgent) -> bool:
        if isinstance(agent, BaseAgent):
            self._agents[agent.name] = agent
            self._logger.info(f"Registered agent: {agent.name}")
            return True
        return False
    
    def unregister(self, name: str) -> bool:
        if name in self._agents:
            del self._agents[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[BaseAgent]:
        return self._agents.get(name)
    
    def list_agents(self) -> List[str]:
        return list(self._agents.keys())
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        return {name: agent.get_status() for name, agent in self._agents.items()}
    
    def get_agent_count(self) -> int:
        return len(self._agents)

agent_registry = AgentRegistry()
