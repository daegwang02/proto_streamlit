# agents/base_agent.py

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """모든 에이전트의 기본이 되는 추상 클래스입니다."""
    
    @abstractmethod
    def run(self, **kwargs):
        """에이전트의 주요 로직을 실행하는 메서드입니다."""
        pass
