"""
API 路由聚合
"""

from gui.backend.api.config import router as config_router
from gui.backend.api.train import router as train_router
from gui.backend.api.system import router as system_router

__all__ = ['config_router', 'train_router', 'system_router']
