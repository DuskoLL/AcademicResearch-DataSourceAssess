#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastAPI Web 应用
- 与 README 中的 "uvicorn app:app" 对齐
- 提供基本只读接口以便仪表板或外部系统查询当前状态
  - GET /           : 简要说明
  - GET /health     : 健康检查
  - GET /chain      : 返回 state/chain.json
  - GET /registry   : 返回 state/registry.json（若存在）
  - GET /master     : 返回 state/master_table.json（若存在）
  - GET /proposals  : 返回 proposals 目录中的文件名列表
"""

import os
import json
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(BASE_DIR, "state")
PROPOSALS_DIR = os.path.join(STATE_DIR, "proposals")


def _load_json(file_path: str, default: Any) -> Any:
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取 {file_path} 失败: {e}")
    return default


app = FastAPI(
    title="DataSourceAssess Web API",
    description="提供链状态、注册表、主表与提案信息的只读访问接口",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "name": "DataSourceAssess Web API",
        "version": "0.1.0",
        "endpoints": [
            "/health",
            "/chain",
            "/registry",
            "/master",
            "/proposals",
        ],
        "note": "使用 uvicorn app:app --host 0.0.0.0 --port 8000 --reload 启动",
    }


@app.get("/health")
def health():
    chain_file = os.path.join(STATE_DIR, "chain.json")
    chain_exists = os.path.exists(chain_file)
    proposals_exists = os.path.exists(PROPOSALS_DIR)
    return {
        "status": "ok",
        "chain_file": chain_exists,
        "proposals_dir": proposals_exists,
    }


@app.get("/chain")
def get_chain():
    chain_file = os.path.join(STATE_DIR, "chain.json")
    doc = _load_json(chain_file, default={"blocks": []})
    return JSONResponse(content=doc)


@app.get("/registry")
def get_registry():
    registry_file = os.path.join(STATE_DIR, "registry.json")
    doc = _load_json(registry_file, default={"sources": []})
    return JSONResponse(content=doc)


@app.get("/master")
def get_master():
    master_file = os.path.join(STATE_DIR, "master_table.json")
    doc = _load_json(master_file, default={"rows": []})
    return JSONResponse(content=doc)


@app.get("/proposals")
def list_proposals() -> Dict[str, List[str]]:
    if not os.path.exists(PROPOSALS_DIR):
        return {"files": []}
    try:
        files = [f for f in os.listdir(PROPOSALS_DIR) if os.path.isfile(os.path.join(PROPOSALS_DIR, f))]
        return {"files": sorted(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取 proposals 目录失败: {e}")