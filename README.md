# Oracle数据源评估区块链系统

## 🎯 项目概述

Oracle数据源评估区块链系统是一个基于分布式共识机制的数据源质量评估平台，专为加密货币价格数据源的可靠性评估而设计。

### 核心功能
- **多维度数据源评估**: 8个关键指标（准确度、可用性、响应时间、更新频率、完整性、错误率、历史表现、波动性）
- **智能聚类分析**: K-means算法实现数据源的自动分组和质量分级
- **区块链共识机制**: 分布式提议者-矿工架构确保评估结果的可信性
- **实时API监控**: 多重备用机制保障数据采集的高可用性
- **可视化分析报告**: 专业图表和统计分析

## 🚀 快速开始

### 环境要求
- Python 3.9+
- 虚拟环境（推荐）

### 安装步骤
```bash
# 1. 克隆项目
git clone <repository-url>
cd DataSourceAssess

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

### 运行系统
```bash
# 1. 激活虚拟环境
source .venv/bin/activate

# 2. 启动三个提案节点（需要三个独立终端）
# 终端1 - 启动提案节点1
python proposer_node.py --id proposer-1

# 终端2 - 启动提案节点2
python proposer_node.py --id proposer-2

# 终端3 - 启动提案节点3
python proposer_node.py --id proposer-3

# 3. 启动矿工节点（终端4）
# 保留历史（默认，推荐）
python miner_node.py --id miner-1 --quorum 3
# 或：清空历史并重置（仅当需要全新初始化时）
python miner_node.py --id miner-1 --quorum 3 --reset-state

# 4. 启动Web仪表板（终端5）
source .venv/bin/activate && uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 5. 生成可视化报告
python visualize_reports.py

# 6. 检查系统健康状况
python api_health_check.py
```

## 📊 评估体系

### 质量等级
- **A+级**: 90-100分，顶级数据源
- **A级**: 80-89分，优质数据源
- **B级**: 70-79分，良好数据源
- **C级**: 60-69分，一般数据源
- **D级**: <60分，较差数据源

### 评估维度
1. **准确度**: 与基准价格的偏差程度
2. **可用性**: API服务的在线时间比例
3. **响应时间**: API请求的响应速度
4. **更新频率**: 数据更新的及时性
5. **完整性**: 数据字段的完整程度
6. **错误率**: API调用失败的频率
7. **历史表现**: 长期稳定性评估
8. **波动性**: 数据变化的规律性

## ⚙️ 系统配置

主要配置文件：`config.yaml`
```yaml
# 网络配置
network:
  timeout_sec: 5.0
  retries: 3
  concurrent_requests: 10

# 聚类配置  
clustering:
  k: 5
  max_iter: 100
  tolerance: 1e-4

# 区块链配置
mining:
  interval_sec: 60
  quorum: 3
```

## 📈 系统状态

**当前状态**: ✅ **系统正常运行**

- ✅ 数据源总数: 1,100个
- ✅ 高质量源 (A+/A级): 75.5%
- ✅ API调用成功率: 100%
- ✅ 可视化报告: 13类图表已生成
- ✅ 平均响应时间: 600-1000ms

## 📁 核心文件

```
项目结构/
├── README.md                    # 项目说明
├── config.yaml                  # 系统配置
├── proposer_node.py            # 提议者节点
├── miner_node.py               # 矿工节点
├── clustering.py               # 聚类算法
├── visualize_reports.py        # 可视化报告
├── api_health_check.py         # 健康检查
└── state/                      # 数据存储
    ├── master_table.json       # 评估结果
    ├── data_sources.json       # 数据源列表
    └── reports/                # 生成的报告
```

---

**开发团队**: epochChain  
**版本**: 2.0  
**许可证**: MIT License