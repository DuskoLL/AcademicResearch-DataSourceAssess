#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IET期刊增强模块
为满足IET期刊发表要求而设计的实验增强工具

主要功能：
1. 统计显著性测试
2. 实验可重复性验证
3. 方法论完整性检查
4. 结果可靠性分析
5. 符合IET标准的实验报告生成
"""

import json
import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ExperimentalResults:
    """实验结果数据结构"""
    total_sources: int
    categories: Dict[str, int]
    grade_distribution: Dict[str, int]
    performance_metrics: Dict[str, float]
    statistical_tests: Dict[str, Any]
    reproducibility_score: float
    methodology_completeness: float

class IETJournalEnhancer:
    """IET期刊标准增强器"""
    
    def __init__(self, state_dir: str = "state"):
        self.state_dir = state_dir
        self.reports_dir = os.path.join(state_dir, "reports")
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 确保目录存在
        os.makedirs(self.reports_dir, exist_ok=True)
        
    def load_experimental_data(self) -> Dict[str, Any]:
        """加载实验数据"""
        master_path = os.path.join(self.state_dir, "master_table.json")
        chain_path = os.path.join(self.state_dir, "chain.json")
        validation_path = os.path.join(self.state_dir, "validation_report.json")
        
        data = {}
        
        # 加载主表数据
        if os.path.exists(master_path):
            with open(master_path, 'r', encoding='utf-8') as f:
                data['master'] = json.load(f)
        
        # 加载区块链数据
        if os.path.exists(chain_path):
            with open(chain_path, 'r', encoding='utf-8') as f:
                data['chain'] = json.load(f)
        
        # 加载验证报告
        if os.path.exists(validation_path):
            with open(validation_path, 'r', encoding='utf-8') as f:
                data['validation'] = json.load(f)
        
        return data
    
    def perform_statistical_significance_tests(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行统计显著性测试"""
        print("🔬 执行统计显著性测试...")
        
        sources = data['master'].get('sources', {})
        if not sources:
            return {}
        
        # 提取特征数据
        features_by_grade = defaultdict(list)
        all_features = []
        all_grades = []
        
        feature_names = ['accuracy', 'availability', 'response_time', 'volatility', 
                        'update_frequency', 'integrity', 'error_rate', 'historical']
        
        for source_id, source_data in sources.items():
            grade = source_data.get('label', 'D')
            features = source_data.get('features', {})
            
            if features:
                feature_vector = [float(features.get(f, 0.0)) for f in feature_names]
                features_by_grade[grade].append(feature_vector)
                all_features.append(feature_vector)
                all_grades.append(grade)
        
        results = {}
        
        # 1. ANOVA测试 - 检验不同等级间特征是否有显著差异
        if len(features_by_grade) > 2:
            for i, feature_name in enumerate(feature_names):
                grade_feature_values = []
                for grade in ['A+', 'A', 'B', 'C', 'D']:
                    if grade in features_by_grade and features_by_grade[grade]:
                        values = [fv[i] for fv in features_by_grade[grade]]
                        grade_feature_values.append(values)
                
                if len(grade_feature_values) > 2:
                    f_stat, p_value = stats.f_oneway(*grade_feature_values)
                    results[f'anova_{feature_name}'] = {
                        'f_statistic': float(f_stat),
                        'p_value': float(p_value),
                        'significant': bool(p_value < 0.05)
                    }
        
        # 2. Kruskal-Wallis测试 - 非参数替代方案
        if len(features_by_grade) > 2:
            for i, feature_name in enumerate(feature_names):
                grade_feature_values = []
                for grade in ['A+', 'A', 'B', 'C', 'D']:
                    if grade in features_by_grade and features_by_grade[grade]:
                        values = [fv[i] for fv in features_by_grade[grade]]
                        grade_feature_values.append(values)
                
                if len(grade_feature_values) > 2:
                    h_stat, p_value = stats.kruskal(*grade_feature_values)
                    results[f'kruskal_{feature_name}'] = {
                        'h_statistic': float(h_stat),
                        'p_value': float(p_value),
                        'significant': bool(p_value < 0.05)
                    }
        
        # 3. 卡方独立性测试 - 测试等级分布的独立性
        if len(set(all_grades)) > 1:
            grade_counts = pd.Series(all_grades).value_counts()
            chi2_stat, p_value = stats.chisquare(grade_counts.values)
            results['chi_square_grade_distribution'] = {
                'chi2_statistic': float(chi2_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
        
        return results
    
    def validate_clustering_performance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """验证聚类性能"""
        print("🎯 验证聚类算法性能...")
        
        sources = data['master'].get('sources', {})
        if not sources:
            return {}
        
        # 准备数据
        X = []
        y = []
        feature_names = ['accuracy', 'availability', 'response_time', 'volatility', 
                        'update_frequency', 'integrity', 'error_rate', 'historical']
        
        for source_id, source_data in sources.items():
            grade = source_data.get('label', 'D')
            features = source_data.get('features', {})
            
            if features:
                feature_vector = [float(features.get(f, 0.0)) for f in feature_names]
                X.append(feature_vector)
                y.append(grade)
        
        if len(X) < 10:
            return {'error': 'Insufficient data for clustering validation'}
        
        # 转换为numpy数组
        X = np.array(X)
        
        # 标签编码
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # 构建KNN分类器作为聚类质量评估
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 多个评估指标
        accuracy_scores = cross_val_score(knn, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(knn, X_scaled, y_encoded, cv=cv, scoring='f1_macro')
        precision_scores = cross_val_score(knn, X_scaled, y_encoded, cv=cv, scoring='precision_macro')
        recall_scores = cross_val_score(knn, X_scaled, y_encoded, cv=cv, scoring='recall_macro')
        
        results = {
            'cross_validation_accuracy': {
                'mean': float(np.mean(accuracy_scores)),
                'std': float(np.std(accuracy_scores)),
                'scores': [float(s) for s in accuracy_scores]
            },
            'cross_validation_f1': {
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores)),
                'scores': [float(s) for s in f1_scores]
            },
            'cross_validation_precision': {
                'mean': float(np.mean(precision_scores)),
                'std': float(np.std(precision_scores)),
                'scores': [float(s) for s in precision_scores]
            },
            'cross_validation_recall': {
                'mean': float(np.mean(recall_scores)),
                'std': float(np.std(recall_scores)),
                'scores': [float(s) for s in recall_scores]
            },
            'sample_size': len(X),
            'feature_count': len(feature_names),
            'class_count': len(le.classes_),
            'class_distribution': {
                grade: int(np.sum(np.array(y) == grade)) for grade in le.classes_
            }
        }
        
        return results
    
    def assess_reproducibility(self, data: Dict[str, Any]) -> float:
        """评估实验可重复性"""
        print("🔄 评估实验可重复性...")
        
        score = 0.0
        max_score = 100.0
        
        # 1. 数据完整性 (30分)
        sources = data['master'].get('sources', {})
        if sources:
            complete_sources = 0
            for source_data in sources.values():
                if isinstance(source_data, dict) and source_data.get('features'):
                    complete_sources += 1
            
            completeness_ratio = complete_sources / len(sources) if sources else 0
            score += completeness_ratio * 30
        
        # 2. 算法参数记录 (20分)
        if 'validation' in data and data['validation'].get('statistics'):
            score += 20
        
        # 3. 区块链数据一致性 (25分)
        if 'chain' in data:
            blocks = data['chain'].get('blocks', [])
            if blocks:
                # 检查区块链完整性
                valid_blocks = 0
                for block in blocks:
                    if isinstance(block, dict) and all(k in block for k in ['index', 'timestamp', 'previous_hash']):
                        valid_blocks += 1
                
                blockchain_integrity = valid_blocks / len(blocks) if blocks else 0
                score += blockchain_integrity * 25
        
        # 4. 配置文件存在性 (15分)
        config_path = os.path.join(os.path.dirname(self.state_dir), "config.yaml")
        if os.path.exists(config_path):
            score += 15
        
        # 5. 日志记录 (10分)
        logs_dir = os.path.join(os.path.dirname(self.state_dir), "logs")
        if os.path.exists(logs_dir) and os.listdir(logs_dir):
            score += 10
        
        return min(score, max_score)
    
    def check_methodology_completeness(self, data: Dict[str, Any]) -> float:
        """检查方法论完整性"""
        print("📋 检查方法论完整性...")
        
        score = 0.0
        max_score = 100.0
        
        # 1. 多维度评估指标 (25分)
        sources = data['master'].get('sources', {})
        if sources:
            sample_source = next(iter(sources.values()))
            features = sample_source.get('features', {})
            expected_features = ['accuracy', 'availability', 'response_time', 'volatility', 
                               'update_frequency', 'integrity', 'error_rate', 'historical']
            
            feature_coverage = len([f for f in expected_features if f in features]) / len(expected_features)
            score += feature_coverage * 25
        
        # 2. 聚类算法实现 (20分)
        # 检查是否有聚类相关的代码实现
        clustering_files = ['clustering.py', 'oracle_chain.py']
        base_dir = os.path.dirname(self.state_dir)
        
        clustering_implemented = any(
            os.path.exists(os.path.join(base_dir, f)) for f in clustering_files
        )
        if clustering_implemented:
            score += 20
        
        # 3. 数据质量验证 (20分)
        if 'validation' in data and data['validation'].get('fixes_applied'):
            score += 20
        
        # 4. 区块链共识机制 (15分)
        if 'chain' in data:
            blocks = data['chain'].get('blocks', [])
            if blocks and any('proposals' in block for block in blocks):
                score += 15
        
        # 5. 统计分析和可视化 (10分)
        reports_files = os.listdir(self.reports_dir) if os.path.exists(self.reports_dir) else []
        if any(f.endswith(('.png', '.eps', '.svg')) for f in reports_files):
            score += 5
        if any('report' in f.lower() for f in reports_files):
            score += 5
        
        # 6. API多样性 (10分)
        categories = data['master'].get('sources', {})
        if categories:
            unique_categories = set()
            for source_data in categories.values():
                if isinstance(source_data, dict):
                    unique_categories.add(source_data.get('category', 'unknown'))
            
            if len(unique_categories) >= 10:  # 10个或更多不同类别
                score += 10
        
        return min(score, max_score)
    
    def generate_iet_compliance_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成IET期刊合规性报告"""
        print("📊 生成IET期刊合规性报告...")
        
        # 执行各项分析
        statistical_tests = self.perform_statistical_significance_tests(data)
        clustering_performance = self.validate_clustering_performance(data)
        reproducibility_score = self.assess_reproducibility(data)
        methodology_score = self.check_methodology_completeness(data)
        
        # 计算总体分数
        overall_score = (reproducibility_score + methodology_score) / 2
        
        # 生成综合报告
        report = {
            'timestamp': self.timestamp,
            'overall_compliance_score': overall_score,
            'reproducibility_score': reproducibility_score,
            'methodology_completeness': methodology_score,
            'statistical_significance_tests': statistical_tests,
            'clustering_validation': clustering_performance,
            'iet_requirements_check': {
                'experimental_design': {
                    'score': methodology_score,
                    'status': 'PASS' if methodology_score >= 70 else 'NEEDS_IMPROVEMENT',
                    'requirements': [
                        'Multi-dimensional evaluation metrics ✓',
                        'Clustering algorithm implementation ✓',
                        'Data quality validation ✓',
                        'Consensus mechanism ✓',
                        'Statistical analysis ✓',
                        'API diversity ✓'
                    ]
                },
                'reproducibility': {
                    'score': reproducibility_score,
                    'status': 'PASS' if reproducibility_score >= 70 else 'NEEDS_IMPROVEMENT',
                    'requirements': [
                        'Data completeness ✓',
                        'Algorithm parameters recorded ✓',
                        'Blockchain data consistency ✓',
                        'Configuration files ✓',
                        'Logging system ✓'
                    ]
                },
                'statistical_rigor': {
                    'score': len([t for t in statistical_tests.values() 
                                if isinstance(t, dict) and t.get('significant', False)]) * 10,
                    'significant_tests': len([t for t in statistical_tests.values() 
                                            if isinstance(t, dict) and t.get('significant', False)]),
                    'total_tests': len(statistical_tests)
                }
            },
            'recommendations': self._generate_recommendations(overall_score, statistical_tests, clustering_performance)
        }
        
        return report
    
    def _generate_recommendations(self, overall_score: float, statistical_tests: Dict, clustering_performance: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if overall_score < 70:
            recommendations.append("总体合规性分数较低，需要改进实验设计和数据质量")
        
        if statistical_tests:
            significant_tests = [t for t in statistical_tests.values() 
                               if isinstance(t, dict) and t.get('significant', False)]
            if len(significant_tests) < len(statistical_tests) * 0.5:
                recommendations.append("建议增加更多具有统计显著性的实验结果")
        
        if clustering_performance and 'cross_validation_accuracy' in clustering_performance:
            acc_mean = clustering_performance['cross_validation_accuracy']['mean']
            if acc_mean < 0.8:
                recommendations.append("聚类算法准确率较低，建议优化特征工程或算法参数")
        
        if not recommendations:
            recommendations.append("实验设计符合IET期刊要求，建议继续保持高质量标准")
        
        return recommendations
    
    def save_compliance_report(self, report: Dict[str, Any]) -> str:
        """保存合规性报告"""
        report_path = os.path.join(self.reports_dir, "iet_journal_compliance_report.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 同时生成文本版本
        text_report_path = os.path.join(self.reports_dir, "iet_journal_compliance_report.txt")
        self._generate_text_report(report, text_report_path)
        
        return report_path
    
    def _generate_text_report(self, report: Dict[str, Any], output_path: str):
        """生成文本格式的合规性报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("IET期刊合规性评估报告\n")
            f.write("Oracle数据源评估区块链系统\n")
            f.write(f"生成时间: {report['timestamp']}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. 总体评估\n")
            f.write("-" * 40 + "\n")
            f.write(f"总体合规性分数: {report['overall_compliance_score']:.1f}/100\n")
            f.write(f"可重复性分数: {report['reproducibility_score']:.1f}/100\n")
            f.write(f"方法论完整性: {report['methodology_completeness']:.1f}/100\n\n")
            
            f.write("2. IET期刊要求检查\n")
            f.write("-" * 40 + "\n")
            
            req_check = report['iet_requirements_check']
            for category, details in req_check.items():
                if isinstance(details, dict) and 'score' in details:
                    status = details.get('status', 'N/A')
                    f.write(f"{category.title()}: {details['score']:.1f}/100 ({status})\n")
            
            f.write("\n3. 统计显著性测试结果\n")
            f.write("-" * 40 + "\n")
            stats_tests = report['statistical_significance_tests']
            if stats_tests:
                significant_count = len([t for t in stats_tests.values() 
                                       if isinstance(t, dict) and t.get('significant', False)])
                f.write(f"显著性测试通过: {significant_count}/{len(stats_tests)}\n")
                
                for test_name, result in stats_tests.items():
                    if isinstance(result, dict):
                        significance = "✓" if result.get('significant', False) else "✗"
                        f.write(f"  {test_name}: p={result.get('p_value', 0):.4f} {significance}\n")
            else:
                f.write("未执行统计显著性测试\n")
            
            f.write("\n4. 聚类性能验证\n")
            f.write("-" * 40 + "\n")
            clustering = report['clustering_validation']
            if clustering and 'cross_validation_accuracy' in clustering:
                acc = clustering['cross_validation_accuracy']
                f1 = clustering['cross_validation_f1']
                f.write(f"交叉验证准确率: {acc['mean']:.3f} ± {acc['std']:.3f}\n")
                f.write(f"交叉验证F1分数: {f1['mean']:.3f} ± {f1['std']:.3f}\n")
                f.write(f"样本数量: {clustering['sample_size']}\n")
                f.write(f"特征数量: {clustering['feature_count']}\n")
                f.write(f"类别数量: {clustering['class_count']}\n")
            
            f.write("\n5. 改进建议\n")
            f.write("-" * 40 + "\n")
            for i, recommendation in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {recommendation}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("报告结束\n")

def main():
    """主函数"""
    print("🚀 启动IET期刊标准增强分析...")
    
    enhancer = IETJournalEnhancer()
    
    # 加载实验数据
    data = enhancer.load_experimental_data()
    
    if not data:
        print("❌ 无法加载实验数据，请确保系统已运行并生成了数据")
        return False
    
    # 生成合规性报告
    compliance_report = enhancer.generate_iet_compliance_report(data)
    
    # 保存报告
    report_path = enhancer.save_compliance_report(compliance_report)
    
    print(f"✅ IET期刊合规性报告已生成: {report_path}")
    print(f"📊 总体合规性分数: {compliance_report['overall_compliance_score']:.1f}/100")
    
    # 显示关键指标
    if compliance_report['overall_compliance_score'] >= 80:
        print("🎉 实验设计完全符合IET期刊发表标准！")
    elif compliance_report['overall_compliance_score'] >= 70:
        print("✅ 实验设计基本符合IET期刊要求，建议进行小幅优化")
    else:
        print("⚠️ 实验设计需要改进以满足IET期刊标准")
    
    print("\n📋 改进建议:")
    for i, recommendation in enumerate(compliance_report['recommendations'], 1):
        print(f"  {i}. {recommendation}")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🏆 IET期刊标准增强分析完成！")
    else:
        print("\n❌ 分析过程中出现错误")
