# Comments for context:
# This module provides business intelligence capabilities for strategic port simulations.
# It calculates advanced KPIs, generates executive reports, performs trend analysis,
# and provides data-driven insights for strategic decision-making.
#
# The module is designed to work with the StrategicSimulationController to provide
# real-time business analytics and comprehensive reporting for executives and
# strategic planners. It focuses on translating operational metrics into
# business value and strategic insights.

from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json

class KPICategory(Enum):
    """Categories of Key Performance Indicators."""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    CUSTOMER = "customer"
    COMPETITIVE = "competitive"
    SUSTAINABILITY = "sustainability"

class TrendDirection(Enum):
    """Trend direction indicators."""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"

@dataclass
class KPIMetric:
    """Individual KPI metric with metadata."""
    name: str
    value: float
    unit: str
    category: KPICategory
    target: Optional[float] = None
    benchmark: Optional[float] = None
    trend: Optional[TrendDirection] = None
    variance_percentage: Optional[float] = None
    description: str = ""
    
    @property
    def performance_vs_target(self) -> Optional[float]:
        """Calculate performance vs target as percentage."""
        if self.target is None or self.target == 0:
            return None
        return (self.value / self.target) * 100
    
    @property
    def performance_vs_benchmark(self) -> Optional[float]:
        """Calculate performance vs benchmark as percentage."""
        if self.benchmark is None or self.benchmark == 0:
            return None
        return (self.value / self.benchmark) * 100

@dataclass
class BusinessInsight:
    """Business insight with actionable recommendations."""
    title: str
    description: str
    impact_level: str  # "high", "medium", "low"
    confidence_score: float  # 0-100
    recommendations: List[str]
    supporting_data: Dict[str, Any]
    category: KPICategory

@dataclass
class ExecutiveSummary:
    """Executive summary for strategic reporting."""
    period: str
    scenario_name: str
    key_achievements: List[str]
    critical_issues: List[str]
    financial_highlights: Dict[str, float]
    operational_highlights: Dict[str, float]
    strategic_recommendations: List[str]
    next_quarter_outlook: str
    risk_assessment: Dict[str, float]

class BusinessIntelligenceEngine:
    """Advanced business intelligence engine for strategic port analytics.
    
    This engine provides comprehensive business analytics capabilities including:
    - Real-time KPI calculation and monitoring
    - Trend analysis and forecasting
    - Competitive benchmarking
    - Executive reporting and insights
    - ROI and financial analysis
    - Strategic recommendation generation
    """
    
    def __init__(self):
        """Initialize the business intelligence engine."""
        self.logger = logging.getLogger(__name__)
        
        # KPI definitions and targets
        self.kpi_definitions = self._initialize_kpi_definitions()
        
        # Historical data storage
        self.metrics_history: List[Dict[str, Any]] = []
        self.kpi_history: List[Dict[str, KPIMetric]] = []
        
        # Benchmarks and targets
        self.industry_benchmarks = self._load_industry_benchmarks()
        self.strategic_targets = self._load_strategic_targets()
        
        # Analysis parameters
        self.trend_analysis_window = 30  # days
        self.volatility_threshold = 0.15  # 15%
        self.confidence_threshold = 0.7  # 70%
    
    def calculate_kpis(self, 
                      business_metrics: Dict[str, Any], 
                      operational_metrics: Dict[str, Any],
                      scenario_context: Optional[Dict[str, Any]] = None) -> Dict[str, KPIMetric]:
        """Calculate comprehensive KPIs from simulation metrics.
        
        Args:
            business_metrics: Business metrics from strategic simulation
            operational_metrics: Operational metrics from base simulation
            scenario_context: Additional scenario context
            
        Returns:
            dict: Calculated KPIs organized by category
        """
        kpis = {}
        
        # Financial KPIs
        kpis.update(self._calculate_financial_kpis(business_metrics, operational_metrics))
        
        # Operational KPIs
        kpis.update(self._calculate_operational_kpis(business_metrics, operational_metrics))
        
        # Strategic KPIs
        kpis.update(self._calculate_strategic_kpis(business_metrics, operational_metrics, scenario_context))
        
        # Customer KPIs
        kpis.update(self._calculate_customer_kpis(business_metrics, operational_metrics))
        
        # Competitive KPIs
        kpis.update(self._calculate_competitive_kpis(business_metrics, operational_metrics))
        
        # Calculate trends and variances
        self._calculate_trends_and_variances(kpis)
        
        # Store in history
        self.kpi_history.append(kpis.copy())
        
        return kpis
    
    def generate_business_insights(self, 
                                 kpis: Dict[str, KPIMetric],
                                 scenario_context: Optional[Dict[str, Any]] = None) -> List[BusinessInsight]:
        """Generate actionable business insights from KPIs.
        
        Args:
            kpis: Calculated KPIs
            scenario_context: Additional scenario context
            
        Returns:
            list: Business insights with recommendations
        """
        insights = []
        
        # Financial insights
        insights.extend(self._analyze_financial_performance(kpis))
        
        # Operational insights
        insights.extend(self._analyze_operational_efficiency(kpis))
        
        # Strategic insights
        insights.extend(self._analyze_strategic_positioning(kpis, scenario_context))
        
        # Competitive insights
        insights.extend(self._analyze_competitive_advantage(kpis))
        
        # Risk insights
        insights.extend(self._analyze_risk_factors(kpis))
        
        # Sort by impact and confidence
        insights.sort(key=lambda x: (x.impact_level == "high", x.confidence_score), reverse=True)
        
        return insights
    
    def create_executive_summary(self, 
                               kpis: Dict[str, KPIMetric],
                               insights: List[BusinessInsight],
                               scenario_name: str,
                               period: str = "Current Quarter") -> ExecutiveSummary:
        """Create executive summary for strategic reporting.
        
        Args:
            kpis: Calculated KPIs
            insights: Business insights
            scenario_name: Name of the strategic scenario
            period: Reporting period
            
        Returns:
            ExecutiveSummary: Comprehensive executive summary
        """
        # Extract key achievements
        key_achievements = self._extract_key_achievements(kpis, insights)
        
        # Identify critical issues
        critical_issues = self._identify_critical_issues(kpis, insights)
        
        # Financial highlights
        financial_highlights = {
            "revenue_per_hour": kpis.get("revenue_per_hour", KPIMetric("", 0, "", KPICategory.FINANCIAL)).value,
            "roi_percentage": kpis.get("roi_percentage", KPIMetric("", 0, "", KPICategory.FINANCIAL)).value,
            "cost_efficiency": kpis.get("cost_efficiency", KPIMetric("", 0, "", KPICategory.FINANCIAL)).value
        }
        
        # Operational highlights
        operational_highlights = {
            "throughput_efficiency": kpis.get("throughput_efficiency", KPIMetric("", 0, "", KPICategory.OPERATIONAL)).value,
            "berth_utilization": kpis.get("berth_utilization", KPIMetric("", 0, "", KPICategory.OPERATIONAL)).value,
            "processing_efficiency": kpis.get("processing_efficiency", KPIMetric("", 0, "", KPICategory.OPERATIONAL)).value
        }
        
        # Strategic recommendations
        strategic_recommendations = [insight.recommendations[0] for insight in insights[:5] if insight.recommendations]
        
        # Risk assessment
        risk_assessment = self._calculate_risk_assessment(kpis, insights)
        
        # Next quarter outlook
        next_quarter_outlook = self._generate_outlook(kpis, insights)
        
        return ExecutiveSummary(
            period=period,
            scenario_name=scenario_name,
            key_achievements=key_achievements,
            critical_issues=critical_issues,
            financial_highlights=financial_highlights,
            operational_highlights=operational_highlights,
            strategic_recommendations=strategic_recommendations,
            next_quarter_outlook=next_quarter_outlook,
            risk_assessment=risk_assessment
        )
    
    def perform_trend_analysis(self, 
                             metric_name: str, 
                             window_days: Optional[int] = None) -> Dict[str, Any]:
        """Perform trend analysis on historical metrics.
        
        Args:
            metric_name: Name of the metric to analyze
            window_days: Analysis window in days
            
        Returns:
            dict: Trend analysis results
        """
        window = window_days or self.trend_analysis_window
        
        if len(self.kpi_history) < 2:
            return {"trend": TrendDirection.STABLE, "confidence": 0.0, "message": "Insufficient data"}
        
        # Extract metric values from history
        values = []
        for kpi_snapshot in self.kpi_history[-window:]:
            if metric_name in kpi_snapshot:
                values.append(kpi_snapshot[metric_name].value)
        
        if len(values) < 2:
            return {"trend": TrendDirection.STABLE, "confidence": 0.0, "message": "Insufficient data"}
        
        # Calculate trend
        trend_analysis = self._calculate_trend(values)
        
        return trend_analysis
    
    def generate_forecast(self, 
                        metric_name: str, 
                        forecast_periods: int = 30) -> Dict[str, Any]:
        """Generate forecast for a specific metric.
        
        Args:
            metric_name: Name of the metric to forecast
            forecast_periods: Number of periods to forecast
            
        Returns:
            dict: Forecast results with confidence intervals
        """
        if len(self.kpi_history) < 10:
            return {"forecast": [], "confidence": 0.0, "message": "Insufficient historical data"}
        
        # Extract historical values
        values = []
        for kpi_snapshot in self.kpi_history:
            if metric_name in kpi_snapshot:
                values.append(kpi_snapshot[metric_name].value)
        
        if len(values) < 10:
            return {"forecast": [], "confidence": 0.0, "message": "Insufficient historical data"}
        
        # Simple linear regression forecast
        forecast_results = self._generate_linear_forecast(values, forecast_periods)
        
        return forecast_results
    
    def export_dashboard_data(self, 
                            kpis: Dict[str, KPIMetric],
                            insights: List[BusinessInsight]) -> Dict[str, Any]:
        """Export data formatted for dashboard visualization.
        
        Args:
            kpis: Calculated KPIs
            insights: Business insights
            
        Returns:
            dict: Dashboard-ready data structure
        """
        # Organize KPIs by category
        kpis_by_category = {}
        for category in KPICategory:
            kpis_by_category[category.value] = [
                {
                    "name": kpi.name,
                    "value": kpi.value,
                    "unit": kpi.unit,
                    "target": kpi.target,
                    "performance_vs_target": kpi.performance_vs_target,
                    "trend": kpi.trend.value if kpi.trend else None,
                    "description": kpi.description
                }
                for kpi in kpis.values() if kpi.category == category
            ]
        
        # Format insights for dashboard
        dashboard_insights = [
            {
                "title": insight.title,
                "description": insight.description,
                "impact_level": insight.impact_level,
                "confidence_score": insight.confidence_score,
                "recommendations": insight.recommendations[:3],  # Top 3 recommendations
                "category": insight.category.value
            }
            for insight in insights[:10]  # Top 10 insights
        ]
        
        # Create summary metrics for quick overview
        summary_metrics = self._create_summary_metrics(kpis)
        
        return {
            "kpis_by_category": kpis_by_category,
            "insights": dashboard_insights,
            "summary_metrics": summary_metrics,
            "last_updated": datetime.now().isoformat(),
            "data_quality_score": self._calculate_data_quality_score(kpis)
        }
    
    # Private methods for KPI calculations
    
    def _initialize_kpi_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize KPI definitions with metadata."""
        return {
            "revenue_per_hour": {
                "category": KPICategory.FINANCIAL,
                "unit": "USD/hour",
                "description": "Revenue generated per hour of operation",
                "target_multiplier": 1.0
            },
            "roi_percentage": {
                "category": KPICategory.FINANCIAL,
                "unit": "%",
                "description": "Return on investment percentage",
                "target_multiplier": 1.0
            },
            "throughput_efficiency": {
                "category": KPICategory.OPERATIONAL,
                "unit": "TEU/hour",
                "description": "Container throughput efficiency",
                "target_multiplier": 1.0
            },
            "berth_utilization": {
                "category": KPICategory.OPERATIONAL,
                "unit": "%",
                "description": "Berth utilization percentage",
                "target_multiplier": 1.0
            },
            "customer_satisfaction": {
                "category": KPICategory.CUSTOMER,
                "unit": "Score",
                "description": "Customer satisfaction score",
                "target_multiplier": 1.0
            }
        }
    
    def _load_industry_benchmarks(self) -> Dict[str, float]:
        """Load industry benchmark values."""
        # These would typically be loaded from a database or configuration file
        return {
            "revenue_per_hour": 25000.0,
            "roi_percentage": 15.0,
            "throughput_efficiency": 35.0,
            "berth_utilization": 75.0,
            "customer_satisfaction": 8.5
        }
    
    def _load_strategic_targets(self) -> Dict[str, float]:
        """Load strategic target values."""
        # These would typically be loaded from strategic planning documents
        return {
            "revenue_per_hour": 30000.0,
            "roi_percentage": 20.0,
            "throughput_efficiency": 40.0,
            "berth_utilization": 85.0,
            "customer_satisfaction": 9.0
        }
    
    def _calculate_financial_kpis(self, 
                                business_metrics: Dict[str, Any], 
                                operational_metrics: Dict[str, Any]) -> Dict[str, KPIMetric]:
        """Calculate financial KPIs."""
        kpis = {}
        
        # Revenue per hour
        revenue_per_hour = business_metrics.get("revenue_per_hour", 0)
        kpis["revenue_per_hour"] = KPIMetric(
            name="Revenue per Hour",
            value=revenue_per_hour,
            unit="USD/hour",
            category=KPICategory.FINANCIAL,
            target=self.strategic_targets.get("revenue_per_hour"),
            benchmark=self.industry_benchmarks.get("revenue_per_hour"),
            description="Revenue generated per hour of operation"
        )
        
        # ROI percentage
        roi_percentage = business_metrics.get("roi_percentage", 0)
        kpis["roi_percentage"] = KPIMetric(
            name="Return on Investment",
            value=roi_percentage,
            unit="%",
            category=KPICategory.FINANCIAL,
            target=self.strategic_targets.get("roi_percentage"),
            benchmark=self.industry_benchmarks.get("roi_percentage"),
            description="Return on investment percentage"
        )
        
        # Cost efficiency
        total_revenue = business_metrics.get("total_revenue", 0)
        operational_cost = business_metrics.get("operational_cost", 1)
        cost_efficiency = (total_revenue / operational_cost) * 100 if operational_cost > 0 else 0
        kpis["cost_efficiency"] = KPIMetric(
            name="Cost Efficiency",
            value=cost_efficiency,
            unit="%",
            category=KPICategory.FINANCIAL,
            description="Revenue to cost ratio percentage"
        )
        
        return kpis
    
    def _calculate_operational_kpis(self, 
                                  business_metrics: Dict[str, Any], 
                                  operational_metrics: Dict[str, Any]) -> Dict[str, KPIMetric]:
        """Calculate operational KPIs."""
        kpis = {}
        
        # Throughput efficiency
        throughput_efficiency = business_metrics.get("throughput_teu_per_hour", 0)
        kpis["throughput_efficiency"] = KPIMetric(
            name="Throughput Efficiency",
            value=throughput_efficiency,
            unit="TEU/hour",
            category=KPICategory.OPERATIONAL,
            target=self.strategic_targets.get("throughput_efficiency"),
            benchmark=self.industry_benchmarks.get("throughput_efficiency"),
            description="Container throughput per hour"
        )
        
        # Berth utilization
        berth_utilization = business_metrics.get("berth_utilization_percentage", 0)
        kpis["berth_utilization"] = KPIMetric(
            name="Berth Utilization",
            value=berth_utilization,
            unit="%",
            category=KPICategory.OPERATIONAL,
            target=self.strategic_targets.get("berth_utilization"),
            benchmark=self.industry_benchmarks.get("berth_utilization"),
            description="Percentage of berth capacity utilized"
        )
        
        # Processing efficiency
        processing_efficiency = business_metrics.get("processing_efficiency", 0)
        kpis["processing_efficiency"] = KPIMetric(
            name="Processing Efficiency",
            value=processing_efficiency,
            unit="%",
            category=KPICategory.OPERATIONAL,
            description="Overall processing efficiency percentage"
        )
        
        return kpis
    
    def _calculate_strategic_kpis(self, 
                                business_metrics: Dict[str, Any], 
                                operational_metrics: Dict[str, Any],
                                scenario_context: Optional[Dict[str, Any]]) -> Dict[str, KPIMetric]:
        """Calculate strategic KPIs."""
        kpis = {}
        
        # Capacity utilization
        capacity_utilization = business_metrics.get("capacity_utilization", 0)
        kpis["capacity_utilization"] = KPIMetric(
            name="Capacity Utilization",
            value=capacity_utilization,
            unit="%",
            category=KPICategory.STRATEGIC,
            description="Strategic capacity utilization percentage"
        )
        
        # AI optimization benefit
        ai_optimization_benefit = business_metrics.get("ai_optimization_benefit", 0)
        kpis["ai_optimization_benefit"] = KPIMetric(
            name="AI Optimization Benefit",
            value=ai_optimization_benefit,
            unit="%",
            category=KPICategory.STRATEGIC,
            description="Benefit gained from AI optimization"
        )
        
        return kpis
    
    def _calculate_customer_kpis(self, 
                               business_metrics: Dict[str, Any], 
                               operational_metrics: Dict[str, Any]) -> Dict[str, KPIMetric]:
        """Calculate customer-focused KPIs."""
        kpis = {}
        
        # Customer satisfaction (derived from waiting times)
        avg_waiting_time = business_metrics.get("average_waiting_time_hours", 0)
        # Convert waiting time to satisfaction score (inverse relationship)
        customer_satisfaction = max(0, 10 - (avg_waiting_time * 2)) if avg_waiting_time > 0 else 8.5
        kpis["customer_satisfaction"] = KPIMetric(
            name="Customer Satisfaction",
            value=customer_satisfaction,
            unit="Score",
            category=KPICategory.CUSTOMER,
            target=self.strategic_targets.get("customer_satisfaction"),
            benchmark=self.industry_benchmarks.get("customer_satisfaction"),
            description="Customer satisfaction score based on service quality"
        )
        
        return kpis
    
    def _calculate_competitive_kpis(self, 
                                  business_metrics: Dict[str, Any], 
                                  operational_metrics: Dict[str, Any]) -> Dict[str, KPIMetric]:
        """Calculate competitive positioning KPIs."""
        kpis = {}
        
        # Competitive advantage score
        competitive_advantage = business_metrics.get("competitive_advantage_score", 0)
        kpis["competitive_advantage"] = KPIMetric(
            name="Competitive Advantage",
            value=competitive_advantage,
            unit="Score",
            category=KPICategory.COMPETITIVE,
            description="Competitive positioning score"
        )
        
        return kpis
    
    def _calculate_trends_and_variances(self, kpis: Dict[str, KPIMetric]):
        """Calculate trends and variances for KPIs."""
        if len(self.kpi_history) < 2:
            return
        
        for kpi_name, kpi in kpis.items():
            # Calculate trend
            trend_analysis = self.perform_trend_analysis(kpi_name, window_days=7)
            kpi.trend = trend_analysis.get("trend", TrendDirection.STABLE)
            
            # Calculate variance
            if len(self.kpi_history) >= 2:
                previous_value = self.kpi_history[-1].get(kpi_name)
                if previous_value:
                    variance = ((kpi.value - previous_value.value) / previous_value.value) * 100 if previous_value.value != 0 else 0
                    kpi.variance_percentage = variance
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend direction and confidence."""
        if len(values) < 2:
            return {"trend": TrendDirection.STABLE, "confidence": 0.0}
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Calculate R-squared for confidence
        correlation = np.corrcoef(x, y)[0, 1]
        r_squared = correlation ** 2
        
        # Determine trend direction
        if abs(slope) < (np.std(values) * 0.1):  # Small change relative to standard deviation
            trend = TrendDirection.STABLE
        elif slope > 0:
            trend = TrendDirection.IMPROVING
        else:
            trend = TrendDirection.DECLINING
        
        # Check for volatility
        cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        if cv > self.volatility_threshold:
            trend = TrendDirection.VOLATILE
        
        return {
            "trend": trend,
            "confidence": r_squared,
            "slope": slope,
            "volatility": cv
        }
    
    def _generate_linear_forecast(self, values: List[float], periods: int) -> Dict[str, Any]:
        """Generate linear forecast for given values."""
        x = np.arange(len(values))
        y = np.array(values)
        
        # Fit linear model
        coeffs = np.polyfit(x, y, 1)
        
        # Generate forecast
        forecast_x = np.arange(len(values), len(values) + periods)
        forecast_y = np.polyval(coeffs, forecast_x)
        
        # Calculate confidence (simplified)
        residuals = y - np.polyval(coeffs, x)
        mse = np.mean(residuals ** 2)
        confidence = max(0, 1 - (mse / np.var(y))) if np.var(y) > 0 else 0
        
        return {
            "forecast": forecast_y.tolist(),
            "confidence": confidence,
            "trend_slope": coeffs[0],
            "periods": periods
        }
    
    # Private methods for insight generation
    
    def _analyze_financial_performance(self, kpis: Dict[str, KPIMetric]) -> List[BusinessInsight]:
        """Analyze financial performance and generate insights."""
        insights = []
        
        # ROI analysis
        roi_kpi = kpis.get("roi_percentage")
        if roi_kpi and roi_kpi.target:
            if roi_kpi.performance_vs_target and roi_kpi.performance_vs_target > 110:
                insights.append(BusinessInsight(
                    title="Exceptional ROI Performance",
                    description=f"ROI of {roi_kpi.value:.1f}% exceeds target by {roi_kpi.performance_vs_target - 100:.1f}%",
                    impact_level="high",
                    confidence_score=90.0,
                    recommendations=[
                        "Scale successful strategies to other operational areas",
                        "Document best practices for replication",
                        "Consider increasing investment in high-ROI initiatives"
                    ],
                    supporting_data={"roi_value": roi_kpi.value, "target": roi_kpi.target},
                    category=KPICategory.FINANCIAL
                ))
        
        return insights
    
    def _analyze_operational_efficiency(self, kpis: Dict[str, KPIMetric]) -> List[BusinessInsight]:
        """Analyze operational efficiency and generate insights."""
        insights = []
        
        # Throughput analysis
        throughput_kpi = kpis.get("throughput_efficiency")
        if throughput_kpi and throughput_kpi.benchmark:
            if throughput_kpi.performance_vs_benchmark and throughput_kpi.performance_vs_benchmark > 120:
                insights.append(BusinessInsight(
                    title="Superior Throughput Performance",
                    description=f"Throughput efficiency of {throughput_kpi.value:.1f} TEU/hour exceeds industry benchmark",
                    impact_level="high",
                    confidence_score=85.0,
                    recommendations=[
                        "Market competitive advantage in efficiency",
                        "Consider premium pricing for expedited services",
                        "Invest in additional capacity to capture market share"
                    ],
                    supporting_data={"throughput": throughput_kpi.value, "benchmark": throughput_kpi.benchmark},
                    category=KPICategory.OPERATIONAL
                ))
        
        return insights
    
    def _analyze_strategic_positioning(self, 
                                     kpis: Dict[str, KPIMetric],
                                     scenario_context: Optional[Dict[str, Any]]) -> List[BusinessInsight]:
        """Analyze strategic positioning and generate insights."""
        insights = []
        
        # Capacity utilization analysis
        capacity_kpi = kpis.get("capacity_utilization")
        if capacity_kpi and capacity_kpi.value > 90:
            insights.append(BusinessInsight(
                title="High Capacity Utilization Alert",
                description=f"Capacity utilization at {capacity_kpi.value:.1f}% approaching maximum",
                impact_level="high",
                confidence_score=95.0,
                recommendations=[
                    "Plan for capacity expansion to avoid bottlenecks",
                    "Implement dynamic pricing during peak periods",
                    "Optimize scheduling to distribute load"
                ],
                supporting_data={"utilization": capacity_kpi.value},
                category=KPICategory.STRATEGIC
            ))
        
        return insights
    
    def _analyze_competitive_advantage(self, kpis: Dict[str, KPIMetric]) -> List[BusinessInsight]:
        """Analyze competitive advantage and generate insights."""
        insights = []
        
        # Competitive positioning
        competitive_kpi = kpis.get("competitive_advantage")
        if competitive_kpi and competitive_kpi.value > 8.0:
            insights.append(BusinessInsight(
                title="Strong Competitive Position",
                description=f"Competitive advantage score of {competitive_kpi.value:.1f} indicates market leadership",
                impact_level="medium",
                confidence_score=80.0,
                recommendations=[
                    "Leverage competitive advantage in marketing",
                    "Maintain technology leadership through continued investment",
                    "Explore strategic partnerships to strengthen position"
                ],
                supporting_data={"competitive_score": competitive_kpi.value},
                category=KPICategory.COMPETITIVE
            ))
        
        return insights
    
    def _analyze_risk_factors(self, kpis: Dict[str, KPIMetric]) -> List[BusinessInsight]:
        """Analyze risk factors and generate insights."""
        insights = []
        
        # Check for declining trends
        declining_kpis = [kpi for kpi in kpis.values() if kpi.trend == TrendDirection.DECLINING]
        if len(declining_kpis) > 2:
            insights.append(BusinessInsight(
                title="Multiple Declining Performance Indicators",
                description=f"{len(declining_kpis)} KPIs showing declining trends",
                impact_level="high",
                confidence_score=85.0,
                recommendations=[
                    "Conduct comprehensive performance review",
                    "Implement corrective action plans",
                    "Increase monitoring frequency for declining metrics"
                ],
                supporting_data={"declining_count": len(declining_kpis)},
                category=KPICategory.STRATEGIC
            ))
        
        return insights
    
    # Private methods for executive summary
    
    def _extract_key_achievements(self, 
                                kpis: Dict[str, KPIMetric], 
                                insights: List[BusinessInsight]) -> List[str]:
        """Extract key achievements from KPIs and insights."""
        achievements = []
        
        # High-performing KPIs
        for kpi in kpis.values():
            if kpi.performance_vs_target and kpi.performance_vs_target > 110:
                achievements.append(f"{kpi.name} exceeded target by {kpi.performance_vs_target - 100:.1f}%")
        
        # High-impact positive insights
        positive_insights = [i for i in insights if i.impact_level == "high" and "exceed" in i.description.lower()]
        for insight in positive_insights[:3]:
            achievements.append(insight.title)
        
        return achievements[:5]  # Top 5 achievements
    
    def _identify_critical_issues(self, 
                                kpis: Dict[str, KPIMetric], 
                                insights: List[BusinessInsight]) -> List[str]:
        """Identify critical issues from KPIs and insights."""
        issues = []
        
        # Underperforming KPIs
        for kpi in kpis.values():
            if kpi.performance_vs_target and kpi.performance_vs_target < 90:
                issues.append(f"{kpi.name} below target by {100 - kpi.performance_vs_target:.1f}%")
        
        # High-impact negative insights
        negative_insights = [i for i in insights if i.impact_level == "high" and any(word in i.description.lower() for word in ["alert", "declining", "risk"])]
        for insight in negative_insights[:3]:
            issues.append(insight.title)
        
        return issues[:5]  # Top 5 issues
    
    def _calculate_risk_assessment(self, 
                                 kpis: Dict[str, KPIMetric], 
                                 insights: List[BusinessInsight]) -> Dict[str, float]:
        """Calculate risk assessment scores."""
        # Simplified risk calculation based on KPI performance and trends
        financial_risk = 0.0
        operational_risk = 0.0
        strategic_risk = 0.0
        
        for kpi in kpis.values():
            risk_score = 0.0
            
            # Performance vs target risk
            if kpi.performance_vs_target and kpi.performance_vs_target < 90:
                risk_score += (90 - kpi.performance_vs_target) / 10
            
            # Trend risk
            if kpi.trend == TrendDirection.DECLINING:
                risk_score += 2.0
            elif kpi.trend == TrendDirection.VOLATILE:
                risk_score += 1.5
            
            # Assign to category
            if kpi.category == KPICategory.FINANCIAL:
                financial_risk += risk_score
            elif kpi.category == KPICategory.OPERATIONAL:
                operational_risk += risk_score
            elif kpi.category == KPICategory.STRATEGIC:
                strategic_risk += risk_score
        
        # Normalize to 0-100 scale
        return {
            "financial_risk": min(100, financial_risk * 10),
            "operational_risk": min(100, operational_risk * 10),
            "strategic_risk": min(100, strategic_risk * 10)
        }
    
    def _generate_outlook(self, 
                        kpis: Dict[str, KPIMetric], 
                        insights: List[BusinessInsight]) -> str:
        """Generate next quarter outlook."""
        positive_trends = sum(1 for kpi in kpis.values() if kpi.trend == TrendDirection.IMPROVING)
        negative_trends = sum(1 for kpi in kpis.values() if kpi.trend == TrendDirection.DECLINING)
        
        if positive_trends > negative_trends:
            return "Positive outlook with improving performance trends across key metrics. Continued focus on operational excellence expected to drive growth."
        elif negative_trends > positive_trends:
            return "Cautious outlook due to declining trends in several key areas. Immediate attention required to address performance gaps."
        else:
            return "Stable outlook with mixed performance indicators. Strategic initiatives should focus on strengthening competitive position."
    
    def _create_summary_metrics(self, kpis: Dict[str, KPIMetric]) -> Dict[str, Any]:
        """Create summary metrics for dashboard overview."""
        total_kpis = len(kpis)
        on_target = sum(1 for kpi in kpis.values() if kpi.performance_vs_target and kpi.performance_vs_target >= 100)
        improving = sum(1 for kpi in kpis.values() if kpi.trend == TrendDirection.IMPROVING)
        
        return {
            "total_kpis": total_kpis,
            "on_target_percentage": (on_target / total_kpis * 100) if total_kpis > 0 else 0,
            "improving_percentage": (improving / total_kpis * 100) if total_kpis > 0 else 0,
            "overall_health_score": (on_target + improving) / (total_kpis * 2) * 100 if total_kpis > 0 else 0
        }
    
    def _calculate_data_quality_score(self, kpis: Dict[str, KPIMetric]) -> float:
        """Calculate data quality score for the dashboard."""
        if not kpis:
            return 0.0
        
        # Simple data quality based on completeness
        complete_kpis = sum(1 for kpi in kpis.values() if kpi.value is not None and kpi.value != 0)
        return (complete_kpis / len(kpis)) * 100