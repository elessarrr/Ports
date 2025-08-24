import unittest
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from hk_port_digital_twin.src.scenarios.investment_planner import (
    InvestmentPlanner, InvestmentType, InvestmentPriority, DemandProjection, InvestmentOption, InvestmentScenario
)

class TestInvestmentPlanner(unittest.TestCase):
    """Test cases for the InvestmentPlanner class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.planner = InvestmentPlanner()
    
    def test_investment_planner_initialization(self):
        """Test that InvestmentPlanner initializes correctly"""
        self.assertIsInstance(self.planner, InvestmentPlanner)
        self.assertIsInstance(self.planner.investment_options, dict)
        self.assertIsInstance(self.planner.demand_projections, list)
        self.assertIsInstance(self.planner.scenarios, dict)
        # InvestmentPlanner comes pre-populated with default options
        self.assertGreater(len(self.planner.investment_options), 0)
        self.assertGreater(len(self.planner.demand_projections), 0)
        self.assertEqual(len(self.planner.scenarios), 0)
    
    def test_create_investment_option(self):
        """Test creating an investment option"""
        option = InvestmentOption(
            investment_id="test_berth",
            name="New Container Berth",
            investment_type=InvestmentType.NEW_BERTH,
            priority=InvestmentPriority.HIGH,
            capital_cost=50_000_000,
            annual_operating_cost=2_000_000,
            implementation_time_months=24,
            capacity_increase=500_000,
            efficiency_improvement=0.15,
            lifespan_years=25
        )
        
        self.assertIsInstance(option, InvestmentOption)
        self.assertEqual(option.name, "New Container Berth")
        self.assertEqual(option.investment_type, InvestmentType.NEW_BERTH)
        self.assertEqual(option.capital_cost, 50_000_000)
        self.assertEqual(option.capacity_increase, 500_000)
        self.assertEqual(option.efficiency_improvement, 0.15)
        self.assertEqual(option.implementation_time_months, 24)
        self.assertEqual(option.priority, InvestmentPriority.HIGH)
    
    def test_create_demand_projection(self):
        """Test creating demand projections"""
        projection = DemandProjection(
            year=2024,
            vessel_arrivals_per_day=25.0,
            average_vessel_size=8000.0,
            growth_rate=0.05,
            peak_season_multiplier=1.3,
            confidence_level=0.8
        )
        
        self.assertIsInstance(projection, DemandProjection)
        self.assertEqual(projection.year, 2024)
        self.assertEqual(projection.vessel_arrivals_per_day, 25.0)
        self.assertEqual(projection.growth_rate, 0.05)
        self.assertEqual(projection.peak_season_multiplier, 1.3)
    
    def test_calculate_roi(self):
        """Test ROI calculation for investment scenarios"""
        # Create a test scenario using existing investment options
        scenario = self.planner.create_investment_scenario(
            scenario_id="test_roi_scenario",
            name="Test ROI Scenario",
            description="Test scenario for ROI calculation",
            investment_ids=["crane_upgrade_1"]
        )
        
        roi_analysis = self.planner.calculate_roi_analysis(scenario, analysis_period_years=10)
        
        self.assertIsInstance(roi_analysis, dict)
        self.assertIn('financial_metrics', roi_analysis)
        self.assertIn('capacity_metrics', roi_analysis)
        self.assertIn('risk_assessment', roi_analysis)
        
        financial_metrics = roi_analysis['financial_metrics']
        self.assertIn('npv', financial_metrics)
        self.assertIn('irr', financial_metrics)
        self.assertIn('payback_period', financial_metrics)
        self.assertIsInstance(financial_metrics['npv'], (int, float))
        self.assertIsInstance(financial_metrics['irr'], (int, float))
        self.assertIsInstance(financial_metrics['payback_period'], (int, float))
    
    def test_calculate_npv(self):
        """Test NPV calculation through financial metrics"""
        # Create a scenario and analyze it to test NPV calculation
        scenario = self.planner.create_investment_scenario(
            scenario_id="test_npv_scenario",
            name="Test NPV Scenario",
            description="Test scenario for NPV calculation",
            investment_ids=["storage_expansion_1"]
        )
        
        analysis = self.planner.calculate_roi_analysis(scenario)
        financial_metrics = analysis['financial_metrics']
        
        self.assertIn('npv', financial_metrics)
        self.assertIsInstance(financial_metrics['npv'], (int, float))
        # NPV can be positive or negative depending on the investment
    
    def test_calculate_payback_period(self):
        """Test payback period calculation through financial metrics"""
        # Create a scenario and analyze it to test payback period calculation
        scenario = self.planner.create_investment_scenario(
            scenario_id="test_payback_scenario",
            name="Test Payback Scenario",
            description="Test scenario for payback period calculation",
            investment_ids=["digital_infra_1"]
        )
        
        analysis = self.planner.calculate_roi_analysis(scenario)
        financial_metrics = analysis['financial_metrics']
        
        self.assertIn('payback_period', financial_metrics)
        self.assertIsInstance(financial_metrics['payback_period'], (int, float))
        self.assertGreater(financial_metrics['payback_period'], 0)
    
    def test_project_demand_growth(self):
        """Test demand growth projection through demand projections"""
        # Test the existing demand projections
        self.assertGreater(len(self.planner.demand_projections), 0)
        
        # Check that projections show growth over time
        projections = self.planner.demand_projections
        
        # Verify that each projection has the required attributes
        for projection in projections:
            self.assertIsInstance(projection.year, int)
            self.assertIsInstance(projection.vessel_arrivals_per_day, float)
            self.assertIsInstance(projection.average_vessel_size, float)
            self.assertIsInstance(projection.growth_rate, float)
            
        # Check that annual throughput can be calculated
        first_projection = projections[0]
        annual_throughput = first_projection.get_annual_throughput()
        self.assertIsInstance(annual_throughput, float)
        self.assertGreater(annual_throughput, 0)
    
    def test_analyze_investment_scenario(self):
        """Test complete investment scenario analysis"""
        # Create a scenario using existing investment options
        scenario = self.planner.create_investment_scenario(
            scenario_id="test_analysis_scenario",
            name="Expansion Scenario",
            description="Major port expansion",
            investment_ids=["new_berth_1", "crane_upgrade_1"]
        )
        
        analysis = self.planner.calculate_roi_analysis(scenario)
        
        self.assertIsInstance(analysis, dict)
        
        # Check required analysis fields
        required_fields = [
            'scenario', 'financial_metrics', 'capacity_metrics',
            'risk_assessment', 'recommendations', 'cash_flows'
        ]
        
        for field in required_fields:
            self.assertIn(field, analysis)
        
        # Verify financial analysis
        financial = analysis['financial_metrics']
        self.assertIn('npv', financial)
        self.assertIn('irr', financial)
        self.assertIn('payback_period', financial)
        
        # Verify capacity analysis
        capacity = analysis['capacity_metrics']
        self.assertIn('final_additional_capacity', capacity)
        self.assertIn('average_utilization', capacity)
        
        # Verify recommendations
        recommendations = analysis['recommendations']
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
    
    def test_compare_investment_scenarios(self):
        """Test comparing multiple investment scenarios"""
        # Create multiple scenarios using existing investment options
        scenario1 = self.planner.create_investment_scenario(
            scenario_id="conservative_scenario",
            name="Conservative Scenario",
            description="Low-risk investments",
            investment_ids=["storage_expansion_1"]
        )
        
        scenario2 = self.planner.create_investment_scenario(
            scenario_id="aggressive_scenario",
            name="Aggressive Scenario",
            description="High-growth investments",
            investment_ids=["new_berth_1", "automation_1"]
        )
        
        comparison = self.planner.compare_investment_scenarios(["conservative_scenario", "aggressive_scenario"])
        
        self.assertIsInstance(comparison, dict)
        
        # Check comparison structure
        self.assertIn('scenarios', comparison)
        self.assertIn('ranking', comparison)
        self.assertIn('summary', comparison)
        
        # Verify scenarios data
        scenarios_data = comparison['scenarios']
        self.assertEqual(len(scenarios_data), 2)
        
        # Each scenario should have analysis results
        for scenario_analysis in scenarios_data:
            self.assertIn('scenario', scenario_analysis)
            self.assertIn('financial_metrics', scenario_analysis)
            self.assertIn('capacity_metrics', scenario_analysis)
        
        # Verify ranking
        ranking = comparison['ranking']
        self.assertIsInstance(ranking, list)
        
        # Verify summary
        summary = comparison['summary']
        self.assertIsInstance(summary, dict)
    
    def test_generate_investment_recommendations(self):
        """Test investment recommendation generation"""
        # Create a scenario using existing investment options
        scenario = self.planner.create_investment_scenario(
            scenario_id="recommendation_test_scenario",
            name="Recommendation Test",
            description="Test scenario for recommendations",
            investment_ids=["new_berth_1", "crane_upgrade_1"]
        )
        
        # Get recommendations through ROI analysis
        analysis = self.planner.calculate_roi_analysis(scenario)
        recommendations = analysis['recommendations']
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check recommendation structure
        for recommendation in recommendations:
            self.assertIsInstance(recommendation, str)
            self.assertGreater(len(recommendation), 0)
    
    def test_assess_investment_risks(self):
        """Test investment risk assessment"""
        # Create a scenario using existing investment options
        scenario = self.planner.create_investment_scenario(
            scenario_id="high_risk_scenario",
            name="High-Risk Scenario",
            description="High-risk, high-reward investment",
            investment_ids=["automation_1", "new_berth_1"]
        )
        
        # Get risk assessment through ROI analysis
        analysis = self.planner.calculate_roi_analysis(scenario)
        risk_assessment = analysis['risk_assessment']
        
        self.assertIsInstance(risk_assessment, dict)
        
        # Check risk categories
        expected_keys = [
            'implementation_risk', 'demand_risk', 'technology_risk',
            'financial_risk', 'regulatory_risk', 'overall_risk_score'
        ]
        
        for key in expected_keys:
            self.assertIn(key, risk_assessment)
        
        # Risk levels should be valid strings or numeric scores
        risk_levels = ['low', 'medium', 'high', 'critical']
        for key in ['implementation_risk', 'demand_risk', 'technology_risk', 'financial_risk', 'regulatory_risk']:
            if key in risk_assessment:
                risk_value = risk_assessment[key]
                if isinstance(risk_value, str):
                    self.assertIn(risk_value, risk_levels)
                else:
                    self.assertGreaterEqual(risk_value, 0.0)
                    self.assertLessEqual(risk_value, 1.0)
        
        # Overall risk should be calculated
        overall_risk = risk_assessment['overall_risk_score']
        self.assertIsInstance(overall_risk, (int, float))
        self.assertGreaterEqual(overall_risk, 0.0)
        self.assertLessEqual(overall_risk, 1.0)
    
    def test_add_investment_option(self):
        """Test that investment options are properly initialized"""
        # The planner comes with pre-initialized investment options
        self.assertGreater(len(self.planner.investment_options), 0)
        
        # Check that some expected investment options exist
        expected_ids = ["new_berth_1", "crane_upgrade_1", "automation_1", "storage_expansion_1"]
        for investment_id in expected_ids:
            self.assertIn(investment_id, self.planner.investment_options)
            
        # Verify investment options have correct structure
        for investment_id, option in self.planner.investment_options.items():
            self.assertIsInstance(option, InvestmentOption)
            self.assertEqual(option.investment_id, investment_id)
            self.assertGreater(option.capital_cost, 0)
            self.assertGreaterEqual(option.capacity_increase, 0)
    
    def test_create_investment_scenario(self):
        """Test creating investment scenarios"""
        # Use existing investment options
        scenario = self.planner.create_investment_scenario(
            scenario_id="expansion_phase_1",
            name="Expansion Phase 1",
            description="Major expansion project",
            investment_ids=["new_berth_1", "crane_upgrade_1"]
        )
        
        self.assertIsInstance(scenario, InvestmentScenario)
        self.assertEqual(scenario.scenario_id, "expansion_phase_1")
        self.assertEqual(scenario.name, "Expansion Phase 1")
        self.assertEqual(len(scenario.investments), 2)
        
        # Verify investments are properly loaded
        investment_ids = [inv.investment_id for inv in scenario.investments]
        self.assertIn("new_berth_1", investment_ids)
        self.assertIn("crane_upgrade_1", investment_ids)
        
        # Verify scenario is stored in planner
        self.assertIn("expansion_phase_1", self.planner.scenarios)
        
        # Verify total costs are calculated
        self.assertGreater(scenario.total_capital_cost, 0)
        self.assertGreaterEqual(scenario.total_annual_operating_cost, 0)

if __name__ == '__main__':
    unittest.main()