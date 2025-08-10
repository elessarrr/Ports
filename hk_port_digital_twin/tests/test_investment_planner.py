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
        results = self.planner.analyze_investment_scenario(
            investment_type=InvestmentType.NEW_BERTH,
            investment_amount=50_000_000,
            demand_growth_rate=0.06,
            analysis_years=10
        )
        
        self.assertIsInstance(results, dict)
        
        # Check required result fields
        required_fields = [
            'roi_percentage', 'npv', 'payback_years',
            'capacity_impact', 'financial_projections',
            'recommendations', 'risk_assessment'
        ]
        
        for field in required_fields:
            self.assertIn(field, results)
        
        # Verify data types
        self.assertIsInstance(results['roi_percentage'], (int, float))
        self.assertIsInstance(results['npv'], (int, float))
        self.assertIsInstance(results['payback_years'], (int, float))
        self.assertIsInstance(results['capacity_impact'], dict)
        self.assertIsInstance(results['financial_projections'], list)
        self.assertIsInstance(results['recommendations'], list)
        self.assertIsInstance(results['risk_assessment'], dict)
        
        # Verify capacity impact structure
        capacity_impact = results['capacity_impact']
        self.assertIn('throughput_increase', capacity_impact)
        self.assertIn('efficiency_improvement', capacity_impact)
        
        # Verify financial projections length
        self.assertEqual(len(results['financial_projections']), 10)
    
    def test_compare_investment_scenarios(self):
        """Test comparing multiple investment scenarios"""
        # Create multiple scenarios
        scenario1_results = self.planner.analyze_investment_scenario(
            investment_type=InvestmentType.NEW_BERTH,
            investment_amount=50_000_000,
            demand_growth_rate=0.05,
            analysis_years=10
        )
        
        scenario2_results = self.planner.analyze_investment_scenario(
            investment_type=InvestmentType.CRANE_UPGRADE,
            investment_amount=20_000_000,
            demand_growth_rate=0.05,
            analysis_years=10
        )
        
        scenarios = {
            'New Berth': scenario1_results,
            'Crane Upgrade': scenario2_results
        }
        
        comparison = self.planner.compare_investment_scenarios(scenarios)
        
        self.assertIsInstance(comparison, dict)
        
        # Check comparison structure
        self.assertIn('best_roi_scenario', comparison)
        self.assertIn('best_npv_scenario', comparison)
        self.assertIn('fastest_payback_scenario', comparison)
        self.assertIn('scenario_rankings', comparison)
        
        # Verify scenario rankings
        rankings = comparison['scenario_rankings']
        self.assertIsInstance(rankings, list)
        self.assertEqual(len(rankings), 2)
        
        for ranking in rankings:
            self.assertIn('scenario_name', ranking)
            self.assertIn('overall_score', ranking)
            self.assertIn('roi_rank', ranking)
            self.assertIn('npv_rank', ranking)
    
    def test_generate_investment_recommendations(self):
        """Test investment recommendation generation"""
        # Test recommendations for different investment types
        berth_recs = self.planner.generate_investment_recommendations(
            InvestmentType.NEW_BERTH,
            roi_percentage=15.5,
            payback_years=6.2,
            risk_level="Medium"
        )
        
        self.assertIsInstance(berth_recs, list)
        self.assertGreater(len(berth_recs), 0)
        
        # Test automation recommendations
        automation_recs = self.planner.generate_investment_recommendations(
            InvestmentType.AUTOMATION,
            roi_percentage=22.3,
            payback_years=4.1,
            risk_level="Low"
        )
        
        self.assertIsInstance(automation_recs, list)
        self.assertGreater(len(automation_recs), 0)
        
        # Recommendations should be different for different investment types
        self.assertNotEqual(berth_recs, automation_recs)
    
    def test_assess_investment_risks(self):
        """Test investment risk assessment"""
        risk_assessment = self.planner.assess_investment_risks(
            investment_type=InvestmentType.NEW_BERTH,
            investment_amount=50_000_000,
            demand_growth_rate=0.05,
            market_volatility=0.15
        )
        
        self.assertIsInstance(risk_assessment, dict)
        
        # Check risk categories
        expected_risks = [
            'Market Risk', 'Financial Risk', 'Operational Risk',
            'Regulatory Risk', 'Technology Risk'
        ]
        
        for risk in expected_risks:
            self.assertIn(risk, risk_assessment)
        
        # Verify risk levels are valid
        valid_risk_levels = ['Low', 'Medium', 'High', 'Critical']
        for risk_level in risk_assessment.values():
            self.assertIn(risk_level, valid_risk_levels)
    
    def test_add_investment_option(self):
        """Test adding investment options to the planner"""
        option = InvestmentOption(
            investment_id="smart_terminal",
            name="Smart Terminal System",
            investment_type=InvestmentType.AUTOMATION,
            priority=InvestmentPriority.HIGH,
            capital_cost=30_000_000,
            annual_operating_cost=1_500_000,
            implementation_time_months=18,
            capacity_increase=400_000,
            efficiency_improvement=0.25,
            lifespan_years=20
        )
        
        self.planner.add_investment_option(option)
        
        self.assertEqual(len(self.planner.investment_options), 1)
        self.assertEqual(self.planner.investment_options[0], option)
    
    def test_create_investment_scenario(self):
        """Test creating investment scenarios"""
        # Add some investment options first
        option1 = InvestmentOption(
            investment_id="new_berth_test",
            name="New Berth",
            investment_type=InvestmentType.NEW_BERTH,
            priority=InvestmentPriority.HIGH,
            capital_cost=50_000_000,
            annual_operating_cost=2_500_000,
            implementation_time_months=24,
            capacity_increase=500_000,
            efficiency_improvement=0.15,
            lifespan_years=25
        )
        
        option2 = InvestmentOption(
            investment_id="crane_upgrade_test",
            name="Crane Upgrade",
            investment_type=InvestmentType.CRANE_UPGRADE,
            priority=InvestmentPriority.MEDIUM,
            capital_cost=15_000_000,
            annual_operating_cost=750_000,
            implementation_time_months=12,
            capacity_increase=200_000,
            efficiency_improvement=0.20,
            lifespan_years=15
        )
        
        self.planner.add_investment_option(option1)
        self.planner.add_investment_option(option2)
        
        scenario = self.planner.create_investment_scenario(
            name="Expansion Phase 1",
            selected_options=[option1, option2],
            budget_limit=70_000_000,
            timeline_years=3
        )
        
        self.assertIsInstance(scenario, InvestmentScenario)
        self.assertEqual(scenario.name, "Expansion Phase 1")
        self.assertEqual(len(scenario.selected_options), 2)
        self.assertEqual(scenario.budget_limit, 70_000_000)
        self.assertEqual(scenario.timeline_years, 3)
        
        # Total cost should be within budget
        total_cost = sum(option.capital_cost for option in scenario.selected_options)
        self.assertLessEqual(total_cost, scenario.budget_limit)

if __name__ == '__main__':
    unittest.main()