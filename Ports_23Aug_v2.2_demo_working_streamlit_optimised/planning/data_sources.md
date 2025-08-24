# Data Sources for Hong Kong Port Digital Twin

## Overview
This document outlines available data sources for the Hong Kong Port Digital Twin project, categorized by priority and implementation complexity.

## Core Data Requirements

### 1. Vessel Movement & Tracking
- Real-time ship positions, speeds, and routes
- Vessel arrivals and departures
- Ship specifications (length, width, draft, capacity)
- Maritime Mobile Service Identity (MMSI) data

### 2. Port Infrastructure
- Berth specifications and availability
- Container terminal layouts and capacity
- Water depth and navigation channels
- Port facilities and equipment

### 3. Container Operations
- Container throughput statistics (TEU)
- Loading/unloading times
- Container yard occupancy
- Cargo types and destinations

### 4. Performance Metrics
- Port performance indicators
- Vessel turnaround times
- Queue lengths and waiting times
- Resource utilization rates

## Primary Data Sources (MVP Implementation)

### 1. Hong Kong Government Open Data Portal ⭐
- **URL**: https://data.gov.hk/en-datasets/provider/hk-md
- **Type**: Official government data
- **Access**: Free API
- **Data**: Vessel arrivals/departures, port statistics
- **Update Frequency**: Regular
- **Implementation Priority**: HIGH
- **Status**: ✅ Available for immediate use
- **My Comments**: "I think the Hong Kong government open data portal is quite comprehensive. I looked through the available data sources and when I searched for 'Port', I found a few things like Vessel Arrival and Departure, Port cargo statistics, Port cargo throughput statistics, Port transshipment cargo statistics, Container throughput statistics, Port cargo throughput by Greater Bay Area City statistics, Quarter to quarter percentage change of seasonally adjusted series of Port cargo throughput, Non-convention vessel arrivals and departures, Town planning board guidelines, Port cargo throughput by mode of transport. None of these seem to be API though; mostly files (pdf, csv, xls)"

### 2. Hong Kong Marine Department Statistics ⭐
- **URL**: https://www.mardep.gov.hk/en/materials-and-publications/publications/port-and-maritime-statistics/
- **Type**: Official statistics
- **Access**: Public reports
- **Data**: Container throughput, vessel statistics, port performance
- **Update Frequency**: Monthly/Annual
- **Implementation Priority**: HIGH
- **Status**: ✅ Available for immediate use
- **My Comments**: "Data available:
  Vessel Statistics:
  1. Vessel Arrivals by Ocean/River and Cargo/Passenger Vessels	MonthlyPDF	QuarterlyPDF	Year-to-datePDF
  2. Vessel Arrivals by Ship Type and Ocean/River	MonthlyPDF	QuarterlyPDF	Year-to-datePDF
  3. Vessel Arrivals by Flag and Ocean/River	 	QuarterlyPDF	Year-to-datePDF
  4. Number and NRT of Vessels Calling Hong Kong by Flag and Ocean/River	 	QuarterlyPDF	Year-to-datePDF
  5. Average Time in Port for Vessels Departing Hong Kong by Ship Type and Ocean/River	 	QuarterlyPDF	Year-to-datePDF
  6. Ocean Vessel Arrivals by Main Berthing Location	 	QuarterlyPDF	Year-to-datePDF
  7. Ocean Vessel Departures by Main Berthing Location	 	QuarterlyPDF	Year-to-datePDF
  8. Statistics on Ocean Vessels Calling Container Terminals	 	QuarterlyPDF	Year-to-datePDF

  Port Container Throughput:
  The latest container throughputs of Hong Kong Port and Kwai Tsing Container TerminalsPDF
  
  Port Container Throughput by Handling Location and Seaborne/RiverPDF
  
  Port Cargo Throughput:  
  Port Cargo Throughput by Seaborne/River CargoPDF
  Seaborne Cargo Discharged/Loaded
  River Cargo Discharged/Loaded
  Port Cargo Discharged/Loaded
  
  Cross-Boundary Ferry Terminals Statistics:
  1. Statistics on Vessel Arrivals and Departures at Cross-Boundary Ferry Terminals	 	QuarterlyPDF	Year-to-datePDF
  2. Passenger Arrivals and Departures by Routes	 	QuarterlyPDF	Year-to-datePDF
  Others

  Ships Registered in Hong KongPDF
  Time Series of Statistics for the Past Years

  Vessel Statistics:
  1. Vessel Arrivals by Ocean/River and Cargo/Passenger Vessels	MonthlyPDF	QuarterlyPDF	AnnualPDF
  2. Vessel Arrivals by Ship Type and Ocean/River	MonthlyPDF	QuarterlyPDF	AnnualPDF
  3. Vessel Arrivals by Flag and Ocean/River	 	QuarterlyPDF	AnnualPDF
  4. Number and NRT of Vessels Calling Hong Kong by Flag and Ocean/River	 	QuarterlyPDF	AnnualPDF
  5. Average Time in Port for Vessels Departing Hong Kong by Ship Type and Ocean/River	 	QuarterlyPDF	AnnualPDF
  6. Ocean Vessel Arrivals by Main Berthing Location	 	QuarterlyPDF	AnnualPDF
  7. Ocean Vessel Departures by Main Berthing Location	 	QuarterlyPDF	AnnualPDF
  8. Statistics on Ocean Vessels Calling Container Terminals	 	QuarterlyPDF	AnnualPDF
  9. Ocean Vessel Arrivals by Summer Draft	 	 	AnnualPDF
  10. Ocean Vessel Arrivals by Vessel Length	 	 	AnnualPDF
  11. Ocean Vessel Arrivals by Main Reason of Call	 	 	AnnualPDF

  Port Container Throughput:
  1. Port Container Throughput by Handling Locations and Seaborne/River	MonthlyPDF	QuarterlyPDF	AnnualPDF
  
  Port Cargo Throughput:
  1. Port Cargo Throughput by Seaborne/River Cargo	MonthlyPDF	QuarterlyPDF	AnnualPDF
  2. Seaborne Cargo Discharged/Loaded	 
  3. River Cargo Discharged/Loaded	 
  4. Port Cargo Discharged/Loaded	 
  
  Cross-Boundary Ferry Terminals Statistics:
  1. Statistics on Vessel Arrivals and Departures at Cross-Boundary Ferry Terminals	 	QuarterlyPDF	AnnualPDF
  2. Passenger Arrivals and Departures by Routes	 	QuarterlyPDF	AnnualPDF
  
  Others:
  1. Ships Registered in Hong Kong	 	 	AnnualPDF
  2. Marine Accidents	 	 	AnnualPDF
  3. Hong Kong Licensed Vessels	 	 	AnnualPDF
"

### 3. Hong Kong Maritime and Port Board ⭐
- **URL**: https://www.hkmpdb.gov.hk/en/port.html
- **Type**: Port authority data
- **Access**: Public portal
- **Data**: Terminal operations, facility information
- **Update Frequency**: Regular
- **Implementation Priority**: HIGH
- **Status**: ✅ Available for immediate use
- **My Comments**: "Not much data here; more like a govt board website with some links to data that seems available in #1 and #2."

## Enhanced Data Sources (Future Implementation)

### 4. MarineTraffic API 🚢
- **URL**: https://www.marinetraffic.com/
- **Type**: Commercial AIS data provider
- **Access**: Paid API (free tier available)
- **Data**: Real-time vessel tracking, AIS data
- **Update Frequency**: Real-time
- **Implementation Priority**: MEDIUM
- **Cost**: Free tier: 1,000 API calls/month, Paid plans from $99/month
- **Status**: 🔄 Evaluation phase
- **My Comments**: "Pretty visualization, but I don't see a free tier. There are three seven days, but I can't use that because I would need more than seven days to build and test the feature before rolling it out in the demo. We're going to pay the entry-level API here if we can find something useful to do with it. Maybe something around real-time shipping data. Okay, today is the day of the demo. This is real-time data from today. This is what it looks like. It's going to happen today. Now let's assume XYZ and then pull up the visualization. But that is a five-fetched. Again, it's more for the visuals, which maybe the business audience doesn't care so much for. Perhaps just a screenshot or a live view of the ships would be enough."

### 5. OpenAIS
- **URL**: https://open-ais.org/
- **Type**: Open-source AIS tools
- **Access**: Free
- **Data**: AIS data processing tools
- **Update Frequency**: Community-driven
- **Implementation Priority**: MEDIUM
- **Status**: 🔄 Evaluation phase

### 6. AISViz Project
- **URL**: https://github.com/AISViz
- **Type**: Research initiative
- **Access**: Open-source
- **Data**: Vessel tracking visualization tools
- **Update Frequency**: Project-based
- **Implementation Priority**: LOW
- **Status**: 📋 Research phase

## Optional Enhancements

### Real-Time Map Integration 🗺️

#### MarineTraffic Live Map Embedding
- **Purpose**: Visual enhancement for dashboard
- **Implementation**: Iframe embedding or API integration
- **Benefits**: 
  - Real-time vessel visualization
  - Professional maritime interface
  - User engagement enhancement
- **Considerations**:
  - Requires API subscription for commercial use
  - May need custom styling to match dashboard theme
  - Performance impact on dashboard loading

#### Implementation Options:

**Option A: Iframe Embedding (Simplest)**
```html
<iframe 
  src="https://www.marinetraffic.com/en/ais/embed/zoom:10/centery:22.3/centerx:114.2/maptype:4/shownames:false/mmsi:0/shipid:0/fleet:0/fleet_hide_old_positions:false/fleet_hide_fishing_vessels:false/fleet_hide_passenger_vessels:false"
  width="100%" 
  height="400px"
  frameborder="0">
</iframe>
```

**Option B: API Integration (Advanced)**
- Use MarineTraffic API for vessel data
- Custom map implementation with Leaflet/Mapbox
- Full control over styling and functionality

**Option C: Hybrid Approach**
- Use free tier API for basic vessel data
- Combine with open-source mapping libraries
- Fallback to static data when API limits reached

### Container Terminal Operator APIs

#### Modern Terminals Ltd (MTL)
- **Contact**: cad@ModernTerminals.com
- **Data**: Terminal operations, berth availability
- **Access**: Direct partnership required

#### Hong Kong International Terminals (HIT)
- **Location**: Terminal 4, Container Port Road South
- **Data**: 12 berths operations data
- **Access**: Commercial agreement needed

#### COSCO-HIT Terminals
- **Location**: CHT Tower, Terminal 8 East
- **Data**: Joint venture operations
- **Access**: Partnership required

## Implementation Roadmap

### Phase 1: MVP (Current)
- ✅ Sample data generation
- ✅ Basic simulation engine
- ✅ Dashboard framework

### Phase 2: Government Data Integration
- 🔄 Hong Kong Open Data Portal API
- 🔄 Marine Department statistics
- 🔄 HKMPB data integration

### Phase 3: Real-Time Enhancements
- 📋 MarineTraffic API evaluation
- 📋 Live map integration
- 📋 Real-time vessel tracking

### Phase 4: Commercial Partnerships
- 📋 Terminal operator data access
- 📋 Enhanced operational metrics
- 📋 Predictive analytics

## Technical Considerations

### API Rate Limits
- Government APIs: Generally unlimited for public data
- MarineTraffic: 1,000 calls/month (free), higher limits with paid plans
- Commercial APIs: Varies by provider

### Data Refresh Rates
- Real-time: Every 1-5 minutes (vessel positions)
- Operational: Every 15-30 minutes (berth status)
- Statistical: Daily/weekly (performance metrics)

### Storage Requirements
- Historical data: ~1GB per year for basic metrics
- Real-time cache: ~100MB for active vessels
- Map tiles: ~500MB for Hong Kong region

## Security & Compliance

### Data Privacy
- No personal information collection
- Vessel tracking uses public AIS data
- Compliance with Hong Kong data protection laws

### API Security
- Environment variables for API keys
- Rate limiting implementation
- Error handling for API failures

## Cost Estimation

### Free Tier (MVP)
- Government data: $0
- OpenAIS tools: $0
- Basic MarineTraffic: $0 (limited)
- **Total**: $0/month

### Enhanced Tier
- MarineTraffic Professional: $99/month
- Additional APIs: $50-200/month
- **Total**: $150-300/month

### Enterprise Tier
- Full MarineTraffic access: $500+/month
- Terminal operator partnerships: Variable
- **Total**: $1000+/month

## Next Steps

1. **Immediate**: Implement government data sources
2. **Short-term**: Evaluate MarineTraffic free tier
3. **Medium-term**: Prototype live map integration
4. **Long-term**: Establish commercial partnerships

---

*Last updated: December 2024*
*Status: Living document - updated as new sources are identified*