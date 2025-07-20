# Data Sources for Hong Kong Port Digital Twin

This document outlines all available data sources for the Hong Kong Port Digital Twin project, including real-world APIs, open-source datasets, and integration options.

## Core Data Requirements

### 1. Vessel Movement Data
- **Ship arrivals and departures**
- **Real-time vessel positions (AIS data)**
- **Vessel specifications** (size, type, capacity)
- **Route information** and destinations
- **Speed and navigation data**

### 2. Port Infrastructure Data
- **Berth availability and specifications**
- **Terminal layouts and capacities**
- **Crane and equipment status**
- **Channel and anchorage information**
- **Port facility utilization**

### 3. Container Operations Data
- **Container throughput statistics**
- **Loading/unloading times**
- **Container yard occupancy**
- **TEU (Twenty-foot Equivalent Unit) metrics**
- **Cargo manifest information**

### 4. Operational Performance Metrics
- **Waiting times and queue lengths**
- **Berth utilization rates**
- **Turnaround times**
- **Efficiency indicators**
- **Cost and revenue data**

### 5. Environmental and Weather Data
- **Weather conditions** (wind, visibility, sea state)
- **Tidal information**
- **Environmental monitoring** (air quality, noise)
- **Safety incident reports**

---

## Primary Data Sources (MVP Implementation)

### 1. Hong Kong Government Open Data Portal
- **URL**: https://data.gov.hk/
- **Datasets Available**:
  - Marine Department vessel arrival/departure data
  - Port and maritime statistics
  - Weather and environmental data
  - Infrastructure and facility information
- **Format**: CSV, JSON, XML
- **Update Frequency**: Daily to monthly
- **Cost**: Free
- **API**: Available for most datasets

### 2. Hong Kong Marine Department Statistics
- **URL**: https://www.mardep.gov.hk/en/publication/port.html
- **Data Types**:
  - Monthly port statistics
  - Vessel registration data
  - Port facility information
  - Safety and incident reports
- **Format**: PDF reports, Excel files
- **Update Frequency**: Monthly
- **Cost**: Free
- **Integration**: Manual download, parsing required

### 3. Hong Kong Maritime and Port Board Data
- **URL**: https://www.hkmpb.gov.hk/
- **Data Types**:
  - Strategic port development data
  - Industry statistics and trends
  - Policy and regulatory information
- **Format**: Reports and publications
- **Update Frequency**: Quarterly/Annual
- **Cost**: Free
- **Integration**: Manual processing

---

## Enhanced Data Sources (Future Implementation)

### 1. MarineTraffic API ⭐ **RECOMMENDED**
- **URL**: https://www.marinetraffic.com/en/ais-api-services
- **Data Types**:
  - Real-time vessel positions (AIS)
  - Vessel details and specifications
  - Port calls and voyage information
  - Historical tracking data
- **Coverage**: Global, excellent Hong Kong coverage
- **Update Frequency**: Real-time (every few minutes)
- **Cost**: 
  - Free tier: 1,000 API calls/month
  - Paid plans: $50-500+/month
- **API Quality**: Excellent documentation and reliability

### 2. OpenAIS Project
- **URL**: https://github.com/OpenAIS/OpenAIS
- **Data Types**:
  - Raw AIS message processing
  - Vessel tracking algorithms
  - Open-source AIS decoders
- **Coverage**: Global (requires AIS receiver setup)
- **Update Frequency**: Real-time
- **Cost**: Free (hardware costs for receivers)
- **Integration**: Requires technical setup

### 3. AISViz Project
- **URL**: https://aisviz.org/
- **Data Types**:
  - Processed AIS data visualization
  - Maritime traffic analysis
  - Port activity metrics
- **Coverage**: Selected ports worldwide
- **Update Frequency**: Near real-time
- **Cost**: Free for research use
- **API**: Limited availability

---

## Optional Enhancements

### Real-time Map Integration

#### MarineTraffic Live Map Embedding ✅ **IMPLEMENTED**
- **Implementation**: Iframe embedding of MarineTraffic's live map
- **Status**: ✅ **Integrated in Dashboard** (Live Map tab)
- **URL**: `https://www.marinetraffic.com/en/ais/embed/zoom:11/centery:22.3/centerx:114.15/maptype:4/shownames:false/mmsi:0/shipid:0/fleet:0/fleet_hide_old_positions:false/fleet_hide_fishing_vessels:false/fleet_hide_cargo_vessels:false/fleet_hide_naval_vessels:false`
- **Features**: 
  - Real-time vessel positions around Hong Kong
  - Interactive map with zoom controls
  - Multiple map types (Satellite, Terrain, Basic)
  - Vessel information display
  - Customizable view options
- **Dashboard Integration**:
  - New "🌊 Live Map" tab in Streamlit dashboard
  - Map controls for zoom level and map type
  - Optional API integration for vessel data
  - Real-time refresh capabilities
- **Cost**: Free (with MarineTraffic branding)
- **API Alternative**: MarineTraffic API for custom integration (optional)

### Container Terminal Operator APIs

#### Major Hong Kong Terminal Operators
1. **COSCO Shipping Ports**
   - Operates multiple terminals at Kwai Tsing
   - May provide operational data through partnerships

2. **Modern Terminals Ltd (MTL)**
   - Container terminal operations
   - Potential for operational data sharing

3. **Hongkong International Terminals (HIT)**
   - Major container terminal operator
   - Historical data and statistics available

4. **Asia Container Terminals (ACT)**
   - Container handling services
   - Operational metrics potentially available

**Note**: These operators typically require business partnerships or agreements for data access.

---

## Implementation Roadmap

### Phase 1: MVP Data Sources (Weeks 1-4) ✅
- [x] Sample data generation
- [x] Basic simulation framework
- [x] Core visualization components
- [x] **MarineTraffic Live Map Integration** 🆕

### Phase 2: Government Data Integration (Weeks 5-6)
- [ ] Hong Kong Marine Department API integration
- [ ] Government Open Data Portal connection
- [ ] Data validation and cleaning pipelines
- [ ] MarineTraffic API data integration (optional)

### Phase 3: Enhanced Data Sources (Weeks 7-8)
- [ ] OpenAIS data processing
- [ ] Real-time data streaming
- [ ] Advanced MarineTraffic features

### Phase 4: Optional Enhancements (Weeks 9-10)
- [ ] Container terminal operator APIs
- [ ] Weather data integration
- [ ] Advanced analytics and ML features

---

## Technical Considerations

### API Rate Limits and Costs
- **MarineTraffic**: 1,000 free calls/month, paid plans available
- **Government APIs**: Usually unlimited but may have fair use policies
- **OpenAIS**: No limits but requires infrastructure setup

### Data Refresh Rates
- **Real-time data**: Every 1-5 minutes (AIS, weather)
- **Operational data**: Every 15-60 minutes (port operations)
- **Statistical data**: Daily to monthly updates

### Data Storage and Processing
- **Time-series database** recommended for real-time data
- **Data lake** approach for mixed format integration
- **Caching strategies** for expensive API calls
- **Data validation** and quality checks essential

### Security and Compliance
- **API key management** and rotation
- **Data privacy** considerations for vessel tracking
- **Rate limiting** and error handling
- **Backup data sources** for critical operations

---

## Cost Estimation

### Free Tier (MVP)
- Government data sources: **$0/month**
- MarineTraffic free tier: **$0/month** (1,000 calls)
- OpenAIS setup: **$200-500** (one-time hardware)

### Enhanced Tier
- MarineTraffic API: **$50-200/month**
- Additional data sources: **$100-300/month**
- Infrastructure costs: **$50-100/month**

### Enterprise Tier
- Premium APIs: **$500-2000/month**
- Terminal operator partnerships: **Variable**
- Advanced analytics: **$200-500/month**

---

## Next Steps

1. **✅ Complete**: MarineTraffic live map integration
2. **Priority 1**: Set up Hong Kong Government API connections
3. **Priority 2**: Implement MarineTraffic API integration (optional)
4. **Priority 3**: Explore OpenAIS for cost-effective real-time data
5. **Priority 4**: Establish partnerships with terminal operators

---

## Configuration

To enable optional data sources, update your `.env` file:

```bash
# MarineTraffic API (Optional)
MARINETRAFFIC_API_KEY=your_api_key_here

# Hong Kong Government APIs (Optional)
HK_MARINE_DEPT_API_KEY=your_api_key_here
```

See `.env.example` for complete configuration options.

---

*Last updated: 2024-12-19*
*Status: MarineTraffic live map integration completed ✅*