import { ParkinsonsDataManager } from './dataManager.js';
import { MLForecaster } from './mlForecaster.js';
import { ChartRenderer } from './chartRenderer.js';

class App {
    constructor() {
        this.dataManager = new ParkinsonsDataManager();
        this.mlForecaster = new MLForecaster();
        this.chartRenderer = new ChartRenderer();
        this.init();
    }
    
    async init() {
        console.log('🚀 Initializing Lee County Parkinson\'s Disease Analysis Platform...');
        
        // Load initial data
        await this.loadInitialData();
        
        // Setup event listeners
        document.getElementById('yearSelector').addEventListener('change', () => this.renderAgeDistribution());
        document.getElementById('projectionType').addEventListener('change', () => this.renderPopulationProjection());
        document.getElementById('heatmapYear').addEventListener('change', () => this.renderHeatmap());
        document.getElementById('heatmapMetric').addEventListener('change', () => this.renderHeatmap());
    }
    
    async loadInitialData() {
        try {
            console.log('📥 Fetching Census data from U.S. Census Bureau...');
            const censusData = await this.dataManager.fetchCensusData();
            console.log('✅ Census data loaded:', censusData);
            
            console.log('📥 Fetching population projections from BEBR/EDR...');
            const projections = await this.dataManager.fetchPopulationProjections();
            console.log('✅ Population projections loaded:', projections);
            
            console.log('📥 Fetching Parkinson\'s mortality data from CDC WONDER...');
            const mortalityData = await this.dataManager.fetchMortalityData();
            console.log('✅ Mortality data loaded:', mortalityData);
            
            // Render initial charts
            await this.renderAgeDistribution();
            await this.renderPopulationProjection();
            await this.loadMortalityData();
            await this.renderHeatmap();
        } catch (error) {
            console.error('❌ Error loading data:', error);
            this.showError('Failed to load initial data. Please refresh the page.');
        }
    }
    
    async renderAgeDistribution() {
        const year = document.getElementById('yearSelector').value;
        const data = await this.dataManager.getAgeDistribution(year);
        
        this.chartRenderer.renderAgeDistribution(
            'ageDistributionChart',
            data.ageGroups,
            data.population
        );
        
        // Update stats
        document.getElementById('totalPop').textContent = 
            this.formatNumber(data.totalPopulation);
        document.getElementById('medianAge').textContent = 
            data.medianAge.toFixed(1);
        document.getElementById('seniorPop').textContent = 
            this.formatNumber(data.age65plus);
        document.getElementById('seniorPct').textContent = 
            ((data.age65plus / data.totalPopulation) * 100).toFixed(1) + '%';
    }
    
    async renderPopulationProjection() {
        const projType = document.getElementById('projectionType').value;
        const data = await this.dataManager.getPopulationProjections(projType);
        
        this.chartRenderer.renderProjection(
            'populationProjectionChart',
            data.years,
            data.values,
            projType
        );
    }
    
    async loadMortalityData() {
        const data = await this.dataManager.fetchMortalityData();
        
        this.chartRenderer.renderMortality(
            'mortalityChart',
            data.ageGroups,
            data.rates
        );
        
        // Find highest risk group
        const maxIdx = data.rates.indexOf(Math.max(...data.rates));
        document.getElementById('highestRiskGroup').textContent = data.ageGroups[maxIdx];
        document.getElementById('highestRiskRate').textContent = 
            data.rates[maxIdx].toFixed(1);
    }
    
    async runMLForecast() {
        const forecastYear = parseInt(document.getElementById('forecastYear').value);
        
        try {
            const forecast = await this.mlForecaster.generateForecast(
                await this.dataManager.getCensusData(),
                await this.dataManager.getPopulationProjections('age65plus'),
                await this.dataManager.getMortalityData(),
                forecastYear
            );
            
            this.chartRenderer.renderMLForecast(
                'mlForecastChart',
                forecast.years,
                forecast.cases,
                forecast.confidence
            );
            
            // Update ML stats
            document.getElementById('predictedCases').textContent = 
                this.formatNumber(forecast.cases[forecast.cases.length - 1]);
            document.getElementById('percentChange').textContent = 
                forecast.percentChange.toFixed(1) + '%';
            document.getElementById('modelConfidence').textContent = 
                (forecast.overallConfidence * 100).toFixed(1) + '%';
            document.getElementById('peakRiskYear').textContent = 
                forecast.peakYear;
        } catch (error) {
            console.error('❌ ML Forecast Error:', error);
            this.showError('Failed to generate ML forecast. Please check your input.');
        }
    }
    
    async renderHeatmap() {
        const year = document.getElementById('heatmapYear').value;
        const metric = document.getElementById('heatmapMetric').value;
        
        const heatmapData = await this.dataManager.getGeographicHeatmap(year, metric);
        
        this.chartRenderer.renderHeatmap(
            'heatmapChart',
            heatmapData.geoData,
            heatmapData.values,
            metric
        );
    }
    
    formatNumber(num) {
        return new Intl.NumberFormat('en-US').format(Math.round(num));
    }
    
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error';
        errorDiv.textContent = message;
        document.querySelector('.container').insertBefore(
            errorDiv,
            document.querySelector('.main-content')
        );
    }
}

// Initialize app when DOM is ready
const app = new App();
window.app = app; // Make available globally for button onclick handlers
