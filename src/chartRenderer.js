export class ChartRenderer {
    renderAgeDistribution(elementId, ageGroups, population) {
        const trace = {
            x: ageGroups,
            y: population,
            type: 'bar',
            marker: { color: '#667eea' }
        };
        
        const layout = {
            title: 'Population by Age Group',
            xaxis: { title: 'Age Group' },
            yaxis: { title: 'Population' },
            hovermode: 'x',
            plot_bgcolor: '#f7fafc',
            paper_bgcolor: 'white'
        };
        
        Plotly.newPlot(elementId, [trace], layout, { responsive: true });
    }
    
    renderProjection(elementId, years, values, projType) {
        let title = 'Total Population Projection';
        if (projType === 'age65plus') title = 'Age 65+ Population Projection';
        else if (projType === 'ageStructure') title = 'Age Structure Change';
        
        const trace = {
            x: years,
            y: values,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#667eea', width: 3 },
            marker: { size: 8 },
            fill: 'tozeroy',
            fillcolor: 'rgba(102, 126, 234, 0.2)'
        };
        
        const layout = {
            title,
            xaxis: { title: 'Year' },
            yaxis: { title: 'Population' },
            hovermode: 'x',
            plot_bgcolor: '#f7fafc',
            paper_bgcolor: 'white'
        };
        
        Plotly.newPlot(elementId, [trace], layout, { responsive: true });
    }
    
    renderMortality(elementId, ageGroups, rates) {
        const trace = {
            x: ageGroups,
            y: rates,
            type: 'bar',
            marker: { 
                color: rates,
                colorscale: 'Reds',
                showscale: true
            }
        };
        
        const layout = {
            title: 'Parkinson\'s Mortality Rate by Age Group (per 100,000)',
            xaxis: { title: 'Age Group' },
            yaxis: { title: 'Mortality Rate' },
            hovermode: 'x',
            plot_bgcolor: '#f7fafc',
            paper_bgcolor: 'white'
        };
        
        Plotly.newPlot(elementId, [trace], layout, { responsive: true });
    }
    
    renderMLForecast(elementId, years, cases, confidence) {
        const trace = {
            x: years,
            y: cases,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#764ba2', width: 3 },
            marker: { size: 6 },
            fill: 'tozeroy',
            fillcolor: 'rgba(118, 75, 162, 0.2)',
            name: 'Predicted PD Cases'
        };
        
        const layout = {
            title: 'ML-Forecasted Parkinson\'s Disease Burden',
            xaxis: { title: 'Year' },
            yaxis: { title: 'Number of Cases' },
            hovermode: 'x',
            plot_bgcolor: '#f7fafc',
            paper_bgcolor: 'white'
        };
        
        Plotly.newPlot(elementId, [trace], layout, { responsive: true });
    }
    
    renderHeatmap(elementId, geoData, values, metric) {
        const lats = geoData.map(d => d.lat);
        const lngs = geoData.map(d => d.lng);
        const names = geoData.map(d => d.name);
        
        const trace = {
            type: 'scattergeo',
            lon: lngs,
            lat: lats,
            mode: 'markers+text',
            text: names,
            textposition: 'top center',
            marker: {
                size: values.map(v => Math.sqrt(v) * 3),
                color: values,
                colorscale: 'Reds',
                showscale: true,
                colorbar: {
                    title: `${metric.charAt(0).toUpperCase() + metric.slice(1)}<br>(per 100K)`
                },
                line: { width: 2, color: '#667eea' }
            },
            hovertemplate: '<b>%{text}</b><br>Rate: %{marker.color:.1f}<extra></extra>'
        };
        
        const layout = {
            title: `Lee County, FL - Parkinson's ${metric.charAt(0).toUpperCase() + metric.slice(1)} Distribution`,
            geo: {
                scope: 'usa',
                projection: { type: 'mercator' },
                center: { lat: 26.5, lon: -81.9 },
                zoom: 8
            },
            paper_bgcolor: 'white',
            plot_bgcolor: '#f7fafc'
        };
        
        Plotly.newPlot(elementId, [trace], layout, { responsive: true });
    }
}
