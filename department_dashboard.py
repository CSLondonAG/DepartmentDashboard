# Altair Dual-Axis Chart Prompt

Create a professional dual-axis line chart in Python using Altair to visualize CSAT and NPS trends over time. Follow these specifications:

## Data Structure
```python
import pandas as pd
import altair as alt

# Sample data structure
data = pd.DataFrame({
    'month': ['Q1 AUG', 'Q2 AUG', 'Q3 AUG', 'Q4 AUG', 'Q5 AUG', 'Q6 AUG', 
              'Q7 AUG', 'Q8 AUG', 'Q9 AUG', 'Q10 AUG', 'Q11 AUG', 'Q12 AUG', 'Q13 AUG'],
    'csat': [47.1, 47.1, 41.7, 25.0, 33.3, 30.8, 46.3, 42.7, 44.2, 44.9, 28.5, 44.9, 44.9],
    'nps': [-50, -25, -35, -60, -80, -50, -55, -30, -30, -80, -70, -20, -5]
})
```

## Chart Requirements

### 1. Chart Structure
- Create a dual-axis line chart with CSAT on left axis and NPS on right axis
- Use `alt.layer()` to combine two separate line charts
- Set chart dimensions to 800x400 pixels
- Add a meaningful title: "CSAT & NPS Trend Analysis"

### 2. Left Axis (CSAT)
- Scale domain: [0, 100]
- Line color: blue (#2563eb)
- Axis title: "CSAT (%)"
- Line thickness: 3px
- Point markers: filled circles, size 60

### 3. Right Axis (NPS)
- Scale domain: [-100, 100]
- Line color: red (#dc2626)
- Axis title: "NPS Score"
- Line thickness: 3px
- Point markers: filled circles, size 60
- Add horizontal reference line at y=0 with dashed style

### 4. Styling Requirements
- Grid: light gray, dashed lines
- Background: white
- X-axis labels: rotate 45 degrees for better readability
- Font: clean, readable (Arial or similar)
- Tooltips: show exact values for both metrics on hover
- Legend: positioned at top-right

### 5. Interactive Features
- Add tooltips showing:
  - Month/period
  - CSAT percentage with "%" symbol
  - NPS score value
- Enable point hover highlighting
- Include zoom functionality

### 6. Professional Touches
- Add subtle drop shadow or border around chart area
- Ensure proper spacing and margins
- Use consistent color scheme throughout
- Add data point labels for key insights (optional)

## Expected Output
Generate clean, production-ready code that creates a visually appealing dual-axis chart suitable for business presentations. The chart should clearly differentiate between the two metrics while showing their trends over the same time period.

## Code Structure
Organize the code with:
1. Data preparation section
2. Base chart configuration
3. Individual chart layers (CSAT and NPS)
4. Chart combination and styling
5. Display/export options

Ensure the final chart is publication-ready with proper formatting, clear legends, and professional appearance.
