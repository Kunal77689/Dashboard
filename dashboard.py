## Submission for Final Project Comp 4304/ Charvi and Kunal Sikka.

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px


df = pd.read_csv('open_gym.csv')
df['open_gym_start'] = pd.to_datetime(df['open_gym_start'])

app = dash.Dash(__name__)

facility_locations = df['facility_title'] + ': ' + df['location']
unique_facility_locations = facility_locations.unique()

dropdown_options = [{'label': facility_location, 'value': facility_location} for facility_location in unique_facility_locations]

app.layout = html.Div(style={'backgroundColor': '#191E66', 'padding': '20px'}, children=[
    html.H1(children='Dashboard', style={'textAlign': 'center', 'color': '#E6E6E6'}),
    html.Div([
        html.H3(children='Ratio of Females to Males and Ratio of Residents to Non-Residents  ', style={'textAlign': 'center', 'color': '#E6E6E6'}),
        html.Div(dcc.Graph(id='plot1'), style={'margin': 'auto', 'width': '70%'}),
        dcc.Slider(
            id='year-slider',
            min=df['open_gym_start'].dt.year.min(),
            max=df['open_gym_start'].dt.year.max(),
            step=1,
            value=df['open_gym_start'].dt.year.min(),
            marks={str(year): str(year) for year in df['open_gym_start'].dt.year.unique()},
            included=False
        ),
        dcc.Dropdown(
            id='facility-dropdown',
            options=[{'label': facility, 'value': facility} for facility in df['facility_title'].unique()],
            value=df['facility_title'].unique()[0],
            clearable=False,
            style={'width': '280px', 'margin': 'auto'}
        ),
    ], style={'backgroundColor': '#0f123d', 'padding': '10px', 'margin': '40px auto', 'borderRadius': '50px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'}),
    html.Div([
        html.H3(children='Total Number of Entries by Location', style={'textAlign': 'center', 'color': '#E6E6E6'}),
        html.Div(dcc.Graph(id='plot2'), style={'margin': 'auto', 'width': '60%'}),
        dcc.Dropdown(
            id='facility-dropdown2',
            options=[{'label': facility, 'value': facility} for facility in df['facility_title'].unique()],
            value=df['facility_title'].unique()[0],
            clearable=False,
            style={'width': '280px', 'margin': 'auto'}
        ),
    ], style={'backgroundColor': '#0f123d', 'padding': '10px', 'margin': '40px auto', 'borderRadius': '50px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'}),
    html.Div([
        html.H3(children='Engagement in Open Gym Activities for Desired Location', style={'textAlign': 'center', 'color': '#E6E6E6'}),
        html.Div(dcc.Graph(id='plot4'), style={'margin': 'auto', 'width': '70%'}),
        dcc.Dropdown(
            id='facility-location-dropdown2',
            options=dropdown_options,
            value=dropdown_options[0]['value'],
            clearable=False,
            style={'width': '400px', 'height': '40px', 'font-size': '15px', 'margin': 'auto'}
        ),
    ], style={'backgroundColor': '#0f123d', 'padding': '10px', 'margin': '40px auto', 'borderRadius': '50px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'}),

    html.Div([
    html.H3(children='Open Gym Activities by Group', style={'textAlign': 'center', 'color': '#E6E6E6'}),
    html.Div(dcc.Graph(id='plot3'), style={'margin': 'auto', 'width': '70%'}),
    html.Div([
        dcc.Dropdown(
            id='facility-location-dropdown',
            options=dropdown_options,
            value=dropdown_options[0]['value'],
            clearable=False,
            style={'width': '400px', 'height': '40px', 'font-size': '15px', 'margin-right': '20px'}
        ),
    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),
    html.Br(),  
    html.Div([
        html.Div([
            dcc.Checklist(
                id=activity,
                options=[{'label': activity, 'value': activity}],
                value=[],
                style={'color': '#E6E6E6', 'font-size': '12px', 'display': 'inline-block', 'margin-right': '10px'}
            ) for activity in df['open_gym_activity'].unique()
        ], style={'display': 'inline-block'})
    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'})
], style={'backgroundColor': '#0f123d', 'padding': '10px', 'margin': '40px auto', 'borderRadius': '50px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'})

    ])

@app.callback(
    Output('plot1', 'figure'),
    [Input('year-slider', 'value'),
     Input('facility-dropdown', 'value')]
)
def update_plot1(year, facility_title):

    filtered_df = df[(df['open_gym_start'].dt.year == year) & (df['facility_title'] == facility_title)]
    
    if filtered_df.empty:
        return {
            'data': [],
            'layout': {
                'title': "No data available for the selected year and facility",
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [{
                    'text': "No data available",
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 18}
                }],
                'plot_bgcolor': '#0f123d',
                'paper_bgcolor': '#0f123d',
                'font': {'color': 'white'}
            }
        }

    total_females_to_males_ratio = filtered_df['total_females'].sum() / filtered_df['total_males'].sum()
    total_residents_to_non_residents_ratio = filtered_df['total_residents'].sum() / filtered_df['total_non_residents'].sum()
    
    hover_text1 = f'Total Females: {filtered_df["total_females"].sum()}'
    hover_text3 = f'Total Males: {filtered_df["total_males"].sum()}'
    hover_text2 = f'Total Residents: {filtered_df["total_residents"].sum()}'
    hover_text4 = f'Total Non-Residents: {filtered_df["total_non_residents"].sum()}'

    
    fig = go.Figure()
    fig.add_trace(go.Pie(labels=['Females', 'Males'], values=[total_females_to_males_ratio, 1],
                         name="Females to Males",
                         hoverinfo='text',  
                         text=[hover_text1, hover_text3],  
                         marker=dict(colors=['#da64dc', '#6f56c9'], line=dict(color='white', width=3)),
                         hole=0.7,
                         pull=[0.1, 0],
                         domain={'x': [0, 0.45]},
                         textinfo='percent+label'))

    
    fig.add_trace(go.Pie(labels=['Residents', 'Non-Residents'], values=[total_residents_to_non_residents_ratio, 1],
                         name="Residents to Non-Residents",
                         hoverinfo='text',  
                         text=[hover_text2, hover_text4],  
                         marker=dict(colors=['#da64dc', '#6f56c9'], line=dict(color='white', width=3)),
                         hole=0.7,
                         pull=[0.1, 0],
                         domain={'x': [0.55, 1]},
                         textinfo='percent+label'))

    
    fig.update_layout(title={'text': f"For {facility_title} in {year}",
                              'x': 0.5,
                              'font': {'size': 16, 'color': 'white'}},  
                      width=900,
                      height=500,
                      plot_bgcolor='#0f123d', 
                      paper_bgcolor='#0f123d')
    

    fig.update_layout(showlegend=False)
    
    return fig

@app.callback(
    Output('plot2', 'figure'),
    [Input('facility-dropdown2', 'value')]
)
def update_plot2(facility_title):
 
    filtered_df = df[df['facility_title'] == facility_title]
    
    location_sums = filtered_df.groupby('location')['total'].sum().reset_index(name='total_entries')
    
    fig = px.bar(location_sums, x='location', y='total_entries',
                 title=f'Total number of entries for - {facility_title}',
                 labels={'location': 'Location', 'total_entries': 'Number of entries'},
                 color_discrete_sequence=['#da64dc'],  
                 opacity=0.7)  
    
   
    fig.update_layout(
        plot_bgcolor='#0f123d', 
        paper_bgcolor='#0f123d',  
        font=dict(color='white'),  
        xaxis=dict(tickfont=dict(color='white'), tickangle=-45),  
        yaxis=dict(tickfont=dict(color='white')),  
        title=dict(font=dict(color='white')),  
        legend=dict(title=dict(font=dict(color='white'))),  
        barmode='group', 
        bargap=0.15,  
        bargroupgap=0.1, 
        height=700,  
        width=800  
    )
  
    fig.update_traces(text=location_sums['total_entries'], textposition='outside')
    
    return fig
import plotly.graph_objs as go

@app.callback(
    Output('plot3', 'figure'),
    [Input('facility-location-dropdown', 'value')] + [Input(activity, 'value') for activity in df['open_gym_activity'].unique()]
)
def update_plot3(facility_location, *selected_activities):
    facility_title, location = extract_facility_location(facility_location)
    
    filtered_df = df[(df['facility_title'] == facility_title) & (df['location'] == location)]
    
    selected_activities_list = [activity for activity, selected in zip(df['open_gym_activity'].unique(), selected_activities) if selected]
    if selected_activities_list:
        filtered_df = filtered_df[filtered_df['open_gym_activity'].isin(selected_activities_list)]
    
    if filtered_df.empty:
        print("No data available for the selected criteria.")
        return {
            'data': [],
            'layout': {
                'title': "No data available for the selected criteria.",
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [{
                    'text': "No data available",
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 18}
                }],
                'plot_bgcolor': '#0f123d',
                'paper_bgcolor': '#0f123d',
                'font': {'color': 'white'}
            }
        }
    else:
  
        grouped_df = filtered_df.groupby(['open_gym_activity', 'group'])['total'].sum().reset_index()
     
        colors = {
            'Everyone': '#ff56dc',
            'Adult': '#845add',
            'Youth/Teen': '#4dc5cd',
            'Homeschool': '#709edd',
            'Preschool': '#42e0ca'
        }
   
        data = []
        for group, color in colors.items():
            group_df = grouped_df[grouped_df['group'] == group]
            trace = go.Bar(x=group_df['open_gym_activity'], y=group_df['total'],
                           name=group, marker_color=color, opacity=0.7)
            data.append(trace)
        
      
        layout = go.Layout(title=f'{facility_title}:{location}',
                           xaxis=dict(title='Open Gym Activity', tickfont=dict(color='white')),
                           yaxis=dict(title='Total', tickfont=dict(color='white')),
                           barmode='stack',
                           showlegend=True,  
                           plot_bgcolor='#0f123d',
                           paper_bgcolor='#0f123d', 
                           font=dict(color='white')  
                          )
        

        fig = go.Figure(data=data, layout=layout)
        
   
        fig.update_layout(hovermode='closest',
                          hoverlabel=dict(bgcolor="white", font_size=16, font_color="black")
                          )
        
        return fig

def extract_facility_location(facility_location_string):
    if isinstance(facility_location_string, list):
        facility_location_string = facility_location_string[0] 
    print("Facility location string type:", type(facility_location_string))
    colon_index = facility_location_string.find(':')
    facility_title = facility_location_string[:colon_index]
    location = facility_location_string[colon_index + 2:]  
    return facility_title, location

@app.callback(
    Output('plot4', 'figure'),
    [Input('facility-location-dropdown2', 'value')]
)
def update_plot4(facility_location):
    if isinstance(facility_location, list): 
        facility_location = facility_location[0]

   
    colon_index = facility_location.find(':')
    facility_title = facility_location[:colon_index]
    location = facility_location[colon_index + 2:]  

    
    filtered_df = df[(df['facility_title'] == facility_title) & (df['location'] == location)]
    
    activity_totals = filtered_df.groupby('open_gym_activity')['total'].sum().reset_index(name='total_entries')
    
    fig = px.scatter(activity_totals, x='open_gym_activity', y='total_entries', size='total_entries', 
                     hover_name='open_gym_activity', color='open_gym_activity', 
                     title=f'{facility_title}:{location}',
                     color_discrete_sequence=bubble_colors)
    
    
    fig.update_layout(
        plot_bgcolor='#0f123d',
        paper_bgcolor='#0f123d',
        xaxis=dict(showline=False, zeroline=False, showticklabels=True, title='', tickfont=dict(color='white')),
        yaxis=dict(showline=False, zeroline=False, showticklabels=False, title='', tickfont=dict(color='white')),
        legend=dict(title='', orientation='h', yanchor='bottom', y=1, xanchor='right', x=1),
        font=dict(color='white')
    )
    
    return fig
bubble_colors = ['#ff56dc', '#845add', '#4dc5cd', '#709edd', '#42e0ca', '#4698a3']

facility_to_location_mapping = {}
for index, row in df.iterrows():
    facility = row['facility_title']
    location = row['location']
    comma_index = location.find(',')
    if comma_index != -1:
    
        locations = [location[:comma_index], location[comma_index + 2:]]  
    else:
        locations = [location]

    for loc in locations:
        if facility not in facility_to_location_mapping:
            facility_to_location_mapping[facility] = set()
        facility_to_location_mapping[facility].add(loc)

facility_locations = []
for facility, locations in facility_to_location_mapping.items():
    for location in locations:
        facility_locations.append(f"{facility}: {location}")
if __name__ == '__main__':
    app.run_server(debug=True,port=8051)

#Instructions to start main.py
# - Open terminal
# - Open the directory the file is saved
# - Run python main.py
# - You will get the server link, click on that to open the dashboard
    
# Link for the video: https://drive.google.com/file/d/18B-yxz8h1q8iZoJODo8m8uAlfPcD3uSQ/view?usp=sharing
    
# Attributions:
# Class notes and videos
# https://dash.plotly.com/dash-html-components
# https://dash.plotly.com
# https://plotly.com/python/getting-started/
# https://dash.plotly.com/dash-core-components