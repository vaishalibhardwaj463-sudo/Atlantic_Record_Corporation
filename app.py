import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. DATA LOADING & TRANSFORMATION ---
@st.cache_data
def load_and_process_data():
    
    # Load data
    df = pd.read_csv("Atlantic_Spain.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Get Entry Dates
    entry_dates = df.groupby(['song', 'artist'])['date'].min().reset_index()
    entry_dates.rename(columns={'date': 'Entry_Date'}, inplace=True)
    
    # 2. Get Peak Position and Peak Date (First time hitting the peak)
    peak_records = df.sort_values(['position', 'date']).drop_duplicates(['song', 'artist'], keep='first')
    peak_records = peak_records[['song', 'artist', 'date', 'position']]
    peak_records.rename(columns={'date': 'Peak_Date', 'position': 'Peak_Position'}, inplace=True)
    
    # 3. Aggregate total stats per song
    song_stats = df.groupby(['song', 'artist']).agg(
        Exit_Date=('date', 'max'),
        Days_on_Playlist=('date', 'nunique'),
        Is_Explicit=('is_explicit', 'first'),
        Album_Type=('album_type', 'first')
    ).reset_index()
    
    # Merge all stats into a single dataframe
    master_df = pd.merge(song_stats, entry_dates, on=['song', 'artist'])
    master_df = pd.merge(master_df, peak_records, on=['song', 'artist'])
    
    # Calculate Entry-to-Peak Time
    master_df['Entry_to_Peak_Days'] = (master_df['Peak_Date'] - master_df['Entry_Date']).dt.days
    
    # 4. Lifecycle Stage Classification (Heuristic based on lifespan and peak)
    def classify_stage(row):
        if row['Days_on_Playlist'] <= 7:
            return "New Entry"
        elif row['Peak_Position'] <= 10:
            return "Peak Phase"
        elif row['Days_on_Playlist'] >= 30:
            return "Mature Phase"
        else:
            return "Growth/Decline Phase"
            
    master_df['Lifecycle_Stage'] = master_df.apply(classify_stage, axis=1)
    
    return master_df

# Load the processed data
try:
    df = load_and_process_data()
except Exception as e:
    st.error(f"Error loading data from Google Drive: {e}. Please ensure the file is set to 'Anyone with the link can view'.")
    st.stop()

# --- 2. USER CAPABILITIES (SIDEBAR FILTERS) ---
st.sidebar.header("Dashboard Filters")

# Date Range Selector (Safely handle 1 or 2 selected dates)
min_date, max_date = df['Entry_Date'].min(), df['Entry_Date'].max()
date_selection = st.sidebar.date_input("Select Entry Date Range", [min_date, max_date])

# Check if the user has selected both start and end dates
if len(date_selection) == 2:
    start_date, end_date = date_selection
else:
    # If only start date is selected, temporarily set end_date to start_date to prevent crashes
    start_date = date_selection[0]
    end_date = date_selection[0]

# Lifecycle Stage Filter
stages = df['Lifecycle_Stage'].unique().tolist()
selected_stages = st.sidebar.multiselect("Lifecycle Stage", stages, default=stages)

# Explicit Content Toggle
explicit_filter = st.sidebar.radio("Explicit Content", ["All", "Explicit Only", "Clean Only"])

# Album Type Filter
album_types = df['Album_Type'].dropna().unique().tolist()
selected_album_types = st.sidebar.multiselect("Album Type", album_types, default=album_types)

# Apply Filters
start_date_dt = pd.to_datetime(start_date)
end_date_dt = pd.to_datetime(end_date)

filtered_df = df[
    (df['Entry_Date'] >= start_date_dt) & 
    (df['Entry_Date'] <= end_date_dt) &
    (df['Lifecycle_Stage'].isin(selected_stages)) &
    (df['Album_Type'].isin(selected_album_types))
]

if explicit_filter == "Explicit Only":
    filtered_df = filtered_df[filtered_df['Is_Explicit'] == True]
elif explicit_filter == "Clean Only":
    filtered_df = filtered_df[filtered_df['Is_Explicit'] == False]

# Stop execution nicely if the user is in the middle of picking dates so the screen doesn't just go blank
if len(date_selection) < 2:
    st.info("Please select an end date to view the dashboard.")
    st.stop()

# --- 3. CORE MODULES (MAIN DASHBOARD) ---
st.title("🎵 Spain Top 50: Lifecycle & Rotation Analysis")
st.markdown("Insights into content maturity, playlist churn, and release lifecycles based on real playlist data.")

# KPIs
st.subheader("Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Days on Playlist", f"{filtered_df['Days_on_Playlist'].mean():.1f}" if not filtered_df.empty else "0")
col2.metric("Total Unique Songs", len(filtered_df))
col3.metric("Avg Time to Peak (Days)", f"{filtered_df['Entry_to_Peak_Days'].mean():.1f}" if not filtered_df.empty else "0")
col4.metric("Explicit %", f"{(filtered_df['Is_Explicit'].sum() / len(filtered_df) * 100):.1f}%" if not filtered_df.empty else "0%")

# Core Module: Lifecycle Stage Distribution
st.subheader("Lifecycle Stage Distribution")
if not filtered_df.empty:
    fig_stage = px.pie(filtered_df, names='Lifecycle_Stage', title='Proportion of Songs in Each Phase')
    st.plotly_chart(fig_stage, use_container_width=True)
else:
    st.warning("No data available for the selected filters.")

# Core Module: Content Maturity Comparisons (Boxplot)
st.subheader("Content Maturity: Explicit vs. Clean & Album vs. Single")
if not filtered_df.empty:
    col_a, col_b = st.columns(2)
    with col_a:
        # Convert boolean to string for better plotting labels
        plot_df = filtered_df.copy()
        plot_df['Is_Explicit'] = plot_df['Is_Explicit'].replace({True: 'Explicit', False: 'Clean'})
        fig_explicit = px.box(plot_df, x="Is_Explicit", y="Days_on_Playlist", title="Longevity: Explicit vs Clean", color="Is_Explicit")
        st.plotly_chart(fig_explicit, use_container_width=True)
    with col_b:
        fig_album = px.box(filtered_df, x="Album_Type", y="Days_on_Playlist", title="Longevity: Album vs Single", color="Album_Type")
        st.plotly_chart(fig_album, use_container_width=True)

# Core Module: Entry vs Exit Flow (Playlist Churn)
st.subheader("Playlist Churn Analytics (Entry vs Exit Flow)")
if not filtered_df.empty:
    # Group by entry date week/month for churn trends
    churn_df = filtered_df.groupby(filtered_df['Entry_Date'].dt.to_period("W")).size().reset_index(name='New_Entries')
    churn_df['Entry_Date'] = churn_df['Entry_Date'].dt.to_timestamp()
    fig_churn = px.line(churn_df, x='Entry_Date', y='New_Entries', title="Weekly New Entries (Churn Rate)", markers=True)
    st.plotly_chart(fig_churn, use_container_width=True)

# Core Module: Song Lifecycle Timeline Visualizer
st.subheader("Song Lifecycle Timeline (Top 15 Longest Survivors)")
if not filtered_df.empty:
    timeline_df = filtered_df.sort_values(by="Days_on_Playlist", ascending=False).head(15)
    
    # To display song name properly with artist
    timeline_df['Track_Name'] = timeline_df['song'] + " - " + timeline_df['artist']
    
    fig_timeline = px.timeline(timeline_df, x_start="Entry_Date", x_end="Exit_Date", y="Track_Name", 
                               color="Lifecycle_Stage", title="Lifespan of Top Performing Songs",
                               hover_data=["Peak_Position", "Days_on_Playlist"])
    fig_timeline.update_yaxes(autorange="reversed") 
    st.plotly_chart(fig_timeline, use_container_width=True)