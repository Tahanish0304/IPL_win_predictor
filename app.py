import pip
import streamlit as st
import pickle
import pandas as pd

if hasattr(pip, 'main'):
    pip.main(['install', "scikit-learn"])
else:
    pip._internal.main(['install', "scikit-learn"])

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Gujarat Titans',
 'Lucknow Super Giants',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals',
 'Punjab Kings']

cities = ['Hyderabad', 'Rajkot', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata',
       'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai', 'Cape Town',
       'Port Elizabeth', 'Durban', 'Centurion', 'East London',
       'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad',
       'Dharamsala', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah',
       'Cuttack', 'Visakhapatnam', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl','rb'))

st.title('IPL WIN PERCENT PREDICTOR')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the Batting Team',sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the Bowling Team',sorted(teams))

selected_city = st.selectbox('Select Home City',sorted(cities))

target = st.number_input('Target', value=0, step=1, format='%d')


col3, col4, col5 =st.columns(3)

with col3:
    score = st.number_input('Score', value=0, step=1, format='%d')

with col4:
    overs = st.number_input('Overs completed', value=0, min_value=0, max_value=19, step=1, format='%d')

with col5:
    wickets = st.number_input('Wickets', value=0, min_value=0, max_value=9, step=1, format='%d')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],
                  'balls_left':[balls_left],'wickets_left':[wickets],'total_runs_x':[target],'crr':[crr],
                  'rrr':[rrr]})

   

    result = pipe.predict_proba(input_df)

    loss = result[0][0]
    win = result[0][1]

    st.header(batting_team + "- " + str(round(win*100)))
    st.header(bowling_team + "- " + str(round(loss*100)))


































