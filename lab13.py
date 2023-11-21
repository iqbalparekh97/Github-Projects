import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from textblob import TextBlob

plt.style.use('ggplot')

import nltk
import streamlit as st


## Reading the clean data
df=pd.read_csv('C:\\Users\\Hp\\Desktop\\Python-FinalProject\\spotify-cleaned_data.csv')

##Scaling down the size of reviews for MVP
df=df.head(500)
df.head()

# Calculate the correlation coefficient
correlation = df['Rating'].corr(df['Total_thumbsup'])

# Print the correlation coefficient
#print(correlation)


# Create the plot
# Calculate the correlation coefficient
correlation = df['Rating'].corr(df['Total_thumbsup'])

# Create the Streamlit app
st.title('Correlation between Thumbsup and Rating')

# Display the correlation coefficient
st.write('Correlation coefficient:', correlation)

# Create the plot
fig, ax = plt.subplots()
ax.plot(df['Total_thumbsup'], df['Rating'], label='Relation between Rating and Thumbsup')

# Add labels and title
ax.set_xlabel('Total_thumbsup')
ax.set_ylabel('Rating')
ax.set_title('Line Graph')

# Add legend
ax.legend()

# Add the plot to the Streamlit app
st.pyplot(fig)
subset_df = df.iloc[:10, :2]
st.dataframe(subset_df)