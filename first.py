from calendar import c
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
import seaborn as sns

with st.echo(code_location='below'):

    def main():
        st.title("Проект по визуализации данных")
        covid_dataset = pd.read_csv('COVID-19_Coronavirus.csv')
        country_names_covid = covid_dataset['Country'].values
        """## Coronavirus Statistics"""
        """В этом разделе Вы можете посмотреть статистику по показателям, связанных с коронавирусом. 
        Согласно данным на Kaggle информация в датасете была обновлена месяц назад."""
        column_types = np.array(['Total Cases', 'Total Deaths', 'Total cases per one million population',
                                 'Total Deaths per one million population', 'Death percentage'])
        selected_table = st.radio('Please, select the type of information', column_types)
        selected_countries_covid = st.multiselect('Please, select the countries', country_names_covid)
        figure = plt.figure()
        figure, ax = plt.subplots()
        ax.set(xlabel='Country', ylabel=f'{selected_table}')
        covid_dataset.set_index('Country', inplace=True)
        plt.bar(selected_countries_covid, covid_dataset.loc[selected_countries_covid, selected_table], color='#EA13D9')
        st.pyplot(plt, clear_figure=True)

        """## Nobel prize winners"""
        """Данная визуализация предлагает Вам посмотреть на связь года  полкчения Нобелевской премии и возраста лауреата.
        Посмотреть такие данные меня промотивировала интересная статья от BBC "Почему Нобелевские лауреаты становятся все старше?". 
        Вы можете попробовать посмотреть различные выборки данных, например, выбрать одну категорию номинаций и 
        посмотреть, действительно ли возраст лауреатов имеет тенденцию увеличиваться с годами. Кроме того, Вы можете посмотреть
        интересные Вам данных для разных стран."""
        nobel_dataset = pd.read_csv('archive.csv')
        nobel_dataset.dropna(subset=['Birth Date'], inplace=True)
        for i in nobel_dataset.index:
            nobel_dataset.loc[i, 'Birth Date'] = int(nobel_dataset.loc[i, 'Birth Date'][:4])
        country_names_nobel = nobel_dataset['Birth Country'].unique()
        selected_countries_nobel = country_names_nobel
        category_nobel = nobel_dataset['Category'].unique()
        selected_categories_nobel = category_nobel
        all_countries = st.checkbox('All countries')
        if not all_countries:
            selected_countries_nobel = st.multiselect('Please, select the countries', country_names_nobel)
        all_categories = st.checkbox('All categories')
        if not all_categories:
            selected_categories_nobel = st.multiselect('Please, select the categories', category_nobel)
        current_nob = nobel_dataset.loc[nobel_dataset['Birth Country'].isin(selected_countries_nobel)]
        current_nob = current_nob.loc[current_nob['Category'].isin(selected_categories_nobel)]
        current_nob = current_nob.loc[current_nob['Category'].isin(selected_categories_nobel)]
        colors = pd.Series({category_nobel[0]: '#1f77b4', category_nobel[1]: '#ff7f0e', category_nobel[2]: '#2ca02c', category_nobel[3]: '#d62728', category_nobel[4]: '#9467bd', category_nobel[5]: '#8c564b'})
        fig=plt.figure()
        sns.scatterplot(x= current_nob['Year'], hue = 'Category', y = current_nob['Year'] - current_nob['Birth Date'], data= current_nob)
        st.pyplot(fig, clear_figure=True)

        """## Spotify Top 100 Songs of 2010-2019"""
        """В данном разделе используется датасет, связанный с музыкой, которая находится в топ-100 Spotify 2010-2019.
        Вы можете посмотреть и выяснить, как с годами менялись предпочтения пользователей Spotify касательно жанра музыки."""
        st.subheader("Genres of music")
        df = pd.read_csv('dataset.csv')
        pie_df = df.groupby(['top genre', 'top year', 'artist type']).size().reset_index(name='count')
        year = st.slider('Year', min_value=int(pie_df['top year'].min()), max_value=int(pie_df['top year'].max()),
                         step=1)
        artists = st.multiselect('Artist type', options=pie_df['artist type'].unique(),
                                 default=pie_df['artist type'].unique())
        fig = px.pie(pie_df.loc[(pie_df['top year'] == year) & (pie_df['artist type'].isin(artists))], values='count',
                     names='top genre', title=f'Genres distribution in {year} for {", ".join(artists)} artists')
        fig.update_traces(textposition='inside')
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Types of music")
        """Следующая визуализация поможет Вам выяснить, какой тип музыки предпочитается пользователями Spotify в каждом году.
        Под типом подразумевается характеристика музыки: энергичная (nrgy), танцевальная (dnce), с множеством битов (bpm - beats per minute), 
        громкая (db - decibel). Эта визуализация интерактивная, поэтому Вы можете поближе рассмотреть, 
        на каком именно уровне определенного параметра самая высокая концентация музыки для каждого года!
        *Оценка всех параметров имеет разную систему, но, несмотря на это, легко сравнить динамику и показатели по каждому параметру на графике."""

        parameter = st.selectbox('Please, select the parameter', options=['bpm', 'nrgy', 'dnce', 'dB'], index=0)
        st.altair_chart(alt.Chart(df.loc[df['year released'] != 1975], autosize='fit').mark_circle().encode(
            alt.X('year released', scale=alt.Scale(zero=False)),
            alt.Y(parameter, scale=alt.Scale(zero=False)),
            color='year released').properties(
            width=500,
            height=500,
        ).interactive())
        """## Fast food market"""
        """В этой части проекта Вы можете исследовать различные данные для рынка fast food в США. 
        Для разных отраслей индустрии fast food в США Вы можете найти данные по различным экономическим показателям и 
        посмотреть, какие компании являются лидерами в отрасли, какую долю от рынка занимает каждый производитель."""
        food_dataset = pd.read_csv('top_50_fast_food_US.csv')
        category_of_compare = st.selectbox('Please, select the categories', food_dataset.columns[2:-1])
        current_pie_df = pd.DataFrame()
        selected_category_food = st.selectbox('category', food_dataset['category'].unique())
        current_pie_df = food_dataset.loc[food_dataset['category'] == selected_category_food]
        current_pie_df = current_pie_df[['company', category_of_compare]]
        colors=sns.color_palette('pastel')
        plt.pie(current_pie_df[category_of_compare],  labels=current_pie_df['company'], textprops={"fontsize":5},
                autopct='%.1f', pctdistance=0.8, colors=colors)
        plt.legend(loc='upper right', fontsize = 5, title='Companies:', title_fontsize=6)
        st.pyplot(plt, clear_figure=True)

    if __name__ == '__main__':
        main()
