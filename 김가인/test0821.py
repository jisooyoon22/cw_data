import os
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling

import math
import scipy.stats as stats
from scipy.stats import ttest_1samp, ttest_rel, ttest_ind

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels
import statsmodels.api as sm

from collections import namedtuple
import altair as alt
plt.rcParams['font.family'] = 'Malgun Gothic'




# Pandas options
pd.options.display.max_rows = 25
pd.options.display.max_columns = 12
pd.options.display.expand_frame_repr = True
pd.options.display.large_repr = 'truncate'
pd.options.display.float_format = '{:.2f}'.format

PAGES = [
    '01. 데이터 가져오기(데이터 목록) Load Data',
    '02. 데이터 탐색하기 Explore Data',
    '03. 단변량 분석 Univariate Analysis',
    '04. 이변량 분석 Bivariate Analysis',
    '05. 다변량 분석 Multivariate Analysis',
    '06. 데이터 변환 Data Transformation' 
]

data_list =[]

@st.cache_data(show_spinner=False)
def load_data(file_path):
    dataset = pd.read_csv(file_path) # index_col=0
    return dataset
    
def run_UI():
    st.set_page_config(
        page_title="CW_miniproject",
        page_icon="🏠",
        initial_sidebar_state="expanded",
        menu_items={
            'Report a bug': "https://github.com/jisooyoon22/cw_data",
            'About': """            
         hello certiware
         with streamlit
         
         This app is the result of hard work by our team:
        
        - Kang Eung Seon
        - Kim Ga In
        - Song Geun
        - Yoon Ji Soo

        Made by Certiware.
        """
        }
    )
    st.sidebar.title('목차')
    
    if st.session_state.page:
        page=st.sidebar.radio('Navigation', PAGES, index=st.session_state.page)
    else:
        page=st.sidebar.radio('Navigation', PAGES, index=0)
    

    data = pd.DataFrame()    
    st.experimental_set_query_params(page=page)
#################################################
    if page == '01. 데이터 가져오기(데이터 목록) Load Data':
        st.sidebar.write("""
    '01. 데이터 가져오기(데이터 목록) Load Data',
        """)
        st.title('01. Load Data')
#################################################
        option = st.selectbox('업로드 파일의 형식?',('csv', 'txt', 'excel'))
        if option == 'csv':
            file_path = st.file_uploader("Select CSV file to upload", type=["csv"])
            st.session_state.fp = file_path
            if 'df' not in st.session_state:
                dataset = load_data(file_path)
                st.session_state.df = dataset
                data_list.append(dataset)
            else :
                st.table(data_list)
        else :
            st.write('You selected:', option, '... but, Format not yet supported')
        
#################################################            
    elif page == '02. 데이터 탐색하기 Explore Data':
        st.sidebar.write("""
    '02. 데이터 탐색하기 Explore Data',
        """)
        st.title('02. Explore Data')
        df = st.session_state.df
#################################################
        st.write(st.session_state.df)
        
        options = st.selectbox(
                    '기능 선택',
                    ['*Choose an option','Change Column','Profiling']
                 )
        
        # Change Columns #
        if options == 'Change Column':
            if st.checkbox('Rename Column'):
                col_list=[]
                for i in range(len(st.session_state.df.columns)):
                    col_list.append(st.session_state.df.columns[i])
                before_col = st.selectbox(
                        'Before Column Name',
                        col_list
                    )
                after_col = st.text_input('After Column Name')
                
                st.write(st.session_state.df.rename(columns={before_col:after_col}))
                
                if st.button('Rename Apply'):
                    st.session_state.df.rename(columns={before_col:after_col},inplace=True)
            
            st.write('')
        
            if st.checkbox('Delete Columns'):
                col_list=[]
                for i in range(len(st.session_state.df.columns)):
                    col_list.append(st.session_state.df.columns[i])
                select_col = st.multiselect(
                        'Delete Column (Multiselect)',
                        col_list
                    )
                st.write(st.session_state.df.drop(columns=select_col))
               
                if st.button('Delete Apply'):
                    st.session_state.df.drop(columns=select_col,inplace=True)

            st.write('')

            if st.checkbox('Columns Type Conversion'):
                col_list = st.session_state.df.columns.tolist()
                select_col = st.multiselect(
                    'Select Columns to Convert Data Types',
                    col_list
                )
                st.write("Selected Columns:", select_col)

                if st.checkbox('Convert Data Types Apply'):
                    for col in select_col:
                        new_dtype = st.selectbox(
                            f'Select New Data Type for Column "{col}"',
                            ['int', 'float', 'object', 'datetime']  
                        )
                        
                        if new_dtype == 'int':
                            st.session_state.df[col] = pd.to_numeric(st.session_state.df[col], errors='coerce', downcast='integer')
                        elif new_dtype == 'float':
                            st.session_state.df[col] = pd.to_numeric(st.session_state.df[col], errors='coerce', downcast='float')
                        elif new_dtype == 'object':
                            st.session_state.df[col] = st.session_state.df[col].astype('object')
                        elif new_dtype == 'datetime':
                            st.session_state.df[col] = pd.to_datetime(st.session_state.df[col], errors='coerce')
                
                

                st.write(st.session_state.df.dtypes)



            st.write('')
            
        # Profiling #
        if options == 'Profiling':
            pr = st.session_state.df.profile_report()
            st_profile_report(pr)

#################################################
    elif page == '03. 단변량 분석 Univariate Analysis':
        st.sidebar.write("""
    '03. 단변량 분석 Univariate Analysis'
        """)        
        st.title('03. Univariate Analysis')
        df = st.session_state.df
#################################################
        # 탭 생성
        tab1, tab2= st.tabs(['데이터 요약', '시각화'])
        
        col_list = df.columns
        # 수치형과 범주형 변수 구분
        int_col = []
        ob_col = []

        for col in col_list:
            if df[col].dtype == 'int64' or df[col].dtype == 'float64':
                int_col.append(col)
            elif df[col].dtype == 'object':
                ob_col.append(col)
        
        

        with tab1:  
            st.write("일변량 분석을 위한 열을 선택하십시오.")
            selected_column = st.selectbox(
                '확인하고 싶은 컬럼을 선택하세요',col_list,
                key='col'
            )
        
            col1, col2, col3 = st.columns(3)
            with col1:
                if selected_column:
                    st.subheader("데이터 요약 정보")
                    st.write(df[selected_column].describe())
            with col2:
                if selected_column:
                    st.subheader("데이터 상위 5개 항목")
                    st.write(df[selected_column].head())
            with col3:
                if selected_column:
                    st.subheader("데이터 하위 5개 항목")
                    st.write(df[selected_column].tail())     

     
        with tab2:

            col_list = df.columns
            # 수치형과 범주형 변수 구분
            int_col = []
            ob_col = []

            for col in col_list:
                if df[col].dtype == 'int64' or df[col].dtype == 'float64':
                    int_col.append(col)
                elif df[col].dtype == 'object':
                    ob_col.append(col)

            # 단변량 그래프
            st.header("단변량 시각화")
            st.write("시각화를 위한 열을 선택하십시오.")

            column1 = st.selectbox('열을 선택하세요', col_list)

            # 수치형 변수를 선택할 경우 히스토그램과 박스플롯만 
            if column1 in int_col:
                chart1 = st.selectbox('보고싶은 차트를 선택하세요',['hist','box'],key='chart')
            # 범주형 변수를 선택할 경우 막대그래프
            elif column1 in ob_col:
                chart1 = st.selectbox('보고싶은 차트를 선택하세요',['count'],key='chart')

            if chart1 == 'box':
                plot=sns.boxplot(df[column1])
                st.pyplot(plot.figure)
            elif chart1 == 'hist':
                plt.figure(figsize=(10, 6))
                plt.hist(df[column1], bins=20,color='skyblue', alpha=0.7)  # 수정된 부분
                plt.xlabel(column1)
                st.pyplot(plt.gcf())
            elif chart1 == 'count':
                value_counts = df[column1].value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                plt.xticks(rotation=45)  
                st.pyplot(fig)            
                
#################################################
    elif page == '04. 이변량 분석 Bivariate Analysis':
        st.sidebar.write("""
    '04. 이변량 분석 Bivariate Analysis'
        """)
        st.title('04. Bivariate Analysis')
        df = st.session_state.df
#################################################
        int_col = []
        ob_col = []
        col_list = df.columns
        for col in col_list:
            if df[col].dtype == 'int64' or df[col].dtype == 'float64':
                int_col.append(col)
            elif df[col].dtype == 'object':
                ob_col.append(col)     
        # 탭 생성
        tab1, tab2, tab3, tab4, tab5 = st.tabs(['통계검정', '상관계수', '선형회귀분석','One_way분산분석','시각화'])

        with tab1:  
            st.write('')

            # 카이제곱 검정
            st.header("카이제곱 검정")
            st.write("카이제곱 분석을 위한 열을 선택하십시오.")

            # 컬럼 선택해서 교차테이블 생성 (생활단계코드, 업종구분_대) 
            chi_square_columns = st.multiselect('두 컬럼을 선택하세요',ob_col)

            if chi_square_columns:
                contingency_table = pd.crosstab(df[chi_square_columns[0]], df[chi_square_columns[1]])
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

                st.subheader("Chi-Square Test:")
                st.write(f"Chi-Square Statistic: 카이제곱 통계량은 독립성 가정 하에서 예상되는 빈도와 실제 관찰된 빈도(교차표에서) 간의 차이를 나타내는 지표입니다. 카이제곱 통계량 값은 {chi2} 입니다.")
                st.write(f"P-value: {p_value}")
                st.write(f"Degrees of Freedom: 자유도는 카테고리 수에서 1을 뺀 값으로,주어진 유의수준에 대한 카이제곱 분포의 임계값을 결정하는 데 도움을 줍니다. 자유도 값은 {dof} 입니다.")
                st.write("Expected Frequencies Table:")
                st.write(expected)
            elif  chi_square_columns is None :
                st.write('컬럼을 선택하세요.')



                
        with tab2: 
            
            # 두 개의 열 선택
            column1 = st.selectbox('첫 번째 열을 선택하세요', int_col)
            column2 = st.selectbox('두 번째 열을 선택하세요', int_col)
            
            if column1 and column2:  # 두 열 중 하나라도 선택되었을 경우
                # 데이터프레임에서 선택한 열 추출
                selected_data = df[[column1, column2]].values

                # 상관계수 계산
                pearson_corr, p_value = stats.pearsonr(selected_data[:, 0], selected_data[:, 1])

                st.subheader('피어슨 상관계수')
                st.write(f"피어슨 상관계수: {pearson_corr:.4f}")
                
                if pearson_corr > 0:
                    st.write(f"두 변수 간 양의 상관관계가 있습니다.")
                elif pearson_corr < 0:
                    st.write(f"두 변수 간 음의 상관관계가 있습니다.")
                else:
                    st.write(f"두 변수 간 상관관계가 없습니다.")
                
                st.write(f"피어슨 상관계수: {pearson_corr:.4f} 에 따라 두 변수 간 유의미한 차이는 존재합니다.")
            else:
                st.write('열을 선택해주세요.')
            
        with tab3:  
            st.header('회귀분석')
            '''
            독립변수와 종속변수가 연속형일 경우 사용, 하나 혹은 그 이상의 원인이 종속변수에 미치는 영향을 분석하는 방법
            선형회귀분석의 가정 : 독립변수와 종속변수 간의 선형성, 오차의 등분산성, 오차의 정규성, 오차의 독립성
            '''
            st.subheader('단순 선형 회귀분석')
            '''단순선형회귀분석 -입력변수와 출력변수간의 선형성을 점검하기위해 산점도를 확인한다.(독립변수의 개수가 1개인 경우에 사용)'''
            column_x = st.selectbox('독립 변수(X)를 선택하세요(연속형)',   int_col, key='col3')
            column_y = st.selectbox('종속 변수(Y)를 선택하세요(연속형)',int_col, key='col4')

            # 데이터프레임에서 선택한 열 추출
            selected_data = df[[column_x, column_y]]

            # 단순 선형 회귀 분석
            if column_x != column_y:
                slope, intercept, r_value, p_value, std_err = stats.linregress(selected_data[column_x], selected_data[column_y])
                # 회귀식
                st.write(f"회귀식: Y = {slope:.4f} * X + {intercept:.4f}")
                # 상관계수
                st.write(f"상관계수: {r_value:.4f}")
                # P-value
                st.write(f"P-value: {p_value:.4f}")
                # 표준 오차
                st.write(f"표준 오차: {std_err:.4f}")
            else:
                st.write('독립변수로 선택한 컬럼은 종속변수로 선택할 수 없습니다.')
        

           
        
        with tab4:  
            st.subheader('일원 배치 (One-way) 분산 분석')
            '''
            두 개 이상의 집단에서 그룹 평균간 차이를 그룹 내 변동에 비교하는 통계분석 방법입니다.
            종속변수는 연속형, 독립변수는 범주형이어야 합니다.
            업종구분_대 별로 이용금액의 평균이 같은지 혹은 차이가 있는지를 확인하기 위해 일원배치 분산분석을 수행해보자.
                '''
            #종속변수를 선택하세요.(연속형)
            dependent_variable = st.selectbox('종속변수를 선택하세요.(연속형)', int_col, key='col5')

            # 독립변수를 선택하세요. (범주형)
            independent_variable = st.selectbox('독립변수를 선택하세요. (범주형)', ob_col, key='col6')

            if st.button('일원 배치 ANOVA 분석 수행') and dependent_variable and independent_variable:
                st.write("일원 배치 ANOVA 분석 수행 결과")

                # 일원 배치 ANOVA 수행
                categories = df[independent_variable].unique()
                data = [df[dependent_variable][df[independent_variable] == cat] for cat in categories]
                anova_result = stats.f_oneway(*data)

                # 일원 배치 결과 출력
                st.write("One-way ANOVA Result:")
                st.write("F-statistic:", anova_result.statistic)
                st.write("p-value:", anova_result.pvalue)
                #결과 해석 
                if anova_result.pvalue < 0.05:
                    st.write(f"p_value값이 0.05보다 작으므로,귀무가설 기각한다. 따라서 {independent_variable}에 따라 {dependent_variable}간 유의미한 차이는 존재한다.")
                else:
                    st.write(f"p_value값이 0.05보다 크므로,귀무가설을 기각하지 못한다. 따라서 {independent_variable}에 따라 {dependent_variable}간 유의미한 차이는 존재하지 않을 가능성이 있다.")
            elif dependent_variable and independent_variable:
                st.write("일원 배치 ANOVA 분석을 수행하려면 위의 버튼을 클릭하십시오.")
        
        
        with tab5:
            # 시각화를 할 두 개의 변수 선택
            column2 = st.multiselect('변수 두 개를 선택하세요', col_list)

            if len(column2) == 2:  # 수정: 두 변수가 모두 선택되었을 때만 시각화 진행
                if column2[0] in int_col:
                    # 선택한 두 개의 변수 모두 수치형일 경우 산점도
                    if column2[1] in int_col:
                        chart2 = st.selectbox('보고싶은 차트를 선택하세요', ['scatter plot'], key='chart2')
                        # 여기서 산점도 시각화를 구현할 수 있음
                    # 선택한 두 개의 변수가 하나는 수치형, 하나는 범주형일 경우 박스플롯
                    else:
                        chart2 = st.selectbox('보고싶은 차트를 선택하세요', ['box plot'], key='chart2')
                        # 여기서 박스플롯 시각화를 구현할 수 있음
                elif column2[0] in ob_col:
                    # 선택한 두 개의 변수가 하나는 수치형, 하나는 범주형일 경우 박스플롯
                    if column2[1] in int_col:
                        chart2 = st.selectbox('보고싶은 차트를 선택하세요', ['box plot'], key='chart2')
                        # 여기서 박스플롯 시각화를 구현할 수 있음
                    # 선택한 두 개의 변수가 모두 범주형일 경우 막대그래프
                    else:
                        chart2 = st.selectbox('보고싶은 차트를 선택하세요', ['count plot'], key='chart2')
                        # 여기서 막대그래프 시각화를 구현할 수 있음


                if chart2 == 'scatter plot':
                    plot=sns.scatterplot(x = column2[0], y = column2[1], data=df)
                    st.pyplot(plot.figure)
                elif chart2 == 'box plot':
                    if column2[0] in ob_col:
                        plot = sns.boxplot(x = column2[0], y = column2[1], data = df)
                        plt.xticks(rotation=45) 
                        st.pyplot(plot.figure)
                    else:
                        plot = sns.boxplot(x = column2[1], y = column2[0], data = df)
                        st.pyplot(plot.figure)
                elif chart2 == 'count plot':
                    plot = sns.countplot(x=column2[0], hue=column2[1], data=df)
                    st.pyplot(plot.figure)
        
            elif len(column2) == 1:
                st.write('두 개의 컬럼을 선택해주세요.')
            else:
                st.write('컬럼을 선택하세요.')

                      
#################################################        
    elif page == '05. 다변량 분석 Multivariate Analysis':
        st.sidebar.write("""
    '05. 다변량 분석 Multivariate Analysis'
        """)        
        st.title('05. Multivariate Analysis')
        df = st.session_state.df
#################################################
        int_col = []
        ob_col = []
        col_list = df.columns
        for col in col_list:
            if df[col].dtype == 'int64' or df[col].dtype == 'float64':
                int_col.append(col)
            elif df[col].dtype == 'object':
                ob_col.append(col)  
        
          
        # 탭 생성
        tab1, tab2, tab3 = st.tabs( ['다중회귀분석','Two-way 분산 분석','시각화'])
        
        
        with tab1:           
            st.subheader('다중 선형 회귀분석')
        '''다중선형회귀분석은 선형회귀분석의 가정을 모두 만족하는지 확인해야한다.(독립변수가 2 개 이상인 경우에 사용)'''

        dependent_column = st.selectbox('종속 변수(Y)를 선택하세요(연속형)', int_col, key='col7') # 종속 변수 열 이름을 지정해야 합니다
        independent_columns = st.multiselect('독립변수 컬럼을 선택하세요(연속형)',int_col)
        
            # 선택한 독립 변수들과 종속 변수
        if independent_columns and dependent_column:
            # 데이터프레임에서 선택한 열 추출
            selected_data = df[independent_columns + [dependent_column]]

            # 상수항 추가
            X = sm.add_constant(selected_data[independent_columns])
            y = selected_data[dependent_column]

            # 다중 선형 회귀 모델 생성 및 학습
            model = sm.OLS(y, X).fit()

            # 결과 출력
            st.write("다중 선형 회귀 분석 결과:")
            st.write(model.summary())
        else:
            st.write("독립변수와 종속변수를 선택해주세요.")
        
        
        with tab2: 

            st.subheader('이원 배치 (Two-way) 분산 분석')
            '''
            두 개 이상의 집단에서 그룹 평균간 차이를 그룹 내 변동에 비교하는 통계분석 방법입니다.
            종속변수는 연속형, 독립변수는 범주형이어야 합니다.
            업종구분_대 별로 이용금액의 평균이 같은지 혹은 차이가 있는지를 확인하기 위해 일원배치 분산분석을 수행해보자.
                '''
            
            # 종속변수를 선택하세요.(연속형)
            dependent_variable = st.selectbox('종속변수를 선택하세요.(연속형)', int_col, key='col8')
            # 독립변수를 선택하세요. (범주형)
            independent_variable = st.multiselect('독립변수를 선택하세요. (범주형)',  ob_col)


            if dependent_variable and independent_variable:  # 두 열 중 하나라도 선택되었을 경우
                if st.button('이원 배치ANOVA 분석 수행') and dependent_variable and independent_variable:
                    st.write("이원 배치 ANOVA 분석 수행 결과")

                    # 이원 배치 ANOVA 수행 statsmodels 사용
                    model = statsmodels.formula.api.ols(f'{dependent_variable} ~ C({independent_variable[0]}) + C({independent_variable[1]})', data=df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    st.write(anova_table)

                    p_value0 = anova_table.loc[f"C({independent_variable[0]})", 'PR(>F)']
                    st.write(f"P-value: {p_value0}")
                    # 결과 해석 
                    if p_value0 < 0.05:
                        st.write(f"p_value값이 0.05보다 작으므로, 귀무가설 기각한다. 따라서 {independent_variable[0]}에 따라 {dependent_variable}간 유의미한 차이는 존재한다.")
                    else:
                        st.write(f"p_value값이 0.05보다 크므로, 귀무가설을 기각하지 못한다. 따라서 {independent_variable[0]}에 따라 {dependent_variable}간 유의미한 차이는 존재하지 않을 가능성이 있다.")

                    p_value1 = anova_table.loc[f"C({independent_variable[1]})", 'PR(>F)']
                    st.write(f"P-value: {p_value1}")
                    # 결과 해석 
                    if p_value1 < 0.05:
                        st.write(f"p_value값이 0.05보다 작으므로, 귀무가설 기각한다. 따라서 {independent_variable[1]}에 따라 {dependent_variable}간 유의미한 차이는 존재한다.")
                    else:
                        st.write(f"p_value값이 0.05보다 크므로, 귀무가설을 기각하지 못한다. 따라서 {independent_variable[1]}에 따라 {dependent_variable}간 유의미한 차이는 존재하지 않을 가능성이 있다.")
            else:
                st.write('열을 선택해주세요.')

        with tab3:
                
            # 다변량 변수의 경우 수치형만을 시각화
            # 범주형을 고려하여 시각화할 겨우 범주형의 카테고리 개수만큼 산점도 행렬을 생성해야함. 

            selected_columns = st.multiselect( '확인하고 싶은 컬럼을 선택하세요', int_col, key='col2')
            if len(selected_columns) >=3:  # 수정: 두 변수가 모두 선택되었을 때만 시각화 진행
                # 보고싶은 차트를 선택하는 셀렉트 박스
                # 산점도 행렬, 히트맵
                selected_chart = st.selectbox('보고싶은 차트를 선택하세요',['pairplot', 'heatmap'],key='chart2')

                # 산점도 행렬
                if selected_chart == 'pairplot':
                    plot = sns.pairplot(df[selected_columns], diag_kind='hist')
                    st.pyplot(plot.fig)

                # 히트맵
                elif selected_chart == 'heatmap':
                    df_corr = df[selected_columns].corr()
                    plot = sns.heatmap(df_corr, xticklabels = df_corr.columns, yticklabels = df_corr.columns, annot=True)
                    st.pyplot(plot.fig)
                    st.header('05.다변량 분석')
            else:
                st.write("변수를 3개 이상 선택하세요.")




#################################################
    elif page == '06. 데이터 변환 Data Transformation':
        st.sidebar.write("""
    '06. 데이터 변환 Data Transformation'
        """)
        st.title('06. Data Transformation')
#################################################
        options = st.selectbox(
                    '기능 선택',
                    ['*Choose an option','Missing Value','Outlier','Scaling']
                 )
        
        # 컬럼별로 결측치의 개수(count)와 비율(percent) 확인
        missing_df=st.session_state.df.isnull().sum().reset_index()
        missing_df.columns=['column','count']
        missing_df['percent']=round((missing_df['count']/st.session_state.df.shape[0])*100,2)
        missing_df=missing_df.loc[missing_df['percent']!=0].sort_values('percent',ascending=False)
        
        missing_50per=missing_df.loc[missing_df['percent']>=50].sort_values('percent',ascending=False).reset_index(drop=True)
        missing_50per['percent']=missing_50per['percent'].astype(str) + '%'
        
        missing_1050per=missing_df.loc[(missing_df['percent']>=10)&(missing_df['percent']<50)].sort_values('percent',ascending=False).reset_index(drop=True)
        missing_1050per['percent']=missing_1050per['percent'].astype(str) + '%'
        
        missing_10per=missing_df.loc[missing_df['percent']<10].sort_values('percent',ascending=False).reset_index(drop=True)
        missing_10per['percent']=missing_10per['percent'].astype(str) + '%'
        
        # Missing Value #
        if options == 'Missing Value':
            if st.checkbox('결측치 확인'):
                tab1,tab2,tab3= st.tabs(['10% 미만','10%~50%','50% 이상'])
                with tab1:
                    st.write(missing_10per)
                with tab2:
                    st.write(missing_1050per)
                with tab3:
                    st.write(missing_50per)
            st.write('')
            st.write('')
            st.write('')
            
            if st.checkbox('결측치 처리'):
                st.subheader('-GuideLine-')
                st.write('- 10% 미만 : 삭제 or 대치')
                st.write('- 10% ~ 50% : regression or model based imputation')
                st.write('- 50% 이상 : 해당 컬럼(변수) 자체 제거')
                
                missing_col=[]
                for i in range(len(missing_df['column'].values)):
                    missing_col.append(missing_df['column'].values[i])
                
                tab1,tab2,tab3,tab4= st.tabs(['Dropna','Drop Columns','Fillna','Regression'])
                ## Dropna ##
                with tab1:
                    select_col = st.multiselect(
                        'Select Columns',
                        missing_col,
                        key='dropna'
                    )
                    #if len(st.session_state.df.columns[st.session_state.df.isnull().any()])==0:
                    #    st.subheader('There is no Missing Value')
                    #else:
                    #    st.write(st.session_state.df.dropna(axis=0, subset=select_col))
                    #    st.write(st.session_state.df.dropna(axis=0, subset=select_col).shape)
                    st.write(st.session_state.df.dropna(axis=0, subset=select_col))
                    st.write(st.session_state.df.dropna(axis=0, subset=select_col).shape)
                    if st.button('Dropna Apply'):
                        st.session_state.df.dropna(axis=0, how='any', subset=select_col, inplace=True)
                ## Drop Columns ##
                with tab2:
                    col_list=[]
                    for i in range(len(st.session_state.df.columns)):
                        col_list.append(st.session_state.df.columns[i])
                    select_col = st.multiselect(
                            'Select Columns',
                            missing_col,
                            key='drop column'
                        )
                    st.write(st.session_state.df.drop(columns=select_col))
                    st.write(st.session_state.df.drop(columns=select_col).shape)
                   
                    if st.button('Drop Columns Apply'):
                        st.session_state.df.drop(columns=select_col,inplace=True)
                ## Fillna ##
                with tab3:
                    select_col = st.selectbox(
                        'Select Column',
                        missing_col,
                        key='fillna'
                    )
                    mean = round(st.session_state.df[select_col].mean(),2)
                    median = round(st.session_state.df[select_col].median(),2)
                    
                    col1,col2 = st.columns([1,1])
                    with col1:
                        st.write(f'{select_col} 컬럼의 평균값 : ', mean)
                        st.write(st.session_state.df[select_col].fillna(mean))
                        if st.button('Mean Apply', key='mean'):
                            st.session_state.df[select_col].fillna(mean, inplace=True)
                    with col2:
                        st.write(f'{select_col} 컬럼의 중위값 : ', median)
                        st.write(st.session_state.df[select_col].fillna(median))
                        if st.button('Median Apply', key='median'):
                            st.session_state.df[select_col].fillna(median, inplace=True)
                ## Regression ##
                with tab4:
                    select_col = st.multiselect(
                        'Select Column',
                        missing_col,
                        key='regression'
                    )
                    
        
        if options == 'Outlier':
            if st.checkbox('이상치 확인'):
                st.write('이상치 확인')
            
            if st.checkbox('이상치 처리'):
                st.write('이상치 처리')
    
        # Scaling #
        if options == 'Scaling':
            if st.checkbox('스케일링'):
                st.write('MinMax')
                st.write('Standard')
    
#################################################
    elif page == '09. 결론 Conclusion':
        st.sidebar.write("""
    '09. 결론 Conclusion'
        """)
        st.title('09. Conclusion')
        df = st.session_state.df        
#################################################
    #elif page == '06. 데이터 변환 Data Transformation':
    #    st.sidebar.write("""
    #'06. 데이터 변환 Data Transformation'
    #    """)     
    #    st.title('06. Data Transformation')
#################################################
    else:
        st.sidebar.write("""
            error?
        """)
        st.title("Data Explorer")
#################################################


if __name__ == '__main__':
    st.experimental_set_query_params(page='01. 데이터 가져오기(데이터 목록) Load Data')
    url_params = st.experimental_get_query_params()
    st.session_state.page = PAGES.index(url_params['page'][0])
    
    
    #st.session_state['data_type'] = 'County Level'
    #st.session_state['data_format'] = 'Raw Values'
    #st.session_state['loaded'] = False
    run_UI()
