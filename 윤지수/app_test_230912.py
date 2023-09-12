# Anaconda Prompt 실행문 : streamlit run C:\Users\User\Desktop\app_test230912.py
# streamlit run "C:\Users\rlark\Desktop\stremlit\app_test230912.py"
import os
import numpy as np
import pandas as pd
import streamlit as st
#pip install streamlit-option-menu
#from streamlit_option_menu import option_menu
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
    '데이터 가져오기',
    '데이터 탐색하기',
    '통계분석 및 시각화',
    '데이터 변환' 
]

data_list =[]

@st.cache_data(show_spinner=False) 
def load_data(file_path):
    dataset = pd.read_csv(file_path, encoding='cp949') # index_col=0
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
    if page == '데이터 가져오기':
        st.sidebar.write("""
    'Load Data',
        """)
        st.title('데이터 가져오기')
#################################################
        option = st.selectbox('업로드 파일의 형식?',('csv', 'txt', 'excel'))
        if option == 'csv':
            file_path = st.file_uploader("Select CSV file to upload", type=["csv"])
            if file_path is not None:
                st.session_state.fp = file_path
            
                if 'df' not in st.session_state:
                    dataset = load_data(file_path)
                    st.session_state.df = dataset
                    data_list.append(dataset)
                else :
                    st.table(data_list)
            else :
                st.write('데이터 파일을 추가하세요.')
        else :
            st.write('You selected:', option, '... but, Format not yet supported')
            
#################################################            
    elif page == '데이터 탐색하기':
        st.sidebar.write("""
    'Explore Data',
        """)
        st.title('데이터 탐색하기')
        df = st.session_state.df
#################################################
        # 탭 생성
        tab1, tab2, tab3= st.tabs(['데이터 탐색하기', '데이터 요약', '시각화'])
        with tab1:
            options = st.selectbox(
                        '*기능 선택',
                        ['*Choose an option','컬럼 수정','프로파일링']
                     )
                     
            # 컬럼 수정 #
            if options == '컬럼 수정':
                st.write('')
                tab1_1,tab1_2,tab1_3= st.tabs(['컬럼명 변경','불필요한 컬럼 삭제','컬럼 형변환']) 
                with tab1_1:
                    col_list=[]
                    for i in range(len(st.session_state.df.columns)):
                        col_list.append(st.session_state.df.columns[i])
                    before_col = st.selectbox(
                            '*변경 전 컬럼명',
                            col_list
                        )
                    after_col = st.text_input('*변경 후 컬럼명')
                    
                    st.write(st.session_state.df.rename(columns={before_col:after_col}))
                    
                    if st.button('컬럼명 변경하기'):
                        st.session_state.df.rename(columns={before_col:after_col},inplace=True)
                
                st.write('')
                with tab1_2:
                    col_list=[]
                    for i in range(len(st.session_state.df.columns)):
                        col_list.append(st.session_state.df.columns[i])
                    select_col = st.multiselect(
                            '*불필요한 컬럼(중복 선택 가능)',
                            col_list
                        )
                    st.write(st.session_state.df.drop(columns=select_col))
                   
                    if st.button('불필요한 컬럼 삭제하기'):
                        st.session_state.df.drop(columns=select_col,inplace=True)
                    
                st.write('')    
                with tab1_3:
                    st.write(st.session_state.df.dtypes)
                    col_list = st.session_state.df.columns.tolist()
                    select_col = st.multiselect(
                        'Select Columns to Convert Data Types',
                        col_list
                    )
                    
                    for col in select_col:
                        new_dtype = st.selectbox(
                            f'Select New Data Type for Column "{col}"',
                            ['int', 'float', 'object', 'datetime']  
                        )
                        
                        if st.button('컬럼 형변환하기'):
                            if new_dtype == 'int':
                                st.session_state.df[col] = pd.to_numeric(st.session_state.df[col], errors='coerce', downcast='integer')
                            elif new_dtype == 'float':
                                st.session_state.df[col] = pd.to_numeric(st.session_state.df[col], errors='coerce', downcast='float')
                            elif new_dtype == 'object':
                                st.session_state.df[col] = st.session_state.df[col].astype('object')
                            elif new_dtype == 'datetime':
                                st.session_state.df[col] = pd.to_datetime(st.session_state.df[col], errors='coerce')

                    st.write('')
                
        
            # 프로파일링 #
            if options == '프로파일링':
                pr = st.session_state.df.profile_report()
                st_profile_report(pr)
                
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

        with tab3:

                tab3_1, tab3_2, tab3_3 = st.tabs(['단변량 시각화', 'pairplot', 'heatmap'])
                col_list = df.columns
                # 수치형과 범주형 변수 구분
                int_col = []
                ob_col = []

                for col in col_list:
                    if df[col].dtype == 'int64' or df[col].dtype == 'float64':
                        int_col.append(col)
                    elif df[col].dtype == 'object':
                        ob_col.append(col)

                with tab3_1:
                    st.write('')


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

                with tab3_2 : 
                    st.write('')

                    # pairplot
                    st.header('pairplot')
                    plot = sns.pairplot(df, diag_kind='hist')
                    st.pyplot(plot.fig)

                with tab3_3:
                    st.write('')

                    # pairplot
                    st.header('heatmap')
                    # df_corr = df.corr()
                    # plot = sns.heatmap(df_corr, xticklabels = df_corr.columns, yticklabels = df_corr.columns, annot=True)
                    # st.pyplot(plot.fig)


                    df_corr = df.corr()
                    fig, ax = plt.subplots()
                    sns.heatmap(df_corr, xticklabels=df_corr.columns, yticklabels=df_corr.columns, annot=True, ax=ax)
                    st.pyplot(fig)

                
#################################################
    elif page == '통계분석 및 시각화':
        st.sidebar.write("""
    'Statistical Analysis & Visualization'
        """)        
        st.title('통계분석 및 시각화')
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
        
        tab1,tab2,tab3 = st.tabs(['이변량 분석','다변량 분석','시각화'])
        with tab1:
            
            tab1_1,tab1_2,tab1_3,tab1_4 = st.tabs(['통계검정', '상관계수', '선형회귀분석','One_way분산분석'])
            with tab1_1:
                st.write('')

                # 카이제곱 검정
                st.header("카이제곱 검정")

                st.markdown('카이제곱 검정은 두 :blue[범주형] 변수에 대한 분석 방법입니다.')
                st.write('예를 들면, 성별에 따른 정당 지지율 비교 문제가 이에 해당합니다.')
                st.write("카이제곱 분석을 위한 열을 선택하십시오.")
                
                # 컬럼 선택해서 교차테이블 생성 (생활단계코드, 업종구분_대) 
                chi_square_columns = st.multiselect('두 컬럼을 선택하세요',ob_col)

                if  len(chi_square_columns) == 2:
                    contingency_table = pd.crosstab(df[chi_square_columns[0]], df[chi_square_columns[1]])
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

                    st.markdown("카이제곱 검정 결과")
                    st.markdown(f"Chi-Square Statistic: 카이제곱 통계량은 독립성 가정 하에서 예상되는 빈도와 실제 관찰된 빈도(교차표에서) 간의 차이를 나타내는 지표입니다.")
                    st.markdown(f"카이제곱 통계량 값은 **{chi2}** 입니다.")
                    st.markdown(f"Degrees of Freedom: 자유도는 카테고리 수에서 1을 뺀 값으로,주어진 유의수준에 대한 카이제곱 분포의 임계값을 결정하는 데 도움을 줍니다.")
                    st.markdown(f"자유도 값은 **{dof}** 입니다.")
                    if p_value < 0.05:
                        st.write("P-value 해석: P-value가 유의수준 (예: 0.05) 보다 작습니다.")
                        st.write("해석: 주어진 검정에서는 귀무 가설을 기각할 충분한 증거가 있으며, 결과는 통계적으로 유의합니다.")
                    elif p_value >= 0.05:
                        st.write("P-value 해석: P-value가 유의수준 (예: 0.05) 이상입니다.")
                        st.write("해석: 주어진 검정에서는 귀무 가설을 기각할 충분한 증거가 없으며, 결과는 통계적으로 유의하지 않습니다.")
                    else:
                        st.write("P-value 해석: P-value가 유의수준과 비슷하거나 올바르게 설정되지 않았습니다.")
                        st.write("해석: 유의수준을 다시 고려하고 결과를 해석하세요.")

                        st.write("Expected Frequencies Table:")
                        st.write(expected)

                elif len(chi_square_columns) == 1:
                    st.write('두 개의 컬럼을 선택해주세요.')
                else:
                    st.write('컬럼을 선택하세요.')
        
            with tab1_2:
                # 상관계수 검정
                st.subheader("상관계수")
                st.markdown('피어슨 상관계수(Pearson Correlation Coefficient ,PCC)란 두 변수 X 와 Y 간의 선형 상관관계를 계량화한 수치다.')
                st.write('상관계수의 값은 항상 **+1과 -1 ** 사이의 값을 가집니다.')
                st.write('**+1**은 완벽한 양의 선형 상관관계, **0**은 선형 상관 관계 없음, **-1**은 완벽한 음의 선형 상관 관계를 의미한다.')
                st.markdown('상관계수의 절대값이 **클수록**, 즉 상관계수의 1또는 -1에 가까울 수록 **연관성이 크고**, 0에 가까울 수록 연관성이 **매우 약함**을 의미합니다. ')
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
                        st.write(f"두 변수 간 양의 상관관계가 있습니다. 이는 한 변수의 증가가 다른 변수의 증가와 관련되어 있음을 나타냅니다.")
                    elif pearson_corr < 0:
                        st.write(f"두 변수 간 음의 상관관계가 있습니다. 이는 한 변수의 증가가 다른 변수의 감소와 관련되어 있음을 나타냅니다.")
                    else:
                        st.write(f"두 변수 간 상관관계가 없습니다.")

                    # 상관계수의 유의사항 추가
                    st.write("상관계수에 대한 유의사항: 상관계수는 두 변수 간의 선형 관계를 측정하며 인과관계를 나타내지 않습니다.")
                else:
                    st.write('열을 선택해주세요.')

            with tab1_3:  
                st.subheader('회귀분석')
                st.markdown("독립변수와 종속변수가 :red[연속형]일 경우 사용, 하나 혹은 그 이상의 원인이 종속변수에 미치는 영향을 분석하는 방법입니다.")
                st.markdown('선형회귀분석의 가정 : 독립변수와 종속변수 간의 _선형성_, _오차의 등분산성_, _오차의 정규성_, _오차의 독립성_ .')

                st.subheader('단순 선형 회귀분석')
                st.markdown('다중선형회귀분석은 선형회귀분석의 가정을 모두 만족하는지 확인해야합니다.(독립변수가 2 개 이상인 경우에 사용)')
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


            with tab1_4:
                st.subheader('일원 배치 (One-way) 분산 분석')
                st.markdown("일원배치 분산분석(One-way ANOVA)은 세 개 이상 집단 간 평균을 비교하는 통계 검정 방법입니다. 독립변수가 세 집 단 이상으로 구성된 :blue[범주형] 자료, 종속변수가 :red[연속형] 자료인 경우에 활용합니다.")
                st.markdown("집단을 나타내는 변수인 요인의 수, 즉 독립변수가 1개인 경우 일원배치 분산분석이라고 합니다.")

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
                        st.write(f"p_value값이 0.05보다 작으므로,귀무가설 기각한다. 따라서 **{independent_variable}**에 따라 **{dependent_variable}**간 유의미한 차이는 존재한다.")
                    else:
                        st.write(f"p_value값이 0.05보다 크므로,귀무가설을 기각하지 못한다. 따라서 **{independent_variable}**에 따라 {dependent_variable}**간 유의미한 차이는 존재하지 않을 가능성이 있다.")
                elif dependent_variable and independent_variable:
                    st.write("일원 배치 ANOVA 분석을 수행하려면 위의 버튼을 클릭하십시오.")
        
        with tab2:
        
            tab2_1, tab2_2 = st.tabs( ['다중회귀분석','Two-way 분산 분석'])
            with tab2_1:
                st.subheader('다중 선형 회귀분석')
                st.markdown('다중선형회귀분석은 선형회귀분석의 가정을 모두 만족하는지 확인해야합니다.(독립변수가 2 개 이상인 경우에 사용)')

                dependent_column = st.selectbox('종속 변수(Y)를 선택하세요(연속형)', int_col, key='col7') # 종속 변수 열 이름을 지정해야 합니다
                independent_columns = st.multiselect('독립변수 컬럼을 선택하세요(연속형)',int_col)
                
                    # 선택한 독립 변수들과 종속 변수
                if independent_columns and dependent_column:
                    # 데이터프레임에서 선택한 열 추출
                    selected_data = df[independent_columns + [dependent_column]]

                    # 상수항 추가
                    X = sm.add_constant(df[independent_columns])
                    y = df[dependent_column]

                    # 다중 선형 회귀 모델 생성 및 학습
                    model = sm.OLS(y, X).fit()

                    # 회귀식 생성
                    coefficients = model.params  # 회귀 계수 추출
                    intercept = coefficients['const']  # 상수항
                    coefficients = coefficients.drop('const')  # 상수항 제거

                    # 회귀식 출력
                    st.write("다중 선형 회귀식:")
                    regression_equation = f"{intercept:.2f} + "
                    for col, coef in coefficients.items():
                        regression_equation += f"{coef:.2f}*{col} + "
                    regression_equation = regression_equation[:-2] 
                    st.write(regression_equation)

                    # 결과 해석
                    st.write("다중 선형 회귀 분석 결과:")
                    st.write(model.summary())
                else:
                    st.write("독립변수를 2개 이상 선택해주세요.")

            with tab2_2:
                st.subheader('이원 배치 (Two-way) 분산 분석')
                st.markdown('이원배치 분산분석(TWO-Way ANOVA)은 **2 개의 독립변수에 따라 종속변수의 평균 차이를 검증**하고, 2개의 독립변수 간 **상호작용 효과를 검증**하는 방법입니다.')
                st.markdown('독립변수가 2개인 경우 이원배치 분산분석이라고 합니다.')
         
                
                # 종속변수를 선택하세요.(연속형)
                dependent_variable = st.selectbox('종속변수를 선택하세요.(연속형)', int_col, key='col8')
                # 독립변수를 선택하세요. (범주형)
                independent_variables = st.multiselect('독립변수를 선택하세요. (범주형)',  ob_col)


                if dependent_variable and independent_variables:
                    st.write(f"이원 배치 ANOVA 분석 결과 for {dependent_variable}:")

                    for independent_variable in independent_variables:
                        st.write(f"독립 변수: {independent_variable}")

                        # 이원 배치 ANOVA 수행 statsmodels 사용
                        formula = f'{dependent_variable} ~ C({independent_variable}) + C({independent_variables[0]})'  #이원분산분석 코드 수정
                        model = sm.formula.ols(formula, data=df).fit()
                        anova_table = sm.stats.anova_lm(model, typ=2)
                        st.write(anova_table)

                        p_value = anova_table.loc[f"C({independent_variable})", 'PR(>F)']
                        st.write(f"P-value for {independent_variable}: {p_value}")

                        # 결과 해석 
                        if p_value < 0.05:
                            st.write(f"p-value 값이 0.05보다 작으므로, 귀무가설 기각합니다. 따라서 **{independent_variable}**에 따라 **{dependent_variable}** 간에 유의미한 차이가 존재합니다.")
                        else:
                            st.write(f"p-value 값이 0.05보다 크므로, 귀무가설을 기각하지 못합니다. 따라서 **{independent_variable}**에 따라 **{dependent_variable}** 간에 유의미한 차이가 존재하지 않을 가능성이 있습니다.")
                else:
                    st.write('열을 선택해주세요.')
               
            
        with tab3: # 그래프 X,Y축 컬럼 동일 에러 출력 수정 

           
            st.write("")
            row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
                (0.1, 1, 0.1, 1, 0.1)
            )
          
            with row3_1:
                  # 시각화를 할 두 개의 변수 선택
                column2_1 = st.multiselect('변수 두 개를 선택하세요', col_list,key='key1')

                if len(column2_1) == 2:  # 수정: 두 변수가 모두 선택되었을 때만 시각화 진행
                    if column2_1[0] in int_col:
                        # 선택한 두 개의 변수 모두 수치형일 경우 산점도
                        if column2_1[1] in int_col:
                            chart2 = st.selectbox('보고싶은 차트를 선택하세요', ['scatter plot'], key='chart2')
                            # 여기서 산점도 시각화를 구현할 수 있음
                        # 선택한 두 개의 변수가 하나는 수치형, 하나는 범주형일 경우 박스플롯
                        else:
                            chart2 = st.selectbox('보고싶은 차트를 선택하세요', ['box plot'], key='chart2')
                            # 여기서 박스플롯 시각화를 구현할 수 있음
                    elif column2_1[0] in ob_col:
                        # 선택한 두 개의 변수가 하나는 수치형, 하나는 범주형일 경우 박스플롯
                        if column2_1[1] in int_col:
                            chart2 = st.selectbox('보고싶은 차트를 선택하세요', ['box plot'], key='chart2')
                            # 여기서 박스플롯 시각화를 구현할 수 있음
                        # 선택한 두 개의 변수가 모두 범주형일 경우 막대그래프
                        else:
                            chart2 = st.selectbox('보고싶은 차트를 선택하세요', ['count plot'], key='chart2')
                            # 여기서 막대그래프 시각화를 구현할 수 있음


                    if chart2 == 'scatter plot':
                        plot=sns.scatterplot(x = column2_1[0], y = column2_1[1], data=df)
                        st.pyplot(plot.figure)
                    elif chart2 == 'box plot':
                        if column2_1[0] in ob_col:
                            plot = sns.boxplot(x = column2_1[0], y = column2_1[1], data = df)
                            plt.xticks(rotation=45) 
                            st.pyplot(plot.figure)
                        else:
                            plot = sns.boxplot(x = column2_1[1], y = column2_1[0], data = df)
                            st.pyplot(plot.figure)
                    elif chart2 == 'count plot':
                        plot = sns.countplot(x=column2_1[0], hue=column2_1[1], data=df)
                        st.pyplot(plot.figure)

                elif len(column2_1) == 1:
                    st.write('두 개의 컬럼을 선택해주세요.')
                else:
                    st.write('컬럼을 선택하세요.')
########################################################################################################
            with row3_2:
                  # 시각화를 할 두 개의 변수 선택
                column2 = st.multiselect('변수 두 개를 선택하세요', col_list,key='key2')

                if len(column2) == 2:  # 수정: 두 변수가 모두 선택되었을 때만 시각화 진행
                    if column2[0] in int_col:
                        # 선택한 두 개의 변수 모두 수치형일 경우 산점도
                        if column2[1] in int_col:
                            chart3 = st.selectbox('보고싶은 차트를 선택하세요', ['scatter plot'], key='chart3')
                            # 여기서 산점도 시각화를 구현할 수 있음
                        # 선택한 두 개의 변수가 하나는 수치형, 하나는 범주형일 경우 박스플롯
                        else:
                            chart3 = st.selectbox('보고싶은 차트를 선택하세요', ['box plot'], key='chart3')
                            # 여기서 박스플롯 시각화를 구현할 수 있음
                    elif column2[0] in ob_col:
                        # 선택한 두 개의 변수가 하나는 수치형, 하나는 범주형일 경우 박스플롯
                        if column2[1] in int_col:
                            chart3 = st.selectbox('보고싶은 차트를 선택하세요', ['box plot'], key='chart3')
                            # 여기서 박스플롯 시각화를 구현할 수 있음
                        # 선택한 두 개의 변수가 모두 범주형일 경우 막대그래프
                        else:
                            chart3 = st.selectbox('보고싶은 차트를 선택하세요', ['count plot'], key='chart3')
                            # 여기서 막대그래프 시각화를 구현할 수 있음


                    if chart3 == 'scatter plot':
                        plot=sns.scatterplot(x = column2[0], y = column2[1], data=df)
                        st.pyplot(plot.figure)
                    elif chart3 == 'box plot':
                        if column2[0] in ob_col:
                            plot = sns.boxplot(x = column2[0], y = column2[1], data = df)
                            plt.xticks(rotation=45) 
                            st.pyplot(plot.figure)
                        else:
                            plot = sns.boxplot(x = column2[1], y = column2[0], data = df)
                            st.pyplot(plot.figure)
                    elif chart3 == 'count plot':
                        plot = sns.countplot(x=column2[0], hue=column2[1], data=df)
                        st.pyplot(plot.figure)

                elif len(column2) == 1:
                    st.write('두 개의 컬럼을 선택해주세요.')
                else:
                    st.write('컬럼을 선택하세요.')
            
#################################################
    elif page == '데이터 변환':
        st.sidebar.write("""
    'Data Transformation'
        """)
        st.title('데이터 변환')
#################################################
        options = st.selectbox(
                    '기능 선택',
                    ['*Choose an option','결측치','이상치','스케일링']
                 )
        
        # 컬럼별로 결측치의 개수(count)와 비율(percent) 확인
        missing_df=st.session_state.df.isnull().sum().reset_index()
        missing_df.columns=['column','count']
        missing_df['percent']=round((missing_df['count']/st.session_state.df.shape[0])*100,2)
        missing_df=missing_df.loc[missing_df['percent']!=0].sort_values('percent',ascending=False).reset_index(drop=True)
        
        missing_50per=missing_df.loc[missing_df['percent']>=50].sort_values('percent',ascending=False).reset_index(drop=True)
        missing_50per['percent']=missing_50per['percent'].astype(str) + '%'
        
        missing_1050per=missing_df.loc[(missing_df['percent']>=10)&(missing_df['percent']<50)].sort_values('percent',ascending=False).reset_index(drop=True)
        missing_1050per['percent']=missing_1050per['percent'].astype(str) + '%'
        
        missing_10per=missing_df.loc[missing_df['percent']<10].sort_values('percent',ascending=False).reset_index(drop=True)
        missing_10per['percent']=missing_10per['percent'].astype(str) + '%'
        
        # Missing Value #
        if options == '결측치':
            tab1,tab2=st.tabs(['결측치 확인','결측치 처리'])
            with tab1:
                if len(st.session_state.df.columns[st.session_state.df.isnull().any()])==0:
                    st.subheader('결측치 없음')
                else:
                    if st.checkbox('10% 미만'):
                        st.write(missing_10per)
                    if st.checkbox('10%~50%'):
                        st.write(missing_1050per)
                    if st.checkbox('50% 이상'):
                        st.write(missing_50per)
            st.write('')
            st.write('')   
            
            with tab2:
                if len(st.session_state.df.columns[st.session_state.df.isnull().any()])==0:
                    st.subheader('결측치 없음')
                else:
                    st.subheader('-가이드라인-')
                    st.write('- 10% 미만 : 삭제 or 대치')
                    st.write('- 10% ~ 50% : 회귀모델 추정값 대치')
                    st.write('- 50% 이상 : 해당 컬럼(변수) 자체 제거')
                
                    missing_col=[]
                    for i in range(len(missing_df['column'].values)):
                        missing_col.append(missing_df['column'].values[i])
                
                    tab1,tab2,tab3,tab4= st.tabs(['결측치 삭제(행 단위)','결측치 삭제(열 단위)','결측치 대치','분류/회귀모델 추정치 대치'])
                    ## Dropna ##
                    with tab1:
                        select_col = st.multiselect(
                            '삭제할 컬럼 선택(행 단위)',
                            missing_col,
                            key='dropna'
                        )
                        if len(st.session_state.df.columns[st.session_state.df.isnull().any()])==0:
                            st.subheader('결측치 없음')
                        else:
                            st.write(st.session_state.df.dropna(axis=0, subset=select_col))
                            st.write(st.session_state.df.dropna(axis=0, subset=select_col).shape)
                        #st.write(st.session_state.df.dropna(axis=0, subset=select_col))
                        #st.write(st.session_state.df.dropna(axis=0, subset=select_col).shape)
                        if st.button('결측치 삭제하기(행 단위)'):
                            st.session_state.df.dropna(axis=0, how='any', subset=select_col, inplace=True)
                    ## Drop Columns ##
                    with tab2:
                        col_list=[]
                        for i in range(len(st.session_state.df.columns)):
                            col_list.append(st.session_state.df.columns[i])
                        select_col = st.multiselect(
                                '삭제할 컬럼 선택(열 단위)',
                                missing_col,
                                key='drop column'
                            )
                        st.write(st.session_state.df.drop(columns=select_col))
                        st.write(st.session_state.df.drop(columns=select_col).shape)
                    
                        if st.button('결측치 삭제하기(열 단위)'):
                            st.session_state.df.drop(columns=select_col,inplace=True)
                    ## Fillna ##
                    with tab3:
                        select_col = st.selectbox(
                            '결측치 대치할 컬럼 선택',
                            missing_col,
                            key='fillna'
                        )
                        mean = round(st.session_state.df[select_col].mean(),2)
                        median = round(st.session_state.df[select_col].median(),2)
                        
                        col1,col2 = st.columns([1,1])
                        with col1:
                            st.write(f'{select_col} 컬럼의 평균값 : ', mean)
                            st.write(st.session_state.df[select_col].fillna(mean))
                            if st.button('평균값 대치하기', key='mean'):
                                st.session_state.df[select_col].fillna(mean, inplace=True)
                        with col2:
                            st.write(f'{select_col} 컬럼의 중위값 : ', median)
                            st.write(st.session_state.df[select_col].fillna(median))
                            if st.button('중위값 대치하기', key='median'):
                                st.session_state.df[select_col].fillna(median, inplace=True)
                    ## Regression ##
                    with tab4:
                        select_col = st.selectbox(
                            '1. 종속변수 선택', 
                            missing_col,
                            key='target column'
                        )
                        select_model_type = st.selectbox(
                            '2. 분류 or 회귀 선택',
                            ['','분류','회귀'],
                            key='model type'
                        )
                        select_model = st.selectbox(
                            '3. 모델 선택',
                            ['','RandomForest','DecisionTree'],
                            key='select model' 
                        )
                        if select_model_type=='분류':
                            if select_model=='RandomForest':
                                st.write('')
                                # x_train, y_train 정의
                                y_train=st.session_state.df.dropna()[select_col]
                                x_train=st.session_state.df.dropna().drop(columns=select_col)
                                #st.write(x_train, x_train.shape)
                                #st.write(y_train, y_train.shape)
                                
                                # RandomForest
                                from sklearn.ensemble import RandomForestClassifier
                                RFC=RandomForestClassifier()
                                RFC.fit(x_train, y_train)
                                
                                # x_test, y_test 정의
                                y_test=st.session_state.df.loc[st.session_state.df[select_col].isnull()][select_col]
                                x_test=st.session_state.df.loc[st.session_state.df[select_col].isnull()].drop(columns=select_col)
                                #st.write(x_test, x_test.shape)
                                #st.write(y_test, y_test.shape)
                                
                                y_test_predict=RFC.predict(x_test)
                                st.write('*해당 종속변수 RandomForest 분류 추정치', y_test_predict, y_test_predict.shape)
                                
                                if st.button('추정치 대치하기'):
                                    st.session_state.df.loc[st.session_state.df[select_col].isnull(), select_col]=y_test_predict
                                
                            elif select_model=='DecisionTree':
                                st.write('')
                                # x_train, y_train 정의
                                y_train=st.session_state.df.dropna()[select_col]
                                x_train=st.session_state.df.dropna().drop(columns=select_col)
                                #st.write(x_train, x_train.shape)
                                #st.write(y_train, y_train.shape)
                                
                                # DecisionTree
                                from sklearn.tree import DecisionTreeClassifier
                                DTC=DecisionTreeClassifier()
                                DTC.fit(x_train, y_train)
                                # x_test, y_test 정의
                                y_test=st.session_state.df.loc[st.session_state.df[select_col].isnull()][select_col]
                                x_test=st.session_state.df.loc[st.session_state.df[select_col].isnull()].drop(columns=select_col)
                                #st.write(x_test, x_test.shape)
                                #st.write(y_test, y_test.shape)
                                
                                y_test_predict=DTC.predict(x_test)
                                st.write('*해당 종속변수 DecisionTree 분류 추정치', y_test_predict, y_test_predict.shape)
                                
                                if st.button('추정치 대치하기'):
                                    st.session_state.df.loc[st.session_state.df[select_col].isnull(), select_col]=y_test_predict
                        elif select_model_type=='회귀':
                            if select_model=='RandomForest':
                                st.write('')
                                # x_train, y_train 정의
                                y_train=st.session_state.df.dropna()[select_col]
                                x_train=st.session_state.df.dropna().drop(columns=select_col)
                                #st.write(x_train, x_train.shape)
                                #st.write(y_train, y_train.shape)
                                
                                # RandomForest
                                from sklearn.ensemble import RandomForestRegressor
                                RFR=RandomForestRegressor()
                                RFR.fit(x_train, y_train)
                                
                                # x_test, y_test 정의
                                y_test=st.session_state.df.loc[st.session_state.df[select_col].isnull()][select_col]
                                x_test=st.session_state.df.loc[st.session_state.df[select_col].isnull()].drop(columns=select_col)
                                #st.write(x_test, x_test.shape)
                                #st.write(y_test, y_test.shape)
                                
                                y_test_predict=RFR.predict(x_test)
                                st.write('*해당 종속변수 RandomForest 회귀 추정치', y_test_predict, y_test_predict.shape)
                                
                                if st.button('추정치 대치하기'):
                                    st.session_state.df.loc[st.session_state.df[select_col].isnull(), select_col]=y_test_predict
                                
                            elif select_model=='DecisionTree':
                                st.write('')
                                # x_train, y_train 정의
                                y_train=st.session_state.df.dropna()[select_col]
                                x_train=st.session_state.df.dropna().drop(columns=select_col)
                                #st.write(x_train, x_train.shape)
                                #st.write(y_train, y_train.shape)
                                
                                # DecisionTree
                                from sklearn.tree import DecisionTreeRegressor
                                DTR=DecisionTreeRegressor()
                                DTR.fit(x_train, y_train)
                                # x_test, y_test 정의
                                y_test=st.session_state.df.loc[st.session_state.df[select_col].isnull()][select_col]
                                x_test=st.session_state.df.loc[st.session_state.df[select_col].isnull()].drop(columns=select_col)
                                #st.write(x_test, x_test.shape)
                                #st.write(y_test, y_test.shape)
                                
                                y_test_predict=DTR.predict(x_test)
                                st.write('*해당 종속변수 DecisionTree 회귀 추정치', y_test_predict, y_test_predict.shape)
                                
                                if st.button('추정치 대치하기'):
                                    st.session_state.df.loc[st.session_state.df[select_col].isnull(), select_col]=y_test_predict
                            
                        
        if options == '이상치':
            tab1,tab2=st.tabs(['이상치 확인','이상치 처리'])
            with tab1:
                st.write('이상치 확인')
            with tab2:
                st.write('이상치 처리')
    
        # Scaling #
        if options == '스케일링':
            tab1,tab2=st.tabs(['MinMax','Standard'])
            with tab1:
                st.write('MinMax')
            with tab2:
                st.write('Standard')
    

#################################################
    else:
        st.sidebar.write("""
            error?
        """)
        st.title("Error")
#################################################


if __name__ == '__main__':
    st.experimental_set_query_params(page='데이터 가져오기')
    url_params = st.experimental_get_query_params()
    st.session_state.page = PAGES.index(url_params['page'][0])
    
    
    #st.session_state['data_type'] = 'County Level'
    #st.session_state['data_format'] = 'Raw Values'
    #st.session_state['loaded'] = False
    run_UI()