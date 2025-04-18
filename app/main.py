import sys
import streamlit as st
import yfinance as yf
import pandas as pd
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import gaussian_kde
from bcb import sgs
import numpy as np
import datetime
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import time
import altair as alt
import locale
from PIL import Image
# import ia_page
locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")

st.set_page_config(
    page_title="Simulador de Risco de Ativos",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

with st.container():
    logo = Image.open("logo_riskpilot.png")

    # Crie duas colunas
    col1, col2 = st.columns([1, 12])

    with col1:
        st.image(logo, width=600) 

    with col2:
        st.title("RiskPilot")
        st.markdown("##### Gerenciamento de Riscos de FIIS e Ações")


@st.cache_data
def obter_tickers_acoes():
    url = "https://www.fundamentus.com.br/resultado.php"
    headers = {
        "User-Agent": 
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    try: 
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"id": "resultado"})
            if table: 
                tickers = []
                for tr in table.find_all("tr"):
                    td = tr.find("td", ckass="res_papel")
                    if td: 
                        span = td.find("span", class_="tips")
                        if span:
                            a = span.find("a", href=True)
                            if a: 
                                ticker = a.get_text(strip=True)
                                if ticker: 
                                    tickers.append(ticker) 
                return tickers
        else:
            st.warning("Tabela de ações não encontrada no site.")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Erro na requisição: {e}")
        return []
    except Exception as e:
        st.error(f"Ocorreu um erro durante o scraping: {e}") 
        return []

@st.cache_data
def obter_tickers_fiis():
    url = "https://www.fundamentus.com.br/fii_resultado.php"
    headers = {
        "User-Agent": 
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"id": "tabelaResultado"})
            if table: 
                tickers = []
                for tr in table.find_all("tr"):
                    td = tr.find("td", ckass="res_papel")
                    if td: 
                        span = td.find("span", class_="tips")
                        if span:
                            a = span.find("a", href=True)
                            if a: 
                                ticker = a.get_text(strip=True)
                                if ticker: 
                                    tickers.append(ticker) 
                return tickers
        else:
            st.warning("Tabela de FIIs não encontrada no site.")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Erro na requisição: {e}")
        return []
    except Exception as e:
        st.error(f"Ocorreu um erro durante o scraping: {e}")
        return []


@st.cache_data
@st.cache_data
def get_dados_bcb(codigo, max_retries=3, retry_delay=5):
    from datetime import datetime, timedelta
    data_fim = datetime.today().strftime("%d/%m/%Y")
    data_inicio = (datetime.today() - timedelta(days=10*365)).strftime("%d/%m/%Y")
    
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=json&dataInicial={data_inicio}&dataFinal={data_fim}"
    
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_json(response.text)
            df.set_index('data', inplace=True)
            df.index = pd.to_datetime(df.index, dayfirst=True)
            df.columns = ["SELIC"]
            df["SELIC"] = df["SELIC"] / 100
            return df
        except requests.exceptions.SSLError as e:
            st.error(f"Erro SSL ao obter dados da taxa Selic do BCB (tentativa {retries + 1}): {e}")
            retries += 1
            time.sleep(retry_delay)
        except requests.exceptions.RequestException as e:
            # st.error(f"Erro na requisição (tentativa {retries + 1}): {e}")
            retries += 1
            time.sleep(retry_delay)
        except Exception as e:
            st.error(f"Erro durante o processamento (tentativa {retries + 1}): {e}")
            retries += 1
            time.sleep(retry_delay)
    
    st.error("Falha ao obter dados da taxa Selic do BCB após várias tentativas.")
    return None

@st.cache_data
def get_dados_raw_yfinance(ativos_validos, period):

    dados = yf.download(
        ativos_validos, 
        period=period,
        auto_adjust=False,
        multi_level_index=False,
        threads=True
    )

    if dados.empty:
        print("Não há dados disponíveis para os ativos especificados.")
        sys.exit(1)

    return dados


# @st.cache_data
def get_verifica_ativos_validos(tipo_carteira: str, period):

    ativos_validos = []
    ativos_invalidos = []

    for ativo in st.session_state[tipo_carteira]:
        ticker = f"{ativo}.SA"
        dados = yf.download(ticker, period=period, auto_adjust=False, threads=True)
        if dados.empty:
            ativos_invalidos.append(ativo)
        else:
            ativos_validos.append(ticker)

    if ativos_invalidos:
        st.warning(f"Não foi possivel recuperar o histórico do(s) Ativo(s): {ativos_invalidos}")
    
    return ativos_validos


def get_dados_ativos_diarios(ativos_validos, period, coluna="Adj Close"):
    """
    Retorna um DataFrame contendo os retornos diários 
    dos ativos para o período especificado pelo usuário.
    """
    dados = get_dados_raw_yfinance(ativos_validos=ativos_validos, period=period)
    dados_diarios = dados[coluna].pct_change().dropna()

    if len(ativos_validos) == 1:
        dados_diarios = dados_diarios.rename(ativos_validos[0]).to_frame()
        dados_diarios.columns = [ativos_validos[0]] 

    return dados_diarios


def get_dataset_volatilidade_desvio_padrao_carteira(dataset, period=252):

    # Calcular a volatilidade (desvio padrão) de cada ativo
    dataset_volatilidade = dataset.std() * np.sqrt(period)  # Desvio padrão anualizado
    dataset_volatilidade = pd.DataFrame(dataset_volatilidade, columns=["Risco"])
    dataset_volatilidade = dataset_volatilidade.sort_values(by="Risco", ascending=False)

    dataset_volatilidade = dataset_volatilidade.reset_index().rename(columns={'index': 'Ticker'})


    return dataset_volatilidade


def get_volatilidade_desvio_padrao_carteira(dataset, perc, period=252):
    """
    Calcula a volatilidade total da carteira.
    """
    percentuais_aloc_normalizados = np.array(perc) / 100
    if len(dataset.shape) == 1:
        volatilidade_carteira = dataset.std() * np.sqrt(period)
    else:
        cov_matrix = dataset.cov() * period
        volatilidade_carteira = np.sqrt(np.dot(percentuais_aloc_normalizados.T, 
                np.dot(cov_matrix, percentuais_aloc_normalizados))
            )
    return volatilidade_carteira


def get_volatilidade_negativa_carteira(dataset, perc):
    """
    Calcula a volatilidade negativa anualizada da carteira.
    """
    percentuais_aloc_normalizados = np.array(perc) / 100
    retorno_carteira_diario = (dataset * percentuais_aloc_normalizados).sum(axis=1)
    retornos_negativos = retorno_carteira_diario[retorno_carteira_diario < 0]
    if retornos_negativos.empty:
        return 0  
    volatilidade_negativa = retornos_negativos.std() * np.sqrt(252)
    return volatilidade_negativa


def get_perc_contribuicao_ativos(dataset, perc, period=252):

    volatilidade_carteira = get_volatilidade_desvio_padrao_carteira(dataset, perc, period)
    percentuais_aloc_normalizados = np.array(perc) / 100

    if len(dataset.shape) > 1:
        cov_matrix = dataset.cov() * period 
        volatilidade_ativo = percentuais_aloc_normalizados * np.sqrt(np.diag(cov_matrix))
        contribuicao_marginal_ativo = (volatilidade_ativo / volatilidade_carteira)
        contribuicao_percentual = contribuicao_marginal_ativo / contribuicao_marginal_ativo.sum() * 100
        colunas = dataset.columns
    else: 
        contribuicao_percentual = 100
        colunas = [dataset.name]

    dataset = {"Ativo": colunas, "Contribuição (%)": contribuicao_percentual}
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sort_values(by="Contribuição (%)", ascending=False)
    dataset["Contribuição (%)"] = dataset["Contribuição (%)"].round(1)
    
    dataset["Cor"] = dataset["Contribuição (%)"].apply(
        _customize_color_perc_contribuicao
    )

    return dataset


def plot_pie_chart(df, value_column, category_column, color_scheme):

    chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(f"{value_column}:Q", stack=True),
        color=alt.Color(f"{category_column}:N", scale=alt.Scale(scheme=color_scheme), legend=alt.Legend(
            orient="right", title="Ativos", titleFontSize=16, labelFontSize=14)
        ),
        tooltip=[f"{category_column}:N", f"{value_column}:Q"]
    ).properties(
        width=350,
        height=350
    )
    st.altair_chart(chart, use_container_width=True)


def plot_horizontal_bar_chat(df, x_axis, y_axis, color):

    barras = alt.Chart(df).mark_bar().encode(
        x=alt.X(x_axis+":Q", title=x_axis),
        y=alt.Y(y_axis+":N", title=None),
        color=alt.Color(color+":N", scale=None),
        tooltip=[y_axis, x_axis]
    )

    # Adicionar labels internos nas barras
    labels = barras.mark_text(
        align="left",
        baseline="middle",
        dx=1,
        fontSize=15
    ).encode(
        text=alt.Text(x_axis+":Q", format=".1f")
    )
    # Combinar gráfico de barras e labels
    grafico_final = alt.layer(barras, labels).configure_axis(
        grid=False 
    ).properties(
        width=350,
        height=150
    )

    st.altair_chart(grafico_final, use_container_width=True)


def plot_risk_return(dataset_diarios_cr, percentuais_cr, dataset_diarios_ct, percentuais_ct, sharpe_cr, sharpe_ct, index_name):
    data = []
    tuple_creal = ("Real", dataset_diarios_cr, percentuais_cr, sharpe_cr)
    tuple_cteorica = ("Teórica", dataset_diarios_ct, percentuais_ct, sharpe_ct)

    for tipo, dataset, percentuais, risk_index in [tuple_creal, tuple_cteorica]:
        if not dataset.empty:
            # retorno_carteira = get_retorno_carteira(dataset, percentuais)
            retornos_diarios = get_retorno_carteira_diario(
                dataset, percentuais
            )
            retorno_anualizado = get_retorno_carteira_anualizado(
                retornos_diarios, period=252
            )
            volatilidade_carteira = get_volatilidade_desvio_padrao_carteira(
                dataset, percentuais
            )
            data.append({
                "Carteira": tipo, 
                "Volatilidade": volatilidade_carteira,
                "Retorno": retorno_anualizado*100, 
                index_name: risk_index}
            )

    df = pd.DataFrame(data)

    melhor_carteira = df.loc[df[index_name].idxmax()]
    
    df["Destaque"] = [
        "Sim" if row["Carteira"] == melhor_carteira["Carteira"] else "Não"
        for index, row in df.iterrows()]

    chart = alt.Chart(df).mark_circle(size=1800).encode(
        x=alt.X("Volatilidade:Q", title="Volatilidade (Risco)"),
        y=alt.Y("Retorno:Q", title="Retorno Médio", scale=alt.Scale(nice=3)),
        color=alt.condition(
            alt.datum["Destaque"] == "Sim", alt.value("#4C6444"), alt.value("#CDBCAB")
        ),
        tooltip=[
            alt.Tooltip("Carteira:N", title="Carteira"),
            alt.Tooltip("Volatilidade:Q", title="Volatilidade", format=".2f"),
            alt.Tooltip("Retorno:Q", title="Retorno Médio (%)", format=".2f"),
            alt.Tooltip(f"{index_name}:Q", title=f"Índice de {index_name}", format=".2f")
        ]
    ).properties(
            width=500,
            height=350,
            title=alt.TitleParams(
            text=f"Indice {index_name} - Risco vs. Retorno das Carteiras (Anualizado)",
            subtitle=f"""Melhor Custo-Benefício: Carteira {melhor_carteira["Carteira"]} ({index_name}: {melhor_carteira[index_name]:.2f})"""
        )
    ).interactive()

    annotation = alt.Chart(pd.DataFrame([melhor_carteira])).mark_text(
        align="center",
        baseline="middle",
        dx=0
    ).encode(
        x="Volatilidade:Q",
        y="Retorno:Q",
        text="Carteira:N"
    )

    st.altair_chart(chart + annotation, use_container_width=True)


def plot_line_chart(dataset, x_axis, y_axis,  color):

    chart_performance = alt.Chart(dataset).mark_line().encode(
        x=f"{x_axis}:T",
        y=f"{y_axis}:Q",
        color="Ativo:N",
        tooltip=[f"{x_axis}:T", f"{color}:N", 
                    alt.Tooltip("Retorno Acumulado:Q",
                                format=".5f")]
    )

    st.altair_chart(chart_performance, use_container_width=True)


def plot_volatilidade_barras_horizontal(dataset):
    """
    Gera um gráfico de barras horizontal da volatilidade dos ativos.
    """
    dataset['Risco'] = dataset['Risco'].round(4)
    
    chart = alt.Chart(dataset).mark_bar(color='#AF4F45', size=20).encode(
        y=alt.Y('Ticker:N', title='', sort='-x'),
        x=alt.X('Risco:Q', title='Risco baseado no desvio padrão anualizado', axis=alt.Axis(format='%')),
        tooltip=['Ticker:N', alt.Tooltip('Risco:Q', format='.2%')]
    )

    chart_customize_text = alt.Chart(dataset).mark_text(
        align='right',
        baseline='middle',
        dx=-5,
        color='white',
        fontSize=14
    ).encode(
        y='Ticker:N',
        x='Risco:Q',
        text=alt.Text('Risco:Q', format='.2%') #Formata o texto do gráfico como porcentagem
    )

    st.altair_chart(chart + chart_customize_text, use_container_width=True)


def plot_tabela_progress_column(indices_real, indices_teorica, nomes_indices):
    """
    Exibe uma tabela com barras de progresso para comparar os índices de risco.
    """
    data = {"Carteira Real": indices_real, "Carteira Teórica": indices_teorica}
    df = pd.DataFrame(data, index=nomes_indices).T

    column_config = {}
    for index in nomes_indices:
        # Informações detalhadas para cada índice
        if index == "Indice Sharpe":  # Ajustado para corresponder ao valor exato
            help_text = "O Indice de Sharpe mede o retorno ajustado ao risco de um investimento."
        elif index == "Indice Sortino":  # Ajustado para corresponder ao valor exato
            help_text = "O Indice de Sortino mede o retorno ajustado ao risco de um investimento, focando no risco de queda."
        else:
            help_text = f"Indice de {index} que mede o retorno ajustado ao risco."

        column_config[index] = st.column_config.ProgressColumn(
            label=index,
            min_value=df[index].min(),
            max_value=df[index].max(),
            format="%.2f",
            width="medium",
            help=help_text
        )

    st.dataframe(df, column_config=column_config)


def plot_var_lollipop_chart(var_dict, title, legend_position="top"):
    """
    Plota um gráfico de lollipop customizado comparando os valores do VaR.
    """

    data = pd.DataFrame({
        "Carteira": list(var_dict.keys()),
        "VaR": list(var_dict.values())
    })

    # Segmento de reta customizado
    segmento_reta = alt.Chart(data).mark_rule(strokeWidth=2, color="#CCCCCC").encode(
        x="Carteira:N",
        y="VaR:Q"
    )

    # Círculos customizados
    circulos = alt.Chart(data).mark_circle(size=200, color="#e45756").encode(
        x="Carteira:N",
        y="VaR:Q",
        color=alt.Color("Carteira:N", scale=alt.Scale(domain=["Carteira Real", "Carteira Teórica"], range=["#40E0D0", "#9400D3"])),
        tooltip=["Carteira:N", alt.Tooltip("VaR:Q", format=".2%")]
    )

    chart = (segmento_reta + circulos).properties(
        title = title,
        width=600,
        height=400
    ).configure_axisX(
        labelFontSize=12 
    ).configure_legend(
        orient=legend_position,
        direction="horizontal",
        title=None
    )
    st.altair_chart(chart, use_container_width=True)


def plot_area_drawdown_chart(retornos, titulo, limiar_drawdown):
    """
    Cria um gráfico de área de drawdown com Altair, destacando períodos de maior perda.
    """

    # Obtém o drawdown e os períodos críticos
    try:
        mdd, drawdowns, drawdown_periods = get_value_mdd_drawdowns_picos(retornos)
        
        # **Nova Verificação:** Evita erro se drawdowns for inválido
        if drawdowns is None or not isinstance(drawdowns, pd.Series) or drawdowns.empty:
            st.warning(f"Não há dados suficientes para calcular o drawdown de {titulo}.")
            return
        
    except Exception as e:
        st.error(f"Erro ao calcular o drawdown de {titulo}: {e}")
        return


    df = pd.DataFrame({'Data': drawdowns.index, 'Drawdown': drawdowns.values})

    # Se o usuário escolher um limiar maior que o drawdown máximo, ajustamos
    if limiar_drawdown > abs(mdd):
        st.warning(f"Limiar selecionado ({limiar_drawdown:.0%}) é maior que o máximo drawdown ({mdd:.2%}). Ajustando automaticamente.")
        limiar_drawdown = abs(mdd)

    area_chart = alt.Chart(df).mark_area(
        strokeWidth=0,
        fill='#cccccc',  
        fillOpacity=0.6,
        stroke='black'
    ).encode(
        x=alt.X('Data:T', title='Data'),
        y=alt.Y('Drawdown:Q', title='Drawdown (%)', axis=alt.Axis(format='%'))
    ).properties(
        title={"text": titulo, 
               "subtitle": f"Avaliando drawdowns superiores a {limiar_drawdown:.0%}"}
    )

    mdd_line = alt.Chart(pd.DataFrame({'y': [mdd]})).mark_rule(color='#d62728', strokeDash=[5,3]).encode(
        y='y:Q',
        tooltip=[alt.Tooltip('y:Q', format='.2%', title='Máximo Drawdown')]
    )

    align_data = df['Data'].min() + (df['Data'].max() - df['Data'].min()) / 2
    mdd_text = alt.Chart(pd.DataFrame({'x': [align_data], 'y': [mdd], 'text': [f'Máx. Drawdown: {mdd:.2%}']})).mark_text(
        align='center',
        baseline='bottom',
        dy=15  
    ).encode(
        x='x:T',
        y='y:Q',
        text='text:N'
    )

    severe_drawdowns = df[df["Drawdown"] < -limiar_drawdown]

    if not severe_drawdowns.empty:
        severe_periods = alt.Chart(severe_drawdowns).mark_rule(
            opacity=0.1,
            color='#f85454'
        ).encode(
            x='Data:T'
        )
    else:
        severe_periods = alt.Chart(pd.DataFrame({'Data':[], 'Drawdown':[]})).mark_rule()

    tooltips = alt.Chart(df).mark_line(color="black").encode(
        x='Data:T',
        y='Drawdown:Q',
        tooltip=['Data:T', alt.Tooltip('Drawdown:Q', format='.2%')]
    )

    chart = (area_chart + severe_periods + mdd_line + mdd_text + tooltips).properties(
        width=600,
        height=300
    )

    st.altair_chart(chart, use_container_width=True)


def plot_barras_metricas_marcadores(resultados_cr, resultados_ct):
    """
    Cria um gráfico de barras com marcadores de valor comparando as métricas de risco das carteiras.
    """

    metricas = ['CVaR', 'Sharpe', 'VaR', 'Volatilidade']
    dados = []
    for metrica in metricas:
        valor_cr = resultados_cr[metrica].iloc[-1]
        valor_ct = resultados_ct[metrica].iloc[-1]

        if metrica == 'Sharpe':
            cor_cr = '#3F6060' if valor_cr > valor_ct else '#DDDDDD'
            cor_ct = '#3F6060' if valor_ct > valor_cr else '#DDDDDD'
        elif metrica in ['VaR', 'CVaR']:
            cor_cr = '#3F6060' if valor_cr > valor_ct else '#DDDDDD'
            cor_ct = '#3F6060' if valor_ct > valor_cr else '#DDDDDD'
        else:
            cor_cr = '#3F6060' if valor_cr < valor_ct else '#DDDDDD'
            cor_ct = '#3F6060' if valor_ct < valor_cr else '#DDDDDD'

        dados.append({'Métrica': metrica, 'Carteira': 'Real', 'Valor': valor_cr, 'Cor': cor_cr})
        dados.append({'Métrica': metrica, 'Carteira': 'Teórica', 'Valor': valor_ct, 'Cor': cor_ct})
    
    df = pd.DataFrame(dados)

    barras = alt.Chart(df).mark_bar().encode(
        x='Carteira:N',
        y='Valor:Q',
        color=alt.Color('Cor:N', scale=None),
        tooltip=['Métrica:N', 'Valor:Q', 'Carteira:N']
    ).properties(width=250)

    marcadores = alt.Chart(df).mark_text(align='center', baseline='bottom', dy=-5).encode(
        x='Carteira:N',
        y='Valor:Q',
        text=alt.Text('Valor:Q', format='.3f')
    )

    chart = (barras + marcadores).facet(
        column=alt.Column('Métrica:N', title=None)
    ).properties(
    title={
        "text": "Resultados do Backtesting: Comparação de Carteiras",
        "subtitle": "Comparação das carteiras após backtesting. Barras verdes destacam a carteira com melhor resultado."
    })

    st.altair_chart(chart, use_container_width=True)


def plot_retornos_acumulados(retorno_carteira_diario_cr, retorno_carteira_diario_ct):
    """
    Gera e exibe um gráfico de área dos retornos acumulados das carteiras.
    """

    retornos_acumulados = pd.DataFrame({
        'Data': retorno_carteira_diario_cr.index,
        'Real': (1 + retorno_carteira_diario_cr).cumprod() - 1,
        'Teórica': (1 + retorno_carteira_diario_ct).cumprod() - 1
    })

    retornos_acumulados = retornos_acumulados.melt('Data', var_name='Carteira', value_name='Retorno Acumulado')

    cores = {'Real': '#3f6060', 'Teórica': '#9f7f52'}
    selection = alt.selection_point(fields=['Carteira'], bind='legend') 

    chart_acumulados = alt.Chart(retornos_acumulados).mark_area().encode(
        x='Data:T',
        y=alt.Y('Retorno Acumulado:Q', axis=alt.Axis(format='%'), stack=None), 
        # color=alt.Color('Carteira:N', scale=alt.Scale(domain=list(cores.keys()), range=list(cores.values()))),
        color=alt.Color('Carteira:N', scale=alt.Scale(domain=['Real', 'Teórica'], range=list(cores.values()))) , 
        tooltip=[alt.Tooltip('Data:T', title='Data'), alt.Tooltip('Retorno Acumulado:Q', title='Retorno Acumulado', format='.2%')], 
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
    ).properties(
        width=800,
        height=400
    ).add_params(
        selection
    )

    st.altair_chart(chart_acumulados, use_container_width=True)


def plot_matriz_correlacao(dataset_diarios, tipo_carteira, color_scheme):
    """
    Plota a matriz de correlação dos ativos usando Altair Chart.

    Args:
        dataset_diarios (pd.DataFrame): DataFrame com os retornos diários dos ativos.
        tipo_carteira (str): Tipo da carteira ("Carteira Real" ou "Carteira Teórica").
    """
    # Garante que a coluna "Ticker" não existe
    if 'Ticker' in dataset_diarios.columns:
        dataset_diarios = dataset_diarios.drop(columns=['Ticker'])

    # Calcular a matriz de correlação
    matriz_correlacao = dataset_diarios.corr(method='pearson')

    # Remove o nome do índice antes de renomear
    matriz_correlacao.index.name = None

    # Converter a matriz de correlação em um DataFrame longo
    matriz_correlacao_long = matriz_correlacao.stack().reset_index()
    matriz_correlacao_long.columns = ['Ativo 1', 'Ativo 2', 'Correlação']

    # Criar o mapa de calor com Altair
    heatmap = alt.Chart(matriz_correlacao_long).mark_rect().encode(
        x=alt.X('Ativo 1:O', title=''),
        y=alt.Y('Ativo 2:O', title=''),
        color=alt.Color('Correlação:Q', scale=alt.Scale(scheme=color_scheme, domain=(-1, 1), reverse=True), title='Nível de Correlação'),
        tooltip=[alt.Tooltip('Ativo 1:O', title='Ativo 1'),
                alt.Tooltip('Ativo 2:O', title='Ativo 2'),
                alt.Tooltip('Correlação:Q', format='.2f')]
    ).properties(
        title=alt.Title(
            text=f'Matriz de Correlação dos Ativos ({tipo_carteira})',
            subtitle="Limiares de correlação: Forte (>0.7), Moderada (0.3-0.7), Fraca (<0.3)"
        ),
        width=600,
        height=400
    )

    # Adicionar rótulos nas barras
    text = heatmap.mark_text(baseline='middle').encode(
        text=alt.Text('Correlação:Q', format='.2f'),
        color=alt.condition(
            alt.datum.Correlação > 0.5,
            alt.value('black'),
            alt.value('white')
        )
    )

    chart = (heatmap + text).interactive().configure_view(
        strokeWidth=0
    ).configure_text(
        fontSize=12
    ).configure_concat(
    spacing=0
    )

    st.altair_chart(chart, use_container_width=True)


def plot_drawdown_stress(mdd_real, mdd_teorico, mdd_real_stress, mdd_teorico_stress):
    """
    Gera e exibe o gráfico de comparação de drawdowns em cenário de stress.
    """

    dados_drawdown_stress = pd.DataFrame({
        "Carteira": ["Real", "Teórica"],
        "Drawdown (Normal)": [mdd_real, mdd_teorico],
        "Drawdown (Stress)": [mdd_real_stress, mdd_teorico_stress],
    })

    grafico_drawdown_stress = alt.Chart(dados_drawdown_stress).transform_fold(
        ["Drawdown (Normal)", "Drawdown (Stress)"], as_=["Cenário", "Drawdown"]
    ).mark_bar().encode(
        x="Carteira:N",
        y="Drawdown:Q",
        color=alt.Color("Cenário:N", scale=alt.Scale(range=["#dddddd", "#a20f0f"])),
        tooltip=["Carteira:N", "Cenário:N", "Drawdown:Q"],
    )

    st.altair_chart(grafico_drawdown_stress, use_container_width=True)


def plot_cauda_distribuicao(retornos, var, cvar, titulo="Análise da Cauda da Distribuição"):
    """
    Plota um histograma elegante destacando a cauda da distribuição, VaR e CVaR.
    """

    histograma = alt.Chart(pd.DataFrame({"Retornos": retornos})).mark_bar(
        opacity=0.7, color="#4c78a8", stroke="white", strokeWidth=0.5
    ).encode(
        x=alt.X("Retornos:Q", bin=alt.Bin(maxbins=30), title="Retornos Diários (%)"),
        y=alt.Y("count()", title="Frequência"),
        tooltip=[alt.Tooltip("Retornos:Q", format=".2%"), "count()"]
    ).properties(
        title=titulo
    )

    cauda = alt.Chart(pd.DataFrame({"Retornos": retornos[retornos <= var]})).mark_bar(
        color="#d62728", opacity=0.5
    ).encode(
        x=alt.X("Retornos:Q", bin=alt.Bin(maxbins=10)),
        y="count()"
    )

    linha_var = alt.Chart(
        pd.DataFrame({"VaR": [var]})
    ).mark_rule(color="#40E0D0").encode(x="VaR:Q").properties(title=f"VaR: {var:.2%} (95%)")
    linha_cvar = alt.Chart(
        pd.DataFrame({"CVaR": [cvar]})
    ).mark_rule(color="#9400D3").encode(x="CVaR:Q").properties(title=f"CVaR: {cvar:.2%} (95%)")

    anotacao_var = alt.Chart(pd.DataFrame({"VaR": [var]})).mark_text(
        text=f"VaR: {var:.2%}", dx=-10, dy=-10, align="right"
    ).encode(x="VaR:Q")
    anotacao_cvar = alt.Chart(pd.DataFrame({"CVaR": [cvar]})).mark_text(
        text=f"CVaR: {cvar:.2%}", dx=-10, dy=-10, align="right"
    ).encode(x="CVaR:Q")

    chart = (histograma + cauda + linha_var + linha_cvar + anotacao_var + anotacao_cvar).properties(
        width=600, height=400
    ).configure_view(
        strokeWidth=0
    ).interactive()

    st.altair_chart(chart, use_container_width=True)


def _validar_percentual_alocacao(percentuais):
        soma_percentuais = sum(percentuais)
        if soma_percentuais != 100:
            st.error(f"""A soma dos percentuais é {soma_percentuais}%. 
                    Ajuste os percentuais de alocação de modo que não ultrapasse 100%.""")
            return False
        return True


def _customize_color_perc_contribuicao(percentual):
    if percentual <= 10:
        return "#3F6060"
    elif 10 < percentual < 50:
        return "#EAAB46"
    else:
        return "#AF4F45"


def _customize_text_var_metric(var_dict):
    """
    Exibe texto explicativo e conclusão sobre o VaR dinamicamente dentro de st.expander().
    """

    with st.expander("Insights", icon="🧠"):
        carteira_teorica = "Carteira Teórica"
        var_teorica = var_dict.get(carteira_teorica, 0)
        carteira_maior_var = max(var_dict, key=var_dict.get)

        st.markdown(f"""
        O VaR da **{carteira_teorica}** é de {var_teorica:.2%}. Isso significa que, com **95% de confiança**, 
        a perda máxima esperada da {carteira_teorica} não deve exceder **{var_teorica:.2%}** do seu valor investido.
        A **{carteira_maior_var}** apresenta o maior VaR, indicando um **maior risco de perdas** em comparação com as outras carteiras. 
        Isso significa que a **{carteira_maior_var} pode experimentar perdas mais significativas** em cenários adversos.
        """)


def _customize_text_cvar_metric(cvar_dict):
    """
    Exibe texto explicativo e conclusão sobre o CVaR dinamicamente dentro de st.expander().
    """

    with st.expander("Insights", icon="🧠"):
        carteira_teorica = "Carteira Teórica"
        cvar_teorica = cvar_dict.get(carteira_teorica, 0)
        carteira_maior_cvar = max(cvar_dict, key=cvar_dict.get)

        st.markdown(f"""
        O CVaR da **{carteira_teorica}** é de {cvar_teorica:.2%}. Isso significa que, com **95% de confiança**, 
        a perda média esperada da {carteira_teorica}, caso as perdas excedam o VaR, é de **{cvar_teorica:.2%}** do seu valor investido.
        A **{carteira_maior_cvar}** apresenta o maior CVaR, indicando um **maior risco de perdas extremas** em comparação com as outras carteiras. 
        Isso significa que a {carteira_maior_cvar} pode experimentar **perdas significativamente maiores em cenários adversos**, superando o VaR.
        """)


def _customize_text_calmar_metric(calmar_cr, calmar_ct, retorno_anualizado_cr, retorno_anualizado_ct):
    """
    Gera o texto personalizado para a comparação dos Calmar Ratios.
    """

    proporcao_retornos = int(retorno_anualizado_ct / retorno_anualizado_cr)

    if calmar_cr > calmar_ct:
        text = f"""
            **Desempenho Ajustado ao Risco (Calmar Ratio):**
            A carteira real apresentou um Calmar Ratio de **{calmar_cr:.2f}**, superando a carteira teórica, que registrou um Calmar Ratio de **{calmar_ct:.2f}**.
            Isso indica que, para cada unidade de risco (medida pelo drawdown máximo), a carteira real gerou um retorno anualizado maior do que a carteira teórica.
            Em termos práticos, isso sugere que a estratégia da **carteira real foi mais eficiente em maximizar os retornos enquanto controlava as perdas máximas**.
            """
    elif calmar_ct > calmar_cr:
        text = f"""
            **Desempenho Ajustado ao Risco (Calmar Ratio):**
            A carteira teórica apresentou um Calmar Ratio de **{calmar_ct:.2f}**, superando a carteira real, que registrou um Calmar Ratio de **{calmar_cr:.2f}**.
            Isso indica que, para cada unidade de risco (medida pelo drawdown máximo), a carteira teórica gerou um retorno anualizado maior do que a carteira real.
            Em termos práticos, isso sugere que a estratégia da **carteira teórica foi mais eficiente em maximizar os retornos enquanto controlava as perdas máximas**.
            """
    else:
        text = f"""
            **Desempenho Ajustado ao Risco (Calmar Ratio):**
            As carteiras real e teórica apresentaram o mesmo Calmar Ratio de **{calmar_cr:.2f}**.
            Isso indica que ambas as carteiras geraram o mesmo retorno anualizado para cada unidade de risco (medida pelo drawdown máximo).
            """

    return text


def _customize_text_cenario_stress(variacao_percentual, mdd_real, mdd_teorico, mdd_real_stress, mdd_teorico_stress):
    """
    Gera o texto personalizado para a análise de cenário de stress.
    """

    if variacao_percentual < 0:
        if mdd_real_stress > mdd_real and mdd_teorico_stress > mdd_teorico:
            texto = f"""
                [💡] **Impacto Significativo em Cenário de Queda:** Ambas as carteiras (Real e Teórica) apresentaram um aumento no drawdown máximo 
                        sob o cenário de queda de {variacao_percentual}%.
                Isso indica que **ambas as carteiras são vulneráveis a grandes variações negativas nos preços dos ativos**.
                Em um cenário de queda de {variacao_percentual}%, a **Carteira Real** poderia ter um drawdown de **até {mdd_real_stress:.2%}**
                    e a **Carteira Teórica** de **até {mdd_teorico_stress:.2%}**.
            """
        elif mdd_real_stress > mdd_real:
            texto = f"""
                [💡] **Vulnerabilidade da Carteira Real em Cenário de Queda:** A Carteira Real apresentou um aumento significativo 
                        no drawdown máximo sob o cenário de queda de {variacao_percentual}%.
                Isso sugere que a **Carteira Real é mais sensível a grandes variações negativas nos preços dos ativos do que a Carteira Teórica**, 
                indicando uma **assimetria de risco maior para a Carteira Real em cenários de baixa**.
                Em um cenário de queda de {variacao_percentual}%, a Carteira Real poderia ter um drawdown de até {mdd_real_stress:.2%}.
            """
        elif mdd_teorico_stress > mdd_teorico:
            texto = f"""
                [💡] **Vulnerabilidade da Carteira Teórica em Cenário de Queda:** A Carteira Teórica apresentou um aumento significativo no drawdown máximo sob o cenário de queda de {variacao_percentual}%.
                Isso sugere que a **Carteira Teórica é mais sensível a grandes variações negativas nos preços dos ativos do que a Carteira Real**, 
                indicando uma **assimetria de risco maior para a Carteira Teórica em cenários de baixa**.
                Em um cenário de queda de {variacao_percentual}%, a Carteira Teórica poderia ter um drawdown de até {mdd_teorico_stress:.2%}.
            """
        else:
            texto = f"""
                [💡] **Resiliência em Cenário de Queda:** Ambas as carteiras (Real e Teórica) mantiveram seus drawdowns máximos 
                        relativamente estáveis sob o cenário de queda de {variacao_percentual}%.
                Isso indica que **ambas as carteiras são resilientes a grandes variações negativas nos preços dos ativos**.
            """
    else:
        if mdd_real_stress < mdd_real and mdd_teorico_stress < mdd_teorico:
            texto = f"""
                [💡] **Potencial de Ganho em Cenário de Alta:** Ambas as carteiras (Real e Teórica) apresentaram uma 
                        **redução no drawdown máximo sob o cenário de alta de {variacao_percentual}%**.
                Isso indica que ambas as carteiras capturam o momento de alta.
                Em um cenário de alta de {variacao_percentual}%, a **Carteira Real** poderia ter um drawdown 
                de **até {mdd_real_stress:.2%}** e a **Carteira Teórica** de **até {mdd_teorico_stress:.2%}**.
            """
        elif mdd_real_stress < mdd_real:
            texto = f"""
                [💡] **Potencial de Ganho da Carteira Real em Cenário de Alta:** A **Carteira Real** apresentou uma redução 
                        significativa no drawdown máximo sob o cenário de alta de {variacao_percentual}%.
                Isso sugere que a **Carteira Real captura o momento de alta de forma mais eficiente do que a Carteira Teórica**, 
                indicando uma **assimetria de risco maior para a Carteira Real em cenários de alta**.
                Em um cenário de alta de {variacao_percentual}%, a Carteira Real poderia ter um drawdown de até {mdd_real_stress:.2%}.
            """
        elif mdd_teorico_stress < mdd_teorico:
            texto = f"""
                [💡] **Potencial de Ganho da Carteira Teórica em Cenário de Alta:** A **Carteira Teórica*** apresentou uma redução 
                        significativa no drawdown máximo sob o cenário de alta de {variacao_percentual}%.
                Isso sugere que a **Carteira Teórica captura o momento de alta de forma mais eficiente do que a Carteira Real**, 
                indicando uma **assimetria de risco maior para a Carteira Teórica em cenários de alta**.
                Em um cenário de alta de {variacao_percentual}%, a Carteira Teórica poderia ter um drawdown de até {mdd_teorico_stress:.2%}.
            """
        else:
            texto = f"""
                [💡] **Captura do Momento de Alta:** Ambas as carteiras (Real e Teórica) mantiveram seus drawdowns máximos 
                        relativamente estáveis sob o cenário de alta de {variacao_percentual}%.
                Isso indica que ambas as carteiras **capturam o momento de alta de forma equilibrada**.
            """

    return texto


def _customize_text_sharpe_index(sharpe_cr, sharpe_ct, taxa_livre_risco):
    """
    Gera o texto personalizado para o Sharpe Ratio.
    """
    text = "#### Sharpe\n"
    if sharpe_cr > sharpe_ct:
        text += f"""A **Carteira Real** tem um melhor custo-benefício (Sharpe: {sharpe_cr:.2f}) em relação à **Carteira Teórica** (Sharpe: {sharpe_ct:.2f}).
        Isso significa que, para cada unidade de risco assumida, a Carteira Real gerou um retorno {sharpe_cr - sharpe_ct:.2f} 
        unidades acima da taxa livre de risco de {taxa_livre_risco:.2%}, enquanto a Carteira Teórica gerou {sharpe_ct:.2f} unidades.\n\n"""
    elif sharpe_ct > sharpe_cr:
        text += f"""A **Carteira Teórica** tem um melhor custo-benefício (Sharpe: {sharpe_ct:.2f}) em relação à **Carteira Real** (Sharpe: {sharpe_cr:.2f}). 
        Isso significa que, para cada unidade de risco assumida, a Carteira Teórica gerou um retorno {sharpe_ct - sharpe_cr:.2f} 
        unidades acima da taxa livre de risco de {taxa_livre_risco:.2%}, enquanto a Carteira Real gerou {sharpe_cr:.2f} unidades.\n\n"""
    else:
        text += f"""As duas carteiras têm o mesmo custo-benefício (Sharpe: {sharpe_cr:.2f}). 
        Ambas geraram {sharpe_cr:.2f} unidades de retorno para cada unidade de risco acima da taxa livre de risco de {taxa_livre_risco:.2%}.\n\n"""
    return text


def _customize_text_sortino_index(sortino_cr, sortino_ct, taxa_livre_risco):
    """
    Gera o texto personalizado para o Sortino Ratio.
    """
    text = "#### Sortino\n"
    if sortino_cr > sortino_ct:
        text += f"""A **Carteira Real** tem um melhor custo-benefício (Sortino: {sortino_cr:.2f}) em relação à **Carteira Teórica** (Sortino: {sortino_ct:.2f}). 
        Isso significa que, para cada unidade de risco de perda assumida, a Carteira Real gerou um retorno {sortino_cr - sortino_ct:.2f} 
        unidades acima da taxa livre de risco de {taxa_livre_risco:.2%}, enquanto a Carteira Teórica gerou {sortino_ct:.2f} unidades.\n\n"""
    elif sortino_ct > sortino_cr:
        text += f"""A **Carteira Teórica** tem um melhor custo-benefício (Sortino: {sortino_ct:.2f}) em relação à **Carteira Real** (Sortino: {sortino_cr:.2f}). 
        Isso significa que, para cada unidade de risco de perda assumida, a Carteira Teórica gerou um retorno {sortino_ct - sortino_cr:.2f} 
        unidades acima da taxa livre de risco de {taxa_livre_risco:.2%}, enquanto a Carteira Real gerou {sortino_cr:.2f} unidades.\n\n"""
    else:
        text += f"""As duas carteiras têm o mesmo custo-benefício (Sortino: {sortino_cr:.2f}). 
        Ambas geraram {sortino_cr:.2f} unidades de retorno para cada unidade de risco de perda acima da taxa livre de risco de {taxa_livre_risco:.2%}.\n\n"""
    return text


def _customize_text_beta_index(beta_cr, beta_ct, ticker_referencia, indices_referencia):
    """
    Gera o texto personalizado para o Beta.
    """
    text = f"#### Beta em relação ao {indices_referencia[ticker_referencia]}\n"
    text += f"""O beta mede a sensibilidade da carteira em relação ao {indices_referencia[ticker_referencia]}.
    \n* **Beta < 1**: A carteira **é menos volátil** que o {indices_referencia[ticker_referencia]}.
    \n* **Beta = 1**: A carteira tem a **mesma volatilidade** que o {indices_referencia[ticker_referencia]}.
    \n* **Beta > 1**: A carteira é **mais volátil** que o {indices_referencia[ticker_referencia]}.\n
    """
    if beta_cr is not None and beta_ct is not None:
        if beta_cr < 0.5 and beta_ct < 0.5:
            text += f"""
            \n[💡] **Beta Muito Baixo:** Ambas as carteiras apresentam um beta inferior a 0.5 em relação ao índice {indices_referencia[ticker_referencia]}, 
            indicando baixa sensibilidade às variações do mercado. o que significa que ambas as carteiras demonstram baixa volatilidade e
            tendem a ser menos sensíveis às oscilações do {indices_referencia[ticker_referencia]}. 
            **Cenários:** Em cenários de alta ou baixa do mercado, espera-se que as carteiras apresentem variações mínimas em relação ao índice de referência.\n"""
        elif beta_cr < 1 and beta_ct < 1:
            text += f"""\n[💡] **Beta Abaixo de 1:** Ambas as carteiras possuem um beta menor que 1 em relação ao índice {indices_referencia[ticker_referencia]},
            sugerindo que são menos voláteis que o mercado. o que significa que ambas as carteiras são menos sensíveis às oscilações do {indices_referencia[ticker_referencia]},
            com a Carteira Teórica (beta = {beta_ct:.2f}) demonstrando menor volatilidade em comparação com a Carteira Real (beta = {beta_cr:.2f}). 
            **Cenários:** Em **cenários de alta** do mercado, espera-se que as carteiras **valorizem menos** que o índice de referência. 
            Em **cenários de baixa**, espera-se que as carteiras desvalorizem menos que o índice de referência.\n"""
        elif beta_cr > 1 and beta_ct > 1:
            text += f"""\n[💡] **Beta Acima de 1:** Ambas as carteiras apresentam um beta superior a 1 em relação ao índice {indices_referencia[ticker_referencia]},
            indicando maior volatilidade que o mercado. Isto significa que ambas as carteiras são mais sensíveis às oscilações do {indices_referencia[ticker_referencia]},
            com a Carteira Real (beta = {beta_cr:.2f}) demonstrando significativamente maior volatilidade em comparação com a Carteira Teórica (beta = {beta_ct:.2f}). 
            **Cenários:** Em **cenários de alta** do mercado, espera-se que as carteiras **valorizem mais** que o índice de referência. 
            Em **cenários de baixa**, espera-se que as carteiras **desvalorizem mais** que o índice de referência.\n"""
        else:
            text += f"""\n[💡] **Betas Diferentes:** A Carteira Real (beta = {beta_cr:.2f}) apresenta um beta significativamente maior que a Carteira Teórica (beta = {beta_ct:.2f}) em relação 
            ao índice {indices_referencia[ticker_referencia]} o que significa que a Carteira Real é consideravelmente mais 
            volátil e sensível às variações do {indices_referencia[ticker_referencia]} em comparação com a Carteira Teórica. 
            **Cenários:** A Carteira Real tende a **amplificar as variações do mercado**, enquanto a Carteira Teórica **tende a atenuá-las**.\n"""
    return text


def _customize_help_show_text_sharpe():
    """
        Retorna a explicação sobre o Índice de Sharpe.
    """
    text = """
        O Índice de Sharpe mede o retorno ajustado ao risco de um investimento. 
        Ele calcula o excesso de retorno (retorno acima da taxa livre de risco) por unidade de risco (volatilidade).

        **Interpretação:**

        * Quanto maior o Índice de Sharpe, melhor o desempenho do investimento em relação ao risco.
        * Um Sharpe positivo indica que o investimento gerou retorno acima da taxa livre de risco.
        * Um Sharpe negativo indica que o investimento não compensou o risco assumido.

        **Para investidores:**

        O Índice de Sharpe ajuda a comparar diferentes investimentos e escolher aqueles que oferecem o 
        melhor retorno para o nível de risco desejado.
    """
    return text


def _customize_help_show_text_var():
    """
    Retorna a explicação sobre o Value at Risk (VaR).
    """
    text = """
        O Value at Risk (VaR) é uma medida estatística que quantifica a **perda potencial máxima que uma carteira pode sofrer** em um determinado período de tempo, com um certo nível de confiança. 
        Neste caso, estamos considerando um nível de confiança de 95%,
        o que significa que temos **95% confiança de que as perdas não excedam o valor do VaR**.
    """

    return text


def _customize_help_show_text_cvar():
    """
    Retorna a explicação sobre o Conditional Value at Risk (CVaR).
    """
    text = """
        O Conditional Value at Risk (CVaR), é uma medida estatística que quantifica a perda média esperada de uma carteira, 
        dado que a perda excedeu o Value at Risk (VaR) 
        em um determinado período de tempo e com um certo nível de confiança. 
        Em outras palavras, o CVaR fornece uma **estimativa da magnitude das perdas que podem ocorrer nos piores cenários**, 
        além do VaR. 
        Neste caso, estamos considerando um **nível de confiança de 95%**,
        o que significa que o **CVaR representa a perda média esperada nos 5% piores cenários**.
    """
    return text


def _customize_help_show_text_hhi():
    """
    Retorna a explicação sobre o Índice de Herfindahl-Hirschman (HHI).
    """
    text = """
        O Índice de Herfindahl-Hirschman (HHI) mede a concentração de uma carteira. 
        Ele é calculado como a soma dos quadrados dos pesos de cada ativo na carteira.

        **Interpretação:**

        * **Valores mais altos do HHI (próximos de 1)** indicam uma carteira mais concentrada, onde uma pequena quantidade de ativos detém uma grande parte do peso total. Isso sugere um maior risco, pois o desempenho da carteira torna-se mais dependente de um número limitado de ativos.
        * **Valores mais baixos do HHI (próximos de 0)** indicam uma carteira mais diversificada, onde o peso total é distribuído de forma mais uniforme entre vários ativos. Isso sugere um menor risco, pois o desempenho da carteira torna-se menos dependente de ativos individuais.
    """
    return text


def get_retorno_acumulado_ativo_carteira(dados_retornos):
    """
    Calcula o retorno acumulado dos ativos.
    """
    if dados_retornos.empty:
        return pd.DataFrame()
    
    retorno_acumulado = (1 + dados_retornos).cumprod()
    return retorno_acumulado


def get_retorno_carteira_diario(dataset, perc):
    """
    Calcula e retorna o retorno diário da carteira.
    """
    percentuais_aloc_normalizados = np.array(perc) / 100
    retorno_carteira_diario = (dataset * percentuais_aloc_normalizados).sum(axis=1)
    return retorno_carteira_diario


def get_retorno_carteira_anualizado(retornos_diarios, period=252):
    """
    Obtém o retorno anualizado da carteira.
    """
    # retorno_anualizado = (1 + retornos_diarios.mean()) ** (period * horizonte_anos) - 1
    # retorno_anualizado = np.mean(retornos_diarios) * period
    retorno_anualizado = (1 + retornos_diarios.mean()) ** period - 1
    return  retorno_anualizado


@st.cache_data
def get_retornos_mercado(retorno_carteira_diario, indice_referencia):
    """
    Obtém os retornos do mercado para o período específico do cálculo do beta, sem alterar métodos existentes.
    """
    start_date = retorno_carteira_diario.index.min()
    end_date = retorno_carteira_diario.index.max()
    dados_mercado = yf.download(indice_referencia, start=start_date, end=end_date)

    if dados_mercado.empty:
        st.error(f"Nenhum dado encontrado para {indice_referencia} no período especificado ‼️")
        return None

    if 'Adj Close' in dados_mercado:
        dados_mercado = dados_mercado['Adj Close']
    elif 'Close' in dados_mercado:
        dados_mercado = dados_mercado['Close']
    else:
        st.error(f"Nenhuma das colunas 'Adj Close' ou 'Close' está disponível no DataFrame.")
        return None

    retornos_mercado = dados_mercado.pct_change().dropna()
    retornos_mercado = retornos_mercado.squeeze()

    return retornos_mercado


def get_indice_sharpe(retorno_anualizado, volatilidade_anualizada, taxa_livre_risco=0.0):
    """
    Calcula o indice de Sharpe. 
    O indice de sharpe calcula o retorno ajustado ao risco 
    medindo o excesso de retorno (acima da taxa livre de risco) por unidade de risco.
    """
    sharpe_ratio = (retorno_anualizado - taxa_livre_risco) / volatilidade_anualizada
    return sharpe_ratio


def get_indice_sortino(retorno_anualizado, volatilidade_negativa_anualizada, taxa_livre_risco=0.0):
    """
    Calcula o indice de Sortino. 
    O indice de sortino considera apenas a volatilidade 
    negativa trazendo uma visão de risco de queda.
    """
    if volatilidade_negativa_anualizada == 0:
        return 0
    sortino_ratio = (retorno_anualizado - taxa_livre_risco) / volatilidade_negativa_anualizada
    return sortino_ratio


def get_value_at_risk(retornos_diarios, nivel_confianca=0.95, period=252):
    """
    Calcula o Value at Risk (VaR) anualizado.
    """
    var_diario = np.percentile(retornos_diarios, (1 - nivel_confianca) * 100)
    var_anualizado = var_diario * np.sqrt(period)
    return -var_anualizado


def get_value_beta(retornos_carteira, retornos_mercado):
    """
    Calcula o beta da carteira em relação ao mercado.
    """
    covariancia = retornos_carteira.cov(retornos_mercado)
    variancia_mercado = retornos_mercado.var()

    beta = covariancia / variancia_mercado

    return beta


def get_value_mdd_drawdowns_picos(retornos):
    """
    Calcula o Maximum Drawdown (MDD), os drawdowns e os picos de uma série de retornos.
    O Maximum Drawdown (MDD) representa a maior queda percentual acumulada 
    do valor de um ativo ou carteira em relação ao seu pico máximo,
    fornecendo uma medida do risco de perda máxima durante um determinado período.
    """
    valor_acumulado_carteira = (1 + retornos).cumprod()
    picos = valor_acumulado_carteira.cummax()
    drawdowns = (valor_acumulado_carteira - picos) / picos
    mdd = drawdowns.min()
    return mdd, drawdowns, picos


def get_value_index_hhi(percentuais):
    """
    Calcula o Índice de Herfindahl-Hirschman.
    """
    return np.sum(np.array(percentuais)**2)


def get_value_index_shannon(percentuais):
    """
    Calcula o Índice de Shannon.
    """
    return scipy.stats.entropy(percentuais)


def calcular_beta_carteira(retorno_carteira_diario, dataset_diarios_alinhado, percentuais, indice_referencia, tratar_outliers=False):
    """
    Calcula o beta para uma carteira.
    """
    try:
        # Alinhamento dos DataFrames
        retorno_carteira = retorno_carteira_diario.reindex(retorno_carteira_diario.index).ffill()
        retornos_mercado = get_retornos_mercado(retorno_carteira_diario, indice_referencia)

        if retornos_mercado is None:
            return None  

        retorno_carteira = retorno_carteira.reindex(retornos_mercado.index).ffill()
        retornos_mercado = retornos_mercado.reindex(retorno_carteira.index).ffill()

        beta = get_value_beta(retorno_carteira, retornos_mercado)

        return beta
    except Exception as e:
        st.error(f"Erro ao calcular o beta: {e}")
        return None


def get_conditional_value_at_risk(retornos_diarios, nivel_confianca=0.95, period=252):
    """
    Calcula o Conditional Value at Risk (CVaR) anualizado.
    """

    var_diario = np.percentile(retornos_diarios, (1 - nivel_confianca) * 100)
    cvar_diario = retornos_diarios[retornos_diarios <= var_diario].mean()
    cvar_anualizado = cvar_diario * np.sqrt(period)

    return -cvar_anualizado


def get_value_calmar_ratio(retornos_diarios, retorno_anualizado):
        """
        Calcula o índice Calmar para uma série de retornos diários.
        """
        # Drawdown máximo
        mdd, _, _ = get_value_mdd_drawdowns_picos(retornos_diarios)

        if mdd == 0:
            return 0  
        calmar_ratio = retorno_anualizado / abs(mdd)

        return calmar_ratio


def get_backtest_carteira(ativos_validos, percentuais, period):
    """
    Realiza um backtesting simplificado da carteira.
    """
    VALOR_INICIAL = 100
    VAR_VALUE = 5

    dados_retornos = get_dados_ativos_diarios(ativos_validos, period)
    retornos_carteira = get_retorno_carteira_diario(dados_retornos, percentuais)
    valor_carteira = (1 + retornos_carteira).cumprod() * VALOR_INICIAL

    # Cálculo das métricas de risco
    volatilidade = retornos_carteira.std()
    sharpe = retornos_carteira.mean() / volatilidade
    var = np.percentile(retornos_carteira, VAR_VALUE) 
    cvar = retornos_carteira[retornos_carteira <= var].mean() 

    resultados = pd.DataFrame({
        'Retorno': retornos_carteira,
        'Valor': valor_carteira,
        'Volatilidade': volatilidade,
        'Sharpe': sharpe,
        'VaR': var,
        'CVaR': cvar
    })

    return resultados


def alinhar_dataframes_reindex(dataset_cr, dataset_ct):
    """
    Alinha os DataFrames pelo período de tempo em comum usando reindexação.
    """
    data_inicio_comum = max(dataset_cr.index.min(), dataset_ct.index.min())
    data_fim_comum = min(dataset_cr.index.max(), dataset_ct.index.max())

    datas_comuns = pd.date_range(data_inicio_comum, data_fim_comum, freq='D')

    dataset_cr_alinhado = dataset_cr.reindex(datas_comuns)
    dataset_ct_alinhado = dataset_ct.reindex(datas_comuns)

    if not dataset_cr_alinhado.isnull().all().all() and not dataset_ct_alinhado.isnull().all().all():
        dataset_cr_alinhado = dataset_cr_alinhado.ffill()
        dataset_ct_alinhado = dataset_ct_alinhado.ffill()

    return dataset_cr_alinhado, dataset_ct_alinhado


def calcular_exibir_metricas_risco_retorno(dataset_diarios, percentuais,
                             tipo_carteira, prefixo_chave):
    
    """"
    Calcula e exibe métricas de risco e retorno para uma carteira de investimentos,
    incluindo o Índice de Sharpe, Índice de sortino, Value At Risk, 
    volatilidade anualizada, retorno anualizado e taxa livre de risco (Selic).
    """
    
    if not dataset_diarios.empty:

        retornos_diarios = get_retorno_carteira_diario(
            dataset_diarios, percentuais
        )

        retorno_anualizado = get_retorno_carteira_anualizado(
            retornos_diarios, period=252
        )
        volatilidade_anualizada = get_volatilidade_desvio_padrao_carteira(
            dataset_diarios, percentuais, period=252
        )
        volatilidade_negativa_anualizada = get_volatilidade_negativa_carteira(
            dataset_diarios, percentuais
        )
        
        # end_date_selic = dataset_diarios.index[-1].to_pydatetime().date()
        # begin_date = dataset_diarios.index[0].to_pydatetime().date()
        codigo_selic = 1178

        # st.write(begin_date.strftime("%d/%m/%Y"))
        # st.write(end_date_selic.strftime("%d/%m/%Y"))

        # df_selic = get_dados_bcb(
        #     codigo_selic,
        #     begin_date.strftime("%d/%m/%Y"),
        #     end_date_selic.strftime("%d/%m/%Y")
        # )

        df_selic = get_dados_bcb(codigo_selic)

        if df_selic is not None and not df_selic.empty:
            taxa_livre_risco = df_selic["SELIC"].mean()
        else:
            st.warning("Não foi possível obter dados da taxa Selic para o período selecionado.")
            return None
        
        sharpe_ratio = get_indice_sharpe(
            retorno_anualizado, volatilidade_anualizada, taxa_livre_risco
        )
        
        sortino_ratio = get_indice_sortino(
            retorno_anualizado, volatilidade_negativa_anualizada, taxa_livre_risco
        )

        retorno_carteira_diario = get_retorno_carteira_diario(
            dataset_diarios, percentuais
        )

        var_anualizado = get_value_at_risk(
            retorno_carteira_diario, nivel_confianca=0.95
        )

        cvar_anualizado = get_conditional_value_at_risk(
            retorno_carteira_diario, nivel_confianca=0.95
        )

        st.session_state[f"sharpe_ratio_{prefixo_chave}"] = sharpe_ratio
        st.session_state[f"sortino_ratio_{prefixo_chave}"] = sortino_ratio
        st.session_state[f"var_anualizado_{prefixo_chave}"] = var_anualizado
        st.session_state[f"cvar_anualizado_{prefixo_chave}"] = cvar_anualizado
        st.session_state[f"volatilidade_{prefixo_chave}"] = volatilidade_anualizada
        st.session_state[f"retorno_anualizado_{prefixo_chave}"] = retorno_anualizado
        st.session_state[f"retorno_diario_{prefixo_chave}"] = retornos_diarios
        st.session_state["taxa_livre_risco"] = taxa_livre_risco 
        st.session_state[f"percentuais_{prefixo_chave}"] = percentuais 
        st.session_state[f"dataset_diarios_{prefixo_chave}"] = dataset_diarios 

        return sharpe_ratio, sortino_ratio, var_anualizado, cvar_anualizado
    else:
        st.warning("Nenhum dado histórico disponível para os ativos válidos para calcular o Índice de Sharpe.")
        return None


def configurar_carteira(tipo_carteira, prefixo_chave):
    st.write(f"#### {tipo_carteira}")

    tickers_acoes = obter_tickers_acoes()
    ticker_fiis = obter_tickers_fiis()

    ativos = {
        "Ações": tickers_acoes,
        "Fundos Imobiliários": ticker_fiis,
        "Todos": tickers_acoes + ticker_fiis
    }

    tipo_ativo = st.selectbox("Selecione o tipo de ativo",
                              ["Todos", "Ações", "Fundos Imobiliários"],
                              key=f"{prefixo_chave}_tipo_ativo")

    if f"{prefixo_chave}_ativo_selected" not in st.session_state:
        st.session_state[f"{prefixo_chave}_ativo_selected"] = []

    ativo_selected = st.multiselect("Buscar ativos (informe o ticker):", 
                                    ativos[tipo_ativo], 
                                    key=f"{prefixo_chave}_busca_ativo"
                                    )
 
    periods_en_to_pt = {
        "6mo": "6 meses",
        "1y": "1 ano",
        "2y": "2 anos",
        "5y": "5 anos",
        "10y": "10 anos",
        "max": "Máximo"
    }

    default_index = list(periods_en_to_pt.values()).index("Máximo")

    dt_period = st.selectbox("Selecione o período histórico", 
                             list(periods_en_to_pt.values()),
                             index=default_index,
                             key=f"{prefixo_chave}_historico_ativo"
                             )    
    
    for en, pt in periods_en_to_pt.items():
        if pt == dt_period:
            dt_period = en
            break

    avancar_clicked = st.button("Avançar", 
                                key=f"{prefixo_chave}_bt_avancar_ativo_selected")

    if avancar_clicked:
        st.session_state[f"{prefixo_chave}_ativo_selected"] = ativo_selected
        st.session_state[f"{prefixo_chave}_confirmar_clicked"] = False

    if st.session_state[f"{prefixo_chave}_ativo_selected"]:
        ativos_validos_selecionados = get_verifica_ativos_validos(
            prefixo_chave + "_ativo_selected", dt_period
        )

        percentuais = []
        cleft, cright =  st.columns(2)
        for index, ativo in enumerate(ativos_validos_selecionados):
            col = cleft if index%2 ==0 else cright
            with col: 
                ativo = ativo.replace(".SA", "")
                percentual = st.number_input(f"{ativo} (% Alocação)",
                                            min_value=0,
                                            max_value=100,
                                            value=10,
                                            step=1,
                                            placeholder="Pressione enter",
                                            help="Informe o percentual de alocação do ativo",
                                            key=f"{prefixo_chave}_perc_aloc_{ativo}")
                percentuais.append(percentual)
        
        if percentuais:
            confirmar_habilitado = _validar_percentual_alocacao(percentuais)
            confirmar_clicked = st.button("Confirmar", 
                                        key=f"{prefixo_chave}_button_confirmar", 
                                        disabled=not confirmar_habilitado
                                        )

            if confirmar_clicked:
                st.session_state[f"ativos_validos_selecionados_{prefixo_chave}"] = ativos_validos_selecionados
                st.session_state[f"dt_period_{prefixo_chave}"] = dt_period
                st.session_state[f"percentuais_{prefixo_chave}"] = percentuais
                st.session_state[f"df_selecionados_{prefixo_chave}"] = pd.DataFrame({
                    "Ativo": ativos_validos_selecionados,
                    "Percentual de Alocação": percentuais
                })
                st.session_state[f"{prefixo_chave}_confirmar_clicked"] = True

            if st.session_state.get(f"{prefixo_chave}_confirmar_clicked", False):
                tab1, tab2, tab3 = st.tabs(
                    ["⭕ Alocação", "📈 Série Histórica", "📈📉 Performance"]
                )

                with tab1:
                    df_selecionados = st.session_state[f"df_selecionados_{prefixo_chave}"]
                    st.write(f"#### Balanceamento dos ativos da {tipo_carteira}")
                    plot_pie_chart(df_selecionados, "Percentual de Alocação", "Ativo", "viridis")
                
                with tab2:
                    # Gráfico dos preços de fechamento
                    dados = get_dados_raw_yfinance(
                        ativos_validos=ativos_validos_selecionados, 
                        period=dt_period
                    )
                    st.write("#### Variação de preços dos ativos da carteira")
                    st.line_chart(dados["Adj Close"])

                with tab3: 
                    dados_retornos = get_dados_ativos_diarios(
                        ativos_validos=ativos_validos_selecionados, 
                        period=dt_period
                    )

                    retorno_acumulado = get_retorno_acumulado_ativo_carteira(dados_retornos)

                    df_retorno_acumulado = retorno_acumulado.reset_index().melt(
                        id_vars="Date", var_name="Ativo",
                        value_name="Retorno Acumulado"
                    )
                    st.write("#### Retorno acumulado dos ativos da carteira")
                    
                    plot_line_chart(df_retorno_acumulado, "Date", "Retorno Acumulado", "Ativo")

                with st.container():
                    
                    dataset_diarios = get_dados_ativos_diarios(
                        ativos_validos=ativos_validos_selecionados, 
                        period=dt_period).reindex(ativos_validos_selecionados, axis=1)
                    

                    if not dataset_diarios.empty:
                        
                        df_contribuicao = get_perc_contribuicao_ativos(dataset_diarios, percentuais)

                        tab1, tab2, tab3, tab4, tab5 = st.tabs(
                            ["✖️ Risco por ativo", "✖️ Volatilidade", "✖️ Risco/Retorno", "✖️ Correlação", "✖️ Diversificação"]
                        )
                        with tab1: 
                            st.write("#### Contribuição dos Ativos para o Risco da Carteira")
                            plot_horizontal_bar_chat(df_contribuicao, 
                                                    "Contribuição (%)", "Ativo", "Cor")
                            
                            st.warning(f"""
                                **💡 Fique ligado** 
                                 - O ativo com maior contribuição de risco é **{df_contribuicao.iloc[0]['Ativo']}** com {df_contribuicao.iloc[0]['Contribuição (%)']}% de contribuição.
                                 - A contribuição de risco indica o **impacto de cada ativo no risco total da carteira**. Ativos com alta contribuição podem exigir uma análise mais detalhada ou ajuste na alocação.
                                """)
                            
                        with tab2:
                            st.write('### Desvio Padrão Anualizado: Risco por Ativo') 
                            dataset_carteira_volatilidade = get_dataset_volatilidade_desvio_padrao_carteira(dataset_diarios)

                            plot_volatilidade_barras_horizontal(dataset_carteira_volatilidade)

                            st.warning(f"""
                                **💡 Fique ligado**
                                - O ativo com maior volatilidade é **{dataset_carteira_volatilidade.iloc[0]['Ticker']}** com {dataset_carteira_volatilidade.iloc[0]['Risco']:.2%}.
                                - O desvio padrão anualizado mede a volatilidade dos ativos. Ativos com alta volatilidade podem ser mais arriscados, mas também podem oferecer maiores retornos potenciais.
                                """)                
                        
                        with tab3: 
                            st.write('### Risco Retorno da Carteira')
                            retornos_diarios = get_retorno_carteira_diario(
                                dataset_diarios, percentuais
                            )
                            retorno_anualizado = get_retorno_carteira_anualizado(
                                retornos_diarios, period=252
                            )
                            volatilidade_anualizada = get_volatilidade_desvio_padrao_carteira(
                                dataset_diarios, percentuais)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    label="Retorno Anualizado",
                                    value=f"{retorno_anualizado * 100:.2f}%",
                                    delta=None,
                                    delta_color="normal",
                                    help="Retorno anualizado da carteira, calculado com base nos retornos diários."
                                )
                            with col2:
                                st.metric(
                                    label="Volatilidade Anualizada",
                                    value=f"{volatilidade_anualizada * 100:.2f}%",
                                    help="""Volatilidade anualizada da carteira, 
                                    calculada com base no desvio padrão dos retornos diários.""")
                                
                            with st.expander("Saber mais", icon="🔍"):
                                st.info("""
                                **Informações:**
                                - O retorno anualizado representa o **ganho percentual** da carteira em um ano.
                                - A volatilidade anualizada mede o risco da carteira, indicando a **variação dos retornos** ao longo do tempo.
                                """)

                            if volatilidade_anualizada > 0.2: 
                                st.warning(" ⚠️ A volatilidade da carteira é considerada **alta**. Considere revisar a alocação dos ativos.")
                        
                        with tab4:

                            if tipo_carteira == "Carteira Real":
                                plot_matriz_correlacao(dataset_diarios, tipo_carteira, 'redblue')
                            else:
                                plot_matriz_correlacao(dataset_diarios, tipo_carteira, 'redblue')

                            st.info(f"""
                                    O mapa de calor exibe o grau de correlação entre os ativos selecionados na {tipo_carteira}. 
                                        **Tons de azul indicam correlação negativa (diversificação)**, 
                                        **tons de vermelho indicam correlação positiva (risco)**.
                                    """)
                            with st.expander("Saber mais", icon="🔍"):
                                st.markdown(f"""
                                **Implicações da Correlação:**

                                - **Alta Correlação Positiva (Vermelho Forte)**: Indica que os ativos tendem a se mover na **mesma direção**. 
                                Isso **aumenta o risco da carteira**, pois uma queda em um ativo provavelmente será acompanhada por quedas em outros ativos.
                                Ideal para **estratégias de crescimento agressivo** em mercados de alta.

                                - **Alta Correlação Negativa (Azul Forte)** :Indica que os ativos tendem a se mover em **direções opostas**. 
                                Isso **reduz o risco da carteira**, pois as perdas em um ativo podem ser 
                                compensadas pelos ganhos em outros ativos.
                                Ideal para **estratégias de proteção e diversificação**.

                                - **Baixa Correlação (Branco):** Indica que os ativos têm pouca ou nenhuma relação entre si. 
                                Isso oferece um **equilíbrio entre risco e retorno**, 
                                permitindo que a carteira se beneficie de diferentes condições de mercado.
                                Ideal para estratégias de diversificação equilibrada.
                                """)

                        with tab5: 
                            st.write('#### Índices de Diversificação da Carteira')
                            
                            hhi = get_value_index_hhi(percentuais)
                            shannon = get_value_index_shannon(percentuais)
                            
                            col1, col2 = st.columns(2)

                            with col1:
                                st.metric(
                                    label="HHI",
                                    value=f"{hhi:.2f}",
                                    help="""
                                    Índice de Herfindahl-Hirschman (HHI) é uma medida que quantifica a concentração de uma carteira de investimentos. 
                                    Um **HHI mais baixo indica maior a diversificação**."""
                                )

                            with col2:
                                st.metric(
                                    label="Índice de Shannon",
                                    value=f"{shannon:.2f}",
                                    help="""O Índice de Shannon mede a entropia da distribuição dos pesos dos ativos. **Quanto maior o índice, maior a diversificação**."""
                                )

                            st.warning("""
                            **💡 Fique ligado**
                            - Avalie a **distribuição dos pesos** dos ativos na sua carteira.
                            - Considere adicionar ativos com **baixa correlação** para aumentar a diversificação.
                            """)

                        sharpe_ratio = calcular_exibir_metricas_risco_retorno(dataset_diarios, 
                                                    percentuais, 
                                                    tipo_carteira,
                                                    prefixo_chave
                                        )
                    
                        st.session_state[f"sharpe_ratio_{prefixo_chave}"] = sharpe_ratio

                    else:
                        st.warning("""Nenhum dado histórico disponível para os ativos válidos.""")


tab_home, tab_help = st.tabs(["Início", "Ajuda"])

# tab_home, tab_ia, tab_help = st.tabs(["Início", "Análise com IA", "Ajuda"])

with tab_home:
    with st.container():
        cleft, cright = st.columns(2, border=True)

        with cleft:
            configurar_carteira("Carteira Real", "cr")

        with cright:
            configurar_carteira("Carteira Teórica", "ct")

    # if st.button("Obter indices de risco"): 
    if st.session_state.get("retorno_diario_cr") is not None and st.session_state.get("retorno_diario_ct") is not None:

        dataset_diarios_cr = st.session_state["dataset_diarios_cr"]
        dataset_diarios_ct = st.session_state["dataset_diarios_ct"]
        percentuais_cr = st.session_state["percentuais_cr"]
        percentuais_ct = st.session_state["percentuais_ct"]
        sharpe_cr = st.session_state["sharpe_ratio_cr"]
        sharpe_ct = st.session_state["sharpe_ratio_ct"]
        taxa_livre_risco = st.session_state["taxa_livre_risco"]
        retorno_diario_cr = st.session_state["retorno_diario_cr"]
        retorno_diario_ct = st.session_state["retorno_diario_ct"]
        retorno_anualizado_cr = st.session_state["retorno_anualizado_cr"]
        retorno_anualizado_ct = st.session_state["retorno_anualizado_ct"]
        var_anualizado_cr = st.session_state["var_anualizado_cr"]
        var_anualizado_ct = st.session_state["var_anualizado_ct"]
        cvar_anualizado_cr = st.session_state["cvar_anualizado_cr"]
        cvar_anualizado_ct = st.session_state["cvar_anualizado_ct"]
        ativos_validos_selecionados_cr = st.session_state["ativos_validos_selecionados_cr"]
        ativos_validos_selecionados_ct = st.session_state["ativos_validos_selecionados_ct"]
        dt_period_cr = st.session_state["dt_period_cr"]
        dt_period_ct = st.session_state["dt_period_ct"]

        dataset_diarios_cr_alinhado, dataset_diarios_ct_alinhado = alinhar_dataframes_reindex(
            dataset_diarios_cr, dataset_diarios_ct
        )
        retorno_carteira_diario_cr = get_retorno_carteira_diario(dataset_diarios_cr_alinhado, percentuais_cr)
        retorno_carteira_diario_ct = get_retorno_carteira_diario(dataset_diarios_ct_alinhado, percentuais_ct)

        # @TODO: avaliar se entrará na aplicação os retornos pa
        # with st.container():
        #     st.write("### Evolução do valor investido ao longo do tempo")
        #     plot_retornos_acumulados(retorno_carteira_diario_cr, retorno_carteira_diario_ct)


        with st.container():

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Desempenho", "Drawdowns", "Stress Testing", "Risco e Retorno", "Distribuição de Retornos"])

            with tab1:
                calmar_cr = get_value_calmar_ratio(retorno_carteira_diario_cr, retorno_anualizado_cr)
                calmar_ct = get_value_calmar_ratio(retorno_carteira_diario_ct, retorno_anualizado_ct)
                
                cleft, ccenter, cright = st.columns([1, 1, 2], border=True)
                with cleft:
                    st.write("#### Carteira Real")
                    st.metric(
                        label="Calmar Ratio",
                        value=f"{calmar_cr:.2f}",
                        help="""Retorno ajustado ao risco, calculado com base no Calmar Ratio.
                        O Calmar Ratio mede o retorno de um investimento em relação ao seu drawdown máximo (a maior perda observada)."""
                    )
                with ccenter:
                    st.write("#### Carteira Teórica")
                    st.metric(
                        label="Calmar Ratio",
                        value=f"{calmar_ct:.2f}",
                        delta=f"{calmar_ct - calmar_cr:.2f}", 
                        delta_color="normal" if calmar_ct > calmar_cr else "inverse",
                        help="""Retorno ajustado ao risco, calculado com base no Calmar Ratio.
                        O Calmar Ratio mede o retorno de um investimento em relação ao seu drawdown máximo (a maior perda observada)."""
                    )

                with cright:
                    st.write("#### Avaliação das carteiras")
                    with st.expander("Insights", icon="🧠"):
                        texto_comparacao = _customize_text_calmar_metric(
                            calmar_cr, calmar_ct, retorno_anualizado_cr, retorno_anualizado_ct)
                        st.info(texto_comparacao)
  
            with tab2: 

                st.write('### Drawdown máximo das carteiras')
                try:
                    mdd_real, _, _ = get_value_mdd_drawdowns_picos(retorno_carteira_diario_cr)
                    mdd_teorico, _, _ = get_value_mdd_drawdowns_picos(retorno_carteira_diario_ct)
                except:
                    mdd_real, mdd_teorico = 0.1, 0.1

                min_slider = 5
                
                cleft, cright = st.columns(2)
                
                with cleft: 
                    limiar_real_pct = st.slider(
                            "Destacar drawdowns da **carteira real** a partir de:",  
                            min_value=min_slider, 
                            max_value=int(abs(mdd_real * 100)), 
                            value=min_slider, 
                            step=5, 
                            format="%d%%"
                        )
                    limiar_drawdown_cr = limiar_real_pct / 100

                    chart_area_cr = plot_area_drawdown_chart(retorno_carteira_diario_cr, 'Máxima Perda da Carteira Real', limiar_drawdown_cr)
                
                with cright:
                    limiar_teorico_pct = st.slider(
                        "Destacar drawdowns da **carteira teórica** a partir de:",
                        min_value=min_slider, 
                        max_value=int(abs(mdd_teorico * 100)), 
                        value=min_slider, 
                        step=1, 
                        format="%d%%"
                    )
                    limiar_drawdown_ct = limiar_teorico_pct / 100
                
                    chart_area_ct = plot_area_drawdown_chart(retorno_carteira_diario_ct, 'Máxima Perda da Carteira Teórica', limiar_drawdown_ct)

            with tab3: 
            
                st.info("""
                    ### Stress Testing e Análise de Assimetria de Risco
                    Uma carteira é mais sensível a quedas ou aumentos? 
                            Use o controle para simular variações percentuais e veja o impacto nos drawdowns. 
                            Variação negativa simula queda, positiva simula alta.
                """)
                                
                
                cleft, cright = st.columns(2)
                with cleft:
                    variacao_percentual = st.slider(
                        "Variação percentual nos preços dos ativos:",
                        min_value=-50, max_value=50, value=-20, step=5, format="%d%%"
                    )

                # if st.button("Simular Cenário de Estresse"):
                    # Simulação do cenário de estresse
                    retorno_carteira_diario_cr_stress = retorno_carteira_diario_cr * (1 + variacao_percentual / 100)
                    retorno_carteira_diario_ct_stress = retorno_carteira_diario_ct * (1 + variacao_percentual / 100)

                    # Cálculo do drawdown em cenário de estresse
                    mdd_real_stress, _, _ = get_value_mdd_drawdowns_picos(retorno_carteira_diario_cr_stress)
                    mdd_teorico_stress, _, _ = get_value_mdd_drawdowns_picos(retorno_carteira_diario_ct_stress)

                    col1, col2 = st.columns(2, border=True)
                    with col1:
                        st.metric(
                            label="Drawdown Máximo - Carteira Real (Stress)",
                            value=f"{mdd_real_stress:.2%}",
                            delta=f"{(mdd_real_stress - mdd_real):.2%}"
                        )

                    
                    with col2:
                        st.metric(
                            label="Drawdown Máximo - Carteira Teórica (Stress)",
                            value=f"{mdd_teorico_stress:.2%}",
                            delta=f"{(mdd_teorico_stress - mdd_teorico):.2%}"
                        )
                with cright:
                    st.write("#### Comparação do Drawdown em Cenário de Stress")
                    plot_drawdown_stress(mdd_real, mdd_teorico, mdd_real_stress, mdd_teorico_stress)
                    texto_analise = _customize_text_cenario_stress(
                    variacao_percentual, mdd_real, mdd_teorico, mdd_real_stress, mdd_teorico_stress)
                    st.warning(texto_analise)

            with tab4:

                with st.container():
                    st.write("### Indices de avaliação de risco")
                    cleft, cright = st.columns(2)

                    with cleft: 

                        sharpe_cr, sortino_cr, var_cr, cvar_cr = calcular_exibir_metricas_risco_retorno(
                            dataset_diarios_cr_alinhado, percentuais_cr, "Carteira Real", "cr"
                        )
                        sharpe_ct, sortino_ct, var_ct, cvar_ct = calcular_exibir_metricas_risco_retorno(
                            dataset_diarios_ct_alinhado, percentuais_ct, "Carteira Teórica", "ct"
                        )

                        indices_real = [sharpe_cr, sortino_cr]
                        indices_teorica = [sharpe_ct, sortino_ct]
                        nomes_indices = ["Indice Sharpe", "Indice Sortino"]
                    
                    plot_tabela_progress_column(indices_real, indices_teorica, nomes_indices)
                
                with st.container():         
                    st.write("### Análise de Beta")

                    indices_referencia = {
                        "^BVSP": "IBOV (Ibovespa)",
                        "IFIX": "IFIX (Índice de Fundos Imobiliários)",
                        "IDIV": "IDIV (Índice de Dividendos)"
                    }

                    indice_selecionado = st.selectbox(
                        "Selecione o índice de referência",
                        options=list(indices_referencia.values()),
                            help="""
                                A escolha correta do índice de referência é crucial para uma análise precisa do beta da sua carteira. 
                                O índice deve refletir as características dos ativos que você possui.

                                **Exemplos:**
                                - **Ibovespa (\\^BVSP):** Ideal para carteiras com ações de empresas de diversos setores.
                                - **IFIX (\\^IFIX):** Adequado para carteiras compostas por Fundos de Investimento Imobiliário (FIIs).
                                - **IDIV (\\^IDIV):** Recomendado para carteiras focadas em ações de empresas que pagam dividendos.
                            """
                    )

                    ticker_referencia = list(indices_referencia.keys())[list(indices_referencia.values()).index(indice_selecionado)]

                    beta_cr = calcular_beta_carteira(retorno_carteira_diario_cr, dataset_diarios_cr_alinhado, percentuais_cr, ticker_referencia)
                    beta_ct = calcular_beta_carteira(retorno_carteira_diario_ct, dataset_diarios_ct_alinhado, percentuais_ct, ticker_referencia)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if beta_cr is not None:
                            st.metric(
                                label=f"Beta - Carteira Real (Referência: {ticker_referencia})",
                                value=f"{beta_cr:.6f}",
                                help=f"""Medida da sensibilidade da carteira real em relação ao {ticker_referencia}. 
                                Quanto maior o beta, maior a volatilidade da carteira em relação ao mercado.""",
                            )
                    
                    with col2:
                        if beta_cr is not None:
                            st.metric(
                                label=f"Beta - Carteira Teórica (Referência: {ticker_referencia})",
                                value=f"{beta_ct:.6f}",
                                help=f"""Medida da sensibilidade da carteira teórica em relação ao {ticker_referencia}. 
                                Quanto maior o beta, maior a volatilidade da carteira em relação ao mercado.""",
                            )
                    
                    
                    with st.expander("Insights", icon="🧠"):
                        st.warning(_customize_text_sharpe_index(sharpe_cr, sharpe_ct, taxa_livre_risco))
                        st.warning(_customize_text_sortino_index(sortino_cr, sortino_ct, taxa_livre_risco))
                        st.info(_customize_text_beta_index(beta_cr, beta_ct,ticker_referencia, indices_referencia))
                    
                with st.container(): 

                    st.write("### Visualização de Risco vs. Retorno")
                    cleft, cright = st.columns(2)
                    with cleft: 
                        plot_risk_return(
                        dataset_diarios_cr, 
                        percentuais_cr, 
                        dataset_diarios_ct, 
                        percentuais_ct, sharpe_cr, sharpe_ct, index_name="Sharpe")
                    with cright:
                        plot_risk_return(
                            dataset_diarios_cr, 
                            percentuais_cr, 
                            dataset_diarios_ct, 
                            percentuais_ct, sortino_cr, sortino_ct, index_name="Sortino")
                    
            with tab5:
                st.markdown("### Análise da Distribuição de Retornos das Carteiras")


                with st.container():
                
                    medida_risco = st.radio(
                        "Selecione a medida de risco:",
                        ("VaR", "CVaR"),
                        index=0,
                        help="""
                        **VaR (Value at Risk):** Estima a perda máxima esperada em um determinado nível de confiança.

                        **CVaR (Conditional Value at Risk):** Estima a perda média que pode ocorrer caso a perda exceda o VaR.
                        """
                    )
                    
                    if medida_risco == "VaR":

                        plot_var_lollipop_chart({
                            "Carteira Real": var_anualizado_cr,
                            "Carteira Teórica": var_anualizado_ct
                        }, title = "Comparação do Value at Risk (VaR) entre as carteiras")

                        _customize_text_var_metric({
                            "Carteira Real": var_anualizado_cr,
                            "Carteira Teórica": var_anualizado_ct
                        })
                        
                    else:

                        plot_var_lollipop_chart({
                            "Carteira Real": cvar_anualizado_cr,
                            "Carteira Teórica": cvar_anualizado_ct
                        }, title = "Comparação do Value at Risk Condicional (CVaR) entre as carteiras")
                        
                        _customize_text_cvar_metric({
                            "Carteira Real": cvar_anualizado_cr,
                            "Carteira Teórica": cvar_anualizado_ct
                        })

                with st.container():
                    st.write("#### Distribuição de Cauda")
                    cleft, cright = st.columns(2)
                    with cleft:
                        plot_cauda_distribuicao(
                            retorno_diario_cr,
                            var_anualizado_cr,
                            cvar_anualizado_cr,
                            titulo="Carteira Real"
                        )
                    with cright:
                        plot_cauda_distribuicao(
                            retorno_diario_ct,
                            var_anualizado_ct,
                            cvar_anualizado_ct,
                            titulo="Carteira Teórica"
                        )
            
                with st.container():
                    resultados_cr = get_backtest_carteira(ativos_validos_selecionados_cr, percentuais_cr, dt_period_cr)
                    resultados_ct = get_backtest_carteira(ativos_validos_selecionados_ct, percentuais_ct, dt_period_ct)
                    plot_barras_metricas_marcadores(resultados_cr, resultados_ct)
                

# with tab_ia:
#     ia_page.main()






# Tab Ajuda 
with tab_help:
    with st.expander("Índice de Sharpe?", icon="❔"):
        st.write(_customize_help_show_text_sharpe())

    with st.expander("Value at Risk (VaR)", icon="❔"):
        st.markdown(_customize_help_show_text_var())

    with st.expander("Conditional Value at Risk (CVaR)", icon="❔"):
        st.markdown(_customize_help_show_text_cvar())

    with st.expander("Índice de Herfindahl-Hirschman (HHI)", icon="❔"):
        st.write(_customize_help_show_text_hhi())